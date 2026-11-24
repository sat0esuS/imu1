// s3_logger -- CoreS3 単体でレーダー(IWR6843AOPEVM)と内蔵IMU(BMI270)を同じ時計で記録する
//
// 構成
//   レーダー : CoreS3 の USB-C をホストにして EVM の CP2105 につなぐ（Type-C→micro-B ケーブル）
//              CFG 口に cfg を送り、データ口のバイト列を受ける
//   IMU      : CoreS3 内蔵の BMI270 を M5Unified で読む
//   保存先   : CoreS3 の microSD
//   時計     : すべて esp_timer（起動からの µs）。レーダーも IMU も同じ時計で打刻する
//
// 出力（PC の pc/logger.py と同じ形式。pc/ の解析スクリプトがそのまま使える）
//   <prefix>.dat         レーダーの生バイト列（Visualizer と同一形式）
//   <prefix>_frames.csv  t_host,frame,n_bytes
//   <prefix>_imu.csv     t_host,t_dev_us,gx,gy,gz,ax,ay,az
//   <prefix>_meta.json   記録条件と取りこぼし数
//   frames.csv の t_host : 記録開始からの秒（esp_timer）
//   imu.csv    の t_host : 読み出し時刻から逆算した各サンプルの時刻（esp_timer、記録開始からの秒）
//              の t_dev_us: BMI270 のサンプル番号 × 5000 µs（センサ自身の時計。間隔が正確）
//   pc/imu.py の load_imu は t_host ≈ a·t_dev + b を当てはめるので、そのまま使える
//
// IMU は BMI270 の FIFO（ジャイロ＋加速度、200 Hz）からまとめて読むので、取りこぼさない
//
// ファイル名は RTC の日時（s3_YYYYMMDD_hhmmss）。RTC は「このファームをビルドした時刻」で
// 初回起動時に合わせる（数十秒〜数分ずれる。ファイル名用）
//
// 操作
//   画面下部をタッチ：記録の開始／停止
//
// EVM のスイッチは PC で使うときと同じ（すべて OFF）にすること。

#include <M5Unified.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <atomic>
#include <vector>
#include <algorithm>
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/ringbuf.h"
#include "usb/usb_host.h"
#include "usb/cdc_acm_host.h"
#include "usb/vcp_cp210x.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"

static const char* TAG = "s3log";

// ===================================================================== 設定
static constexpr uint32_t BAUD_CFG  = 115200;
static constexpr uint32_t BAUD_DATA = 921600;
static constexpr double   BYTES_PER_SEC = 921600.0 / 10.0;   // 8N1 → 1 バイト 10 ビット
static constexpr uint32_t BOOT_WAIT_MS = 3000;                // EVM の起動待ち

// microSD（CoreS3 は LCD と SPI バスを共有。バスは M5GFX が初期化する）
static constexpr int SD_CS   = 4;
static constexpr int SD_SCLK = 36;
static constexpr int SD_MOSI = 37;
static constexpr int SD_MISO = 35;
static const char* MOUNT = "/sdcard";

// 送る cfg（imu1/cfg/corridor.cfg、廊下実験用、2026/08/21 Run1 と同一）
static const char* CFG_NAME = "corridor.cfg";
static const char* CFG_TEXT = R"CFG(
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 60 483 7 200 0 0 20 1 384 2000 0 0 158
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
chirpCfg 2 2 0 0 0 0 0 4
frameCfg 0 2 16 0 66.667 1 0
lowPower 0 0
guiMonitor -1 1 1 0 0 0 1
cfarCfg -1 0 2 8 4 3 0 15 1
cfarCfg -1 1 0 4 2 3 1 15 1
multiObjBeamForming -1 1 0.5
clutterRemoval -1 0
calibDcRangeSig -1 0 -5 8 256
extendedMaxVelocity -1 0
lvdsStreamCfg -1 0 0 0
compRangeBiasAndRxChanPhase 0.0 1 0 -1 0 1 0 -1 0 1 0 -1 0 1 0 -1 0 1 0 -1 0 1 0 -1 0
measureRangeBiasAndRxChanPhase 0 1.5 0.2
CQRxSatMonitor 0 3 19 125 0
CQSigImgMonitor 0 127 6
analogMonitor 0 0
aoaFovCfg -1 -90 90 -90 90
cfarFovCfg -1 0 0 12.00
cfarFovCfg -1 1 -0.61 0.61
calibData 0 0 0
sensorStart
)CFG";

// ===================================================================== 共有状態
static const uint8_t MAGIC[8] = {0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07};

// 表示用の統計（コールバック側で数える。記録の有無に関係なく動く）
static std::atomic<uint32_t> g_rxBytes{0}, g_rxFrames{0}, g_imuN{0};
static int g_magicPos = 0;

// 取りこぼし（リングバッファ満杯）
static std::atomic<uint32_t> g_dropRadar{0}, g_dropImu{0};
static std::atomic<uint32_t> g_imuSkipped{0};   // BMI270 FIFO が溢れて捨てたフレーム数

// USB 受信 → 書き込みタスク
static RingbufHandle_t g_rbRadar = nullptr;   // 要素 = RadarHdr + データ
static RingbufHandle_t g_rbImu   = nullptr;   // 要素 = ImuSample
struct RadarHdr { int64_t t_us; uint32_t len; };
struct ImuSample { int64_t t_us; int64_t t_dev_us; float g[3]; float a[3]; };

// cfg の応答
static char g_cfgLine[128];
static int g_cfgLen = 0;
static SemaphoreHandle_t g_cfgAck;
static std::atomic<bool> g_cfgError{false};
static char g_lastReply[64] = "";
static std::atomic<uint32_t> g_rxPort[2] = {0, 0};
static std::atomic<int> g_cfgIface{-1};

// 記録の開始／停止の要求（タッチ → 書き込みタスク）
static std::atomic<int> g_req{0};        // 1 = 開始, 2 = 停止
static char g_reqPrefix[64] = "";
static char g_reqIso[32] = "";
static std::atomic<bool> g_recording{false};
static std::atomic<bool> g_sdOk{false};
static char g_cfgStatus[40] = "";

// ===================================================================== 画面
static void status(int line, const char* fmt, ...) {
  char buf[64];
  va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
  M5.Display.fillRect(0, 8 + line * 23, 320, 23, TFT_BLACK);
  M5.Display.setCursor(6, 8 + line * 23);
  M5.Display.print(buf);
}

// ===================================================================== USB 受信
static void count_frames(const uint8_t* data, size_t len) {
  g_rxBytes += len;
  for (size_t i = 0; i < len; i++) {
    if (data[i] == MAGIC[g_magicPos]) {
      if (++g_magicPos == 8) { g_rxFrames++; g_magicPos = 0; }
    } else {
      g_magicPos = (data[i] == MAGIC[0]) ? 1 : 0;
    }
  }
}

static void parse_cfg(const uint8_t* data, size_t len) {
  for (size_t i = 0; i < len; i++) {
    char c = (char)data[i];
    if (c == '\n' || c == '\r') {
      if (g_cfgLen > 0) {
        g_cfgLine[g_cfgLen] = 0;
        strncpy(g_lastReply, g_cfgLine, sizeof(g_lastReply) - 1);
        if (strstr(g_cfgLine, "Done")) xSemaphoreGive(g_cfgAck);
        if (strstr(g_cfgLine, "Error")) { g_cfgError = true; xSemaphoreGive(g_cfgAck); }
        g_cfgLen = 0;
      }
    } else if (g_cfgLen < (int)sizeof(g_cfgLine) - 1) {
      g_cfgLine[g_cfgLen++] = c;
    }
  }
}

// 両方の口で共通。user_arg に口の番号（0/1）
static bool rx_cb(const uint8_t* data, size_t len, void* arg) {
  int64_t t = esp_timer_get_time();      // 到着時刻はできるだけ早く取る
  int iface = (int)(intptr_t)arg;
  g_rxPort[iface] += len;
  int cfg = g_cfgIface;
  if (cfg < 0 || iface == cfg) { parse_cfg(data, len); return true; }

  count_frames(data, len);
  if (g_recording) {
    void* p = nullptr;
    if (xRingbufferSendAcquire(g_rbRadar, &p, sizeof(RadarHdr) + len, 0) == pdTRUE) {
      RadarHdr h{t, (uint32_t)len};
      memcpy(p, &h, sizeof(h));
      memcpy((uint8_t*)p + sizeof(h), data, len);
      xRingbufferSendComplete(g_rbRadar, p);
    } else {
      g_dropRadar++;
    }
  }
  return true;
}

static void event_cb(const cdc_acm_host_dev_event_data_t* ev, void*) {
  if (ev->type == CDC_ACM_HOST_DEVICE_DISCONNECTED) {
    ESP_LOGW(TAG, "device disconnected");
    // 記録中なら書き込みタスクにファイルを閉じさせてから再起動する
    if (g_recording) { g_req = 2; vTaskDelay(pdMS_TO_TICKS(1500)); }
    esp_restart();
  }
}

static void usb_lib_task(void*) {
  while (true) {
    uint32_t flags;
    usb_host_lib_handle_events(portMAX_DELAY, &flags);
  }
}

static void set_baud(cdc_acm_dev_hdl_t h, uint32_t baud) {
  cdc_acm_line_coding_t lc = {};
  lc.dwDTERate = baud; lc.bCharFormat = 0; lc.bParityType = 0; lc.bDataBits = 8;
  cdc_acm_host_line_coding_set(h, &lc);
}

static cdc_acm_dev_hdl_t open_port(uint8_t iface, uint32_t baud) {
  cdc_acm_host_device_config_t cfg = {};
  cfg.connection_timeout_ms = 10000;
  cfg.out_buffer_size = 256;
  cfg.in_buffer_size = 512;
  cfg.event_cb = event_cb;
  cfg.data_cb = rx_cb;
  cfg.user_arg = (void*)(intptr_t)iface;
  cdc_acm_dev_hdl_t h = nullptr;
  esp_err_t err = cp210x_vcp_open(CP2105_PID, iface, &cfg, &h);
  if (err != ESP_OK) { status(6, "open if%d err %s", iface, esp_err_to_name(err)); return nullptr; }
  set_baud(h, baud);
  cdc_acm_host_set_control_line_state(h, true, true);  // PC の pyserial と同じ
  return h;
}

static bool send_cfg(cdc_acm_dev_hdl_t h) {
  const char* p = CFG_TEXT;
  int n = 0;
  while (*p) {
    const char* e = strchr(p, '\n');
    size_t len = e ? (size_t)(e - p) : strlen(p);
    if (len > 0 && p[0] != '%') {
      char line[160];
      size_t l = len < sizeof(line) - 2 ? len : sizeof(line) - 2;
      memcpy(line, p, l); line[l++] = '\n'; line[l] = 0;
      xSemaphoreTake(g_cfgAck, 0);
      cdc_acm_host_data_tx_blocking(h, (const uint8_t*)line, l, 1000);
      bool ok = xSemaphoreTake(g_cfgAck, pdMS_TO_TICKS(1500)) == pdTRUE;
      n++;
      status(4, "cfg %d %s", n, ok ? "ok" : "timeout");
      if (g_cfgError) {
        snprintf(g_cfgStatus, sizeof(g_cfgStatus), "cfg ERROR line %d", n);
        status(5, "%.26s", g_lastReply);
        return false;
      }
    }
    if (!e) break;
    p = e + 1;
  }
  snprintf(g_cfgStatus, sizeof(g_cfgStatus), "cfg ok (%d lines)", n);
  return true;
}

// ===================================================================== microSD
static bool mount_sd(uint64_t* total_mb) {
  // バスは M5.begin() で M5GFX が初期化済み。念のため呼び、既に初期化済みなら無視する
  spi_bus_config_t bus = {};
  bus.mosi_io_num = SD_MOSI; bus.miso_io_num = SD_MISO; bus.sclk_io_num = SD_SCLK;
  bus.quadwp_io_num = -1; bus.quadhd_io_num = -1; bus.max_transfer_sz = 4096;
  esp_err_t e = spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO);
  if (e != ESP_OK && e != ESP_ERR_INVALID_STATE) {
    ESP_LOGE(TAG, "spi_bus_initialize: %s", esp_err_to_name(e));
    return false;
  }

  sdmmc_host_t host = SDSPI_HOST_DEFAULT();
  host.slot = SPI2_HOST;
  host.max_freq_khz = 20000;
  sdspi_device_config_t slot = SDSPI_DEVICE_CONFIG_DEFAULT();
  slot.gpio_cs = (gpio_num_t)SD_CS;
  slot.host_id = SPI2_HOST;

  esp_vfs_fat_sdmmc_mount_config_t mc = {};
  mc.format_if_mount_failed = false;
  mc.max_files = 6;
  mc.allocation_unit_size = 16 * 1024;

  sdmmc_card_t* card = nullptr;
  e = esp_vfs_fat_sdspi_mount(MOUNT, &host, &slot, &mc, &card);
  if (e != ESP_OK) { ESP_LOGE(TAG, "sd mount: %s", esp_err_to_name(e)); return false; }
  *total_mb = (uint64_t)card->csd.capacity * card->csd.sector_size / (1024 * 1024);
  return true;
}

// ===================================================================== 書き込みタスク
// フレーム境界を探して frames.csv に時刻を付ける（pc/logger.py の RadarReader と同じ手順）
struct FrameParser {
  std::vector<uint8_t> buf;
  uint64_t buf_start = 0;   // buf[0] の絶対位置（記録開始からのバイト数）
  uint64_t stream_pos = 0;  // これまでに受けた総バイト数
  double t_prev = -1;
  bool synced = false;      // 最初のマジックワードを見つけたか
  uint32_t frames = 0;
  uint64_t dat_bytes = 0;

  void reset() { buf.clear(); buf_start = stream_pos = 0; t_prev = -1; synced = false; frames = 0; dat_bytes = 0; }

  long find_magic() const {
    auto it = std::search(buf.begin(), buf.end(), MAGIC, MAGIC + 8);
    return it == buf.end() ? -1 : (long)(it - buf.begin());
  }

  // t: チャンク到着時刻 [s]（記録開始から）
  void feed(const uint8_t* d, size_t n, double t, FILE* fdat, FILE* ffr) {
    uint64_t chunk_start = stream_pos;
    stream_pos += n;
    buf.insert(buf.end(), d, d + n);

    if (!synced) {
      // .dat はフレーム先頭から書き始める
      long i = find_magic();
      if (i < 0) {
        if (buf.size() > 7) { buf_start += buf.size() - 7; buf.erase(buf.begin(), buf.end() - 7); }
        t_prev = t;
        return;
      }
      synced = true;
      fwrite(buf.data() + i, 1, buf.size() - i, fdat);
      dat_bytes += buf.size() - i;
    } else {
      fwrite(d, 1, n, fdat);
      dat_bytes += n;
    }

    while (true) {
      long i = find_magic();
      if (i < 0 || buf.size() - i < 40) break;
      uint32_t tl, frame;
      memcpy(&tl, &buf[i + 8 + 4], 4);
      memcpy(&frame, &buf[i + 8 + 12], 4);
      uint64_t p_abs = buf_start + i;
      double t_frame;
      if (p_abs >= chunk_start) t_frame = t - (double)(stream_pos - p_abs) / BYTES_PER_SEC;
      else                      t_frame = (t_prev >= 0) ? t_prev : t;
      fprintf(ffr, "%.6f,%lu,%lu\n", t_frame, (unsigned long)frame, (unsigned long)tl);
      frames++;
      buf_start += i + 8;
      buf.erase(buf.begin(), buf.begin() + i + 8);
    }
    if (buf.size() > 65536) {
      buf_start += buf.size() - 4096;
      buf.erase(buf.begin(), buf.end() - 4096);
    }
    t_prev = t;
  }
};

static void write_meta(const char* prefix, const char* iso, double dur, const FrameParser& fp,
                       uint32_t nimu) {
  char path[96];
  snprintf(path, sizeof(path), "%s_meta.json", prefix);
  FILE* f = fopen(path, "w");
  if (!f) return;
  fprintf(f, "{\n  \"device\": \"cores3_s3_logger\",\n");
  fprintf(f, "  \"start_iso\": \"%s\",\n", iso);
  fprintf(f, "  \"clock\": \"radar/imu t_host: esp_timer since start; imu t_dev_us: BMI270 FIFO index x 5000 us\",\n");
  fprintf(f, "  \"imu_fifo_skipped\": %lu,\n", (unsigned long)g_imuSkipped.load());
  fprintf(f, "  \"cfg\": \"%s (embedded)\",\n", CFG_NAME);
  fprintf(f, "  \"duration_s\": %.3f,\n", dur);
  fprintf(f, "  \"radar_frames\": %lu,\n", (unsigned long)fp.frames);
  fprintf(f, "  \"radar_bytes\": %llu,\n", (unsigned long long)fp.dat_bytes);
  fprintf(f, "  \"imu_samples\": %lu,\n", (unsigned long)nimu);
  fprintf(f, "  \"drop_radar_chunks\": %lu,\n", (unsigned long)g_dropRadar.load());
  fprintf(f, "  \"drop_imu_samples\": %lu\n}\n", (unsigned long)g_dropImu.load());
  fclose(f);
}

static void writer_task(void*) {
  FrameParser fp;
  FILE *fdat = nullptr, *ffr = nullptr, *fimu = nullptr;
  char prefix[64] = "", iso[32] = "";
  int64_t t0 = 0;
  uint32_t nimu = 0;
  int64_t lastSync = 0, lastDisp = 0;
  uint32_t lastB = 0, lastF = 0, lastI = 0;

  auto close_all = [&]() {
    double dur = (esp_timer_get_time() - t0) / 1e6;
    if (fdat) { fclose(fdat); fdat = nullptr; }
    if (ffr)  { fclose(ffr);  ffr = nullptr; }
    if (fimu) { fclose(fimu); fimu = nullptr; }
    write_meta(prefix, iso, dur, fp, nimu);
  };

  while (true) {
    // ---- 開始／停止の要求
    int req = g_req.exchange(0);
    if (req == 1 && !g_recording && g_sdOk) {
      strncpy(prefix, g_reqPrefix, sizeof(prefix) - 1);
      strncpy(iso, g_reqIso, sizeof(iso) - 1);
      char path[96];
      snprintf(path, sizeof(path), "%s.dat", prefix);           fdat = fopen(path, "wb");
      snprintf(path, sizeof(path), "%s_frames.csv", prefix);    ffr  = fopen(path, "w");
      snprintf(path, sizeof(path), "%s_imu.csv", prefix);       fimu = fopen(path, "w");
      if (fdat && ffr && fimu) {
        static char b1[16384], b2[4096], b3[16384];
        setvbuf(fdat, b1, _IOFBF, sizeof(b1));
        setvbuf(ffr,  b2, _IOFBF, sizeof(b2));
        setvbuf(fimu, b3, _IOFBF, sizeof(b3));
        fprintf(ffr, "t_host,frame,n_bytes\n");
        fprintf(fimu, "t_host,t_dev_us,gx,gy,gz,ax,ay,az\n");
        fp.reset(); nimu = 0;
        g_dropRadar = 0; g_dropImu = 0; g_imuSkipped = 0;   // 起動直後（cfg 送信中）の溢れは数えない
        t0 = esp_timer_get_time();
        g_recording = true;
      } else {
        if (fdat) fclose(fdat);
        if (ffr) fclose(ffr);
        if (fimu) fclose(fimu);
        fdat = ffr = fimu = nullptr;
        status(7, "file open FAILED");
      }
    } else if (req == 2 && g_recording) {
      g_recording = false;
      vTaskDelay(pdMS_TO_TICKS(20));
      // リングに残っている分を書き切ってから閉じる
      size_t sz; void* it;
      while ((it = xRingbufferReceive(g_rbRadar, &sz, 0))) {
        auto* h = (RadarHdr*)it;
        if (h->t_us >= t0) fp.feed((uint8_t*)it + sizeof(RadarHdr), h->len, (h->t_us - t0) / 1e6, fdat, ffr);
        vRingbufferReturnItem(g_rbRadar, it);
      }
      while ((it = xRingbufferReceive(g_rbImu, &sz, 0))) {
        auto* s = (ImuSample*)it;
        if (s->t_us >= t0) {
          fprintf(fimu, "%.6f,%lld,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", (s->t_us - t0) / 1e6,
                  (long long)s->t_dev_us, s->g[0], s->g[1], s->g[2], s->a[0], s->a[1], s->a[2]);
          nimu++;
        }
        vRingbufferReturnItem(g_rbImu, it);
      }
      close_all();
    }

    // ---- レーダー
    size_t sz; void* it;
    int n = 0;
    while (n++ < 32 && (it = xRingbufferReceive(g_rbRadar, &sz, pdMS_TO_TICKS(n == 1 ? 5 : 0)))) {
      auto* h = (RadarHdr*)it;
      if (g_recording && fdat && h->t_us >= t0)
        fp.feed((uint8_t*)it + sizeof(RadarHdr), h->len, (h->t_us - t0) / 1e6, fdat, ffr);
      vRingbufferReturnItem(g_rbRadar, it);
    }
    // ---- IMU
    while ((it = xRingbufferReceive(g_rbImu, &sz, 0))) {
      auto* s = (ImuSample*)it;
      if (g_recording && fimu && s->t_us >= t0) {
        fprintf(fimu, "%.6f,%lld,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", (s->t_us - t0) / 1e6,
                (long long)s->t_dev_us, s->g[0], s->g[1], s->g[2], s->a[0], s->a[1], s->a[2]);
        nimu++;
      }
      vRingbufferReturnItem(g_rbImu, it);
    }

    int64_t now = esp_timer_get_time();
    // ---- 2 秒ごとにカードへ確定させる（電源が落ちても直前までは残る）
    if (g_recording && now - lastSync > 2000000) {
      fflush(fdat); fflush(ffr); fflush(fimu);
      fsync(fileno(fdat)); fsync(fileno(ffr)); fsync(fileno(fimu));
      lastSync = now;
    }
    // ---- 画面（SPI バスを SD と共有するので、このタスクだけが描く）
    if (now - lastDisp > 1000000) {
      double dt = (now - lastDisp) / 1e6;
      uint32_t b = g_rxBytes, f = g_rxFrames, ni = g_imuN;
      if (lastDisp > 0) {
        status(4, "radar %4.1f fps %5.1f kB/s", (f - lastF) / dt, (b - lastB) / dt / 1024.0);
        status(5, "imu   %5.1f Hz", (ni - lastI) / dt);
        if (g_recording) {
          M5.Display.setTextColor(TFT_RED, TFT_BLACK);
          status(6, "REC %6.1f s  %lu fr", (now - t0) / 1e6, (unsigned long)fp.frames);
          M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
          status(7, "%s", prefix + strlen(MOUNT) + 1);
        } else {
          status(6, "STOPPED");
          status(7, "%s", prefix[0] ? prefix + strlen(MOUNT) + 1 : "");
        }
        status(8, "drop radar=%lu imu=%lu", (unsigned long)g_dropRadar.load(),
               (unsigned long)g_dropImu.load());
      }
      lastB = b; lastF = f; lastI = ni; lastDisp = now;
    }
  }
}

// ===================================================================== IMU（BMI270 FIFO）
// 参照: Bosch BMI270_SensorAPI (bmi2_defs.h / bmi2.c)
static constexpr uint32_t I2C_FREQ = 400000;
static constexpr int64_t  IMU_PERIOD_US = 5000;        // 200 Hz
static constexpr float    ARES = 8.0f / 32768.0f;      // ±8 g（M5Unified と同じ換算）
static constexpr float    GRES = 2000.0f / 32768.0f;   // ±2000 deg/s
static uint8_t g_bmiAddr = 0x69;
static bool imuOk = false;
static int64_t g_imuIdx = 0;                           // BMI270 のサンプル番号
static float g_lastAcc[3] = {0, 0, 0};

static bool bmi_w(uint8_t reg, uint8_t v) { return M5.In_I2C.writeRegister8(g_bmiAddr, reg, v, I2C_FREQ); }
static bool bmi_r(uint8_t reg, uint8_t* buf, size_t n) { return M5.In_I2C.readRegister(g_bmiAddr, reg, buf, n, I2C_FREQ); }

static bool setup_imu_fifo() {
  // M5.begin() が BMI270 を初期化済み。アドレスだけ確認する（CoreS3 は 0x69）
  uint8_t id = 0;
  for (uint8_t a : {0x69, 0x68}) {
    g_bmiAddr = a;
    if (bmi_r(0x00, &id, 1) && id == 0x24) break;
  }
  if (id != 0x24) return false;
  bool ok = true;
  ok &= bmi_w(0x40, 0xA9);   // ACC_CONF : 200 Hz（ジャイロと揃える）, normal, perf
  ok &= bmi_w(0x42, 0xA9);   // GYR_CONF : 200 Hz, normal, perf
  ok &= bmi_w(0x48, 0x00);   // FIFO_CONFIG_0 : 満杯時は古い方を捨てる, sensortime なし
  ok &= bmi_w(0x49, 0xD0);   // FIFO_CONFIG_1 : gyr_en | acc_en | header_en
  vTaskDelay(pdMS_TO_TICKS(5));
  ok &= bmi_w(0x7E, 0xB0);   // FIFO flush
  return ok;
}

static void push_imu(int64_t t_us, const int16_t* g, const float* a) {
  ImuSample s;
  s.t_us = t_us;
  s.t_dev_us = g_imuIdx * IMU_PERIOD_US;
  for (int i = 0; i < 3; i++) { s.g[i] = g[i] * GRES; s.a[i] = a[i]; }
  g_imuN++;
  if (g_recording && xRingbufferSend(g_rbImu, &s, sizeof(s), 0) != pdTRUE) g_dropImu++;
}

static void read_imu_fifo() {
  uint8_t lb[2];
  if (!bmi_r(0x24, lb, 2)) return;
  // 最新のフレームは読み出しの 0〜5 ms 前にできている。平均をとって半周期前とする
  int64_t t_read = esp_timer_get_time() - IMU_PERIOD_US / 2;
  size_t len = lb[0] | ((lb[1] & 0x3F) << 8);
  if (len == 0) return;
  static uint8_t buf[1024];
  if (len > sizeof(buf)) len = sizeof(buf);
  if (!bmi_r(0x26, buf, len)) return;

  // まず gyro を含むフレーム数を数える（最後のフレーム = t_read とし、前へ 5 ms ずつ戻す）
  int nGyr = 0;
  for (size_t i = 0; i < len;) {
    uint8_t h = buf[i++];
    if (h == 0x8C)      { nGyr++; i += 12; }
    else if (h == 0x88) { nGyr++; i += 6; }
    else if (h == 0x84) { i += 6; }
    else if (h == 0x40) { i += 1; }
    else if (h == 0x44) { i += 3; }
    else if (h == 0x48) { i += 4; }
    else break;          // 0x80 = データの終わり、その他 = 想定外
  }

  int k = 0;
  for (size_t i = 0; i < len;) {
    uint8_t h = buf[i++];
    if (h == 0x8C || h == 0x88) {             // ジャイロ(6) [+ 加速度(6)]
      size_t need = (h == 0x8C) ? 12 : 6;
      if (i + need > len) break;
      int16_t g[3];
      memcpy(g, &buf[i], 6);
      if (h == 0x8C) {
        int16_t a[3];
        memcpy(a, &buf[i + 6], 6);
        for (int j = 0; j < 3; j++) g_lastAcc[j] = a[j] * ARES;
      }
      push_imu(t_read - (int64_t)(nGyr - 1 - k) * IMU_PERIOD_US, g, g_lastAcc);
      g_imuIdx++;
      k++;
      i += need;
    } else if (h == 0x84) {                   // 加速度だけ
      if (i + 6 > len) break;
      int16_t a[3];
      memcpy(a, &buf[i], 6);
      for (int j = 0; j < 3; j++) g_lastAcc[j] = a[j] * ARES;
      i += 6;
    } else if (h == 0x40) {                   // スキップ（FIFO 溢れ）。中身 = 捨てたフレーム数
      if (i < len) { g_imuIdx += buf[i]; g_imuSkipped += buf[i]; }
      i += 1;
    } else if (h == 0x44) { i += 3; }         // センサ時刻
      else if (h == 0x48) { i += 4; }         // 設定変更
      else {
      if (h != 0x80) bmi_w(0x7E, 0xB0);       // 想定外のヘッダ → 同期し直す
      break;
    }
  }
}

// ===================================================================== RTC
#include "nvs_flash.h"
#include "nvs.h"
static void nvs_init() {
  esp_err_t e = nvs_flash_init();
  if (e == ESP_ERR_NVS_NO_FREE_PAGES || e == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    nvs_flash_erase();
    nvs_flash_init();
  }
}

// このファームをビルドした日時（__DATE__ "Nov 24 2026", __TIME__ "12:34:56"、ビルドした PC の現地時刻）
static void set_rtc_from_build_time() {
  if (!M5.Rtc.isEnabled()) return;
  static const char* stamp = __DATE__ " " __TIME__;
  // 同じビルドで合わせるのは 1 回だけ（再起動のたびに戻さない）
  nvs_handle_t h;
  char saved[32] = "";
  size_t sz = sizeof(saved);
  bool sameBuild = false;
  if (nvs_open("s3log", NVS_READWRITE, &h) == ESP_OK) {
    if (nvs_get_str(h, "rtc_build", saved, &sz) == ESP_OK && strcmp(saved, stamp) == 0) sameBuild = true;
  } else {
    h = 0;
  }
  auto now = M5.Rtc.getDateTime();
  if (sameBuild && now.date.year >= 2025) { if (h) nvs_close(h); return; }

  static const char* mon = "JanFebMarAprMayJunJulAugSepOctNovDec";
  char m[4] = {__DATE__[0], __DATE__[1], __DATE__[2], 0};
  int month = (int)((strstr(mon, m) - mon) / 3) + 1;
  int day = atoi(__DATE__ + 4), year = atoi(__DATE__ + 7);
  int hh = atoi(__TIME__), mm = atoi(__TIME__ + 3), ss = atoi(__TIME__ + 6);
  // 曜日（Sakamoto の方法、0 = 日曜）
  static const int tw[12] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
  int y = year - (month < 3);
  int wd = (y + y / 4 - y / 100 + y / 400 + tw[month - 1] + day) % 7;

  m5::rtc_datetime_t dt;
  dt.date.year = year; dt.date.month = month; dt.date.date = day; dt.date.weekDay = wd;
  dt.time.hours = hh; dt.time.minutes = mm; dt.time.seconds = ss;
  M5.Rtc.setDateTime(dt);
  if (h) { nvs_set_str(h, "rtc_build", stamp); nvs_commit(h); nvs_close(h); }
}

// ===================================================================== 記録ファイル名
static void make_prefix() {
  auto dt = M5.Rtc.getDateTime();
  if (M5.Rtc.isEnabled() && dt.date.year >= 2025) {
    snprintf(g_reqPrefix, sizeof(g_reqPrefix), "%s/s3_%04d%02d%02d_%02d%02d%02d", MOUNT,
             dt.date.year, dt.date.month, dt.date.date, dt.time.hours, dt.time.minutes, dt.time.seconds);
    snprintf(g_reqIso, sizeof(g_reqIso), "%04d-%02d-%02dT%02d:%02d:%02d", dt.date.year,
             dt.date.month, dt.date.date, dt.time.hours, dt.time.minutes, dt.time.seconds);
    return;
  }
  // RTC が未設定なら連番
  for (int i = 1; i < 10000; i++) {
    char p[96];
    snprintf(g_reqPrefix, sizeof(g_reqPrefix), "%s/s3_%04d", MOUNT, i);
    snprintf(p, sizeof(p), "%s.dat", g_reqPrefix);
    struct stat st;
    if (stat(p, &st) != 0) break;
  }
  snprintf(g_reqIso, sizeof(g_reqIso), "unknown (RTC not set)");
}

// ===================================================================== main
// app_main のスタック（既定 3.5 kB）では足りない可能性があるので、大きめのタスクで動かす
static void app_task(void*);
extern "C" void app_main(void) {
  xTaskCreatePinnedToCore(app_task, "app", 16384, nullptr, 5, nullptr, 0);
}

static void app_task(void*) {
  auto mcfg = M5.config();
  mcfg.internal_imu = true;
  ESP_LOGI(TAG, "M5.begin");
  M5.begin(mcfg);
  M5.Display.setTextSize(2);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.fillScreen(TFT_BLACK);
  status(0, "S3 LOGGER radar+IMU");
  ESP_LOGI(TAG, "display ok");

  // ---- microSD
  uint64_t mb = 0;
  g_sdOk = mount_sd(&mb);
  status(1, g_sdOk ? "SD ok (%llu MB)" : "SD NOT FOUND", (unsigned long long)mb);

  // ---- RTC（初回だけビルド時刻に合わせる）と IMU の FIFO 設定
  // どこで止まるかが分かるよう、各段階を画面の最下行に出す
  status(8, "init: nvs");
  ESP_LOGI(TAG, "init: nvs");
  nvs_init();
  status(8, "init: rtc");
  ESP_LOGI(TAG, "init: rtc");
  set_rtc_from_build_time();
  status(8, "init: imu fifo");
  ESP_LOGI(TAG, "init: imu fifo");
  imuOk = M5.Imu.isEnabled() && setup_imu_fifo();
  status(8, imuOk ? "init: ok" : "init: IMU FIFO FAILED");

  // ---- EVM へ給電して USB ホストを起動
  bool pw = M5.Power.setUsbOutput(true);
  status(2, "USB 5V out: %s", pw ? "on" : "FAILED");

  g_cfgAck = xSemaphoreCreateBinary();
  g_rbRadar = xRingbufferCreate(96 * 1024, RINGBUF_TYPE_NOSPLIT);
  g_rbImu   = xRingbufferCreate(32 * 1024, RINGBUF_TYPE_NOSPLIT);

  usb_host_config_t host_cfg = {};
  host_cfg.skip_phy_setup = false;
  host_cfg.intr_flags = ESP_INTR_FLAG_LEVEL1;
  ESP_ERROR_CHECK(usb_host_install(&host_cfg));
  xTaskCreate(usb_lib_task, "usb_lib", 4096, nullptr, 10, nullptr);
  ESP_ERROR_CHECK(cdc_acm_host_install(nullptr));

  status(3, "waiting for EVM...");
  vTaskDelay(pdMS_TO_TICKS(BOOT_WAIT_MS));

  // ---- CP2105 の 2 つの口のどちらが CFG かを判定（"version" に返事した方）
  cdc_acm_dev_hdl_t h[2] = {open_port(0, BAUD_CFG), open_port(1, BAUD_CFG)};
  if (!h[0] || !h[1]) { status(3, "EVM open failed"); while (true) vTaskDelay(1000); }
  for (int attempt = 0; attempt < 5 && g_cfgIface < 0; attempt++) {
    for (int i = 0; i < 2 && g_cfgIface < 0; i++) {
      uint32_t before = g_rxPort[i];
      cdc_acm_host_data_tx_blocking(h[i], (const uint8_t*)"\nversion\n", 9, 1000);
      vTaskDelay(pdMS_TO_TICKS(500));
      if (g_rxPort[i] != before) g_cfgIface = i;
    }
  }
  if (g_cfgIface < 0) {
    status(3, "no reply from radar");
    status(4, "check EVM switches (all OFF)");
    while (true) vTaskDelay(1000);
  }
  int ci = g_cfgIface, di = 1 - ci;
  status(3, "CFG=if%d DATA=if%d", ci, di);
  vTaskDelay(pdMS_TO_TICKS(300));
  set_baud(h[di], BAUD_DATA);
  bool cfgOk = send_cfg(h[ci]);
  status(3, "%s", g_cfgStatus);
  if (!cfgOk) { while (true) vTaskDelay(1000); }

  // ---- ここから画面は書き込みタスクだけが触る
  M5.Display.fillRect(0, 8 + 4 * 23, 320, 240 - (8 + 4 * 23), TFT_BLACK);
  M5.Display.drawRect(0, 216, 320, 24, g_sdOk ? TFT_GREEN : TFT_DARKGREY);
  M5.Display.setCursor(60, 220);
  M5.Display.print(g_sdOk ? "TOUCH: START/STOP" : "NO SD CARD");
  xTaskCreatePinnedToCore(writer_task, "writer", 8192, nullptr, 4, nullptr, 1);

  // ---- IMU とタッチ（I2C を使うものはこのタスクだけで扱う）
  int64_t lastFifo = 0;
  while (true) {
    M5.update();
    // 10 ms ごとに FIFO をまとめて読む（200 Hz なら 1 回あたり約 2 サンプル）
    if (imuOk && esp_timer_get_time() - lastFifo >= 10000) {
      lastFifo = esp_timer_get_time();
      read_imu_fifo();
    }
    auto t = M5.Touch.getDetail();
    if (t.wasPressed() && t.y > 180 && g_sdOk) {
      if (!g_recording) { make_prefix(); g_req = 1; }
      else              { g_req = 2; }
    }
    vTaskDelay(1);
  }
}
