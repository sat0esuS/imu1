/*
 * cores3_imu.ino  --  M5Stack CoreS3 内蔵IMU(BMI270)ロガー用ファーム
 *
 *  役割:
 *    内蔵IMUを可能な限り速く読み、1サンプル1行でUSBシリアルへ流すだけ。
 *    積分や姿勢推定は一切しない(PC側のPythonでやる)。
 *    画面には動作確認用の値を出すが、記録とは無関係。
 *
 *  出力フォーマット (CSV, 1行1サンプル):
 *    I,<micros>,<gx>,<gy>,<gz>,<ax>,<ay>,<az>
 *       micros : CoreS3起動からの経過時間 [us]  ← 間隔の精度はこちらが正確
 *       gx..gz : ジャイロ [deg/s] (M5Unifiedの単位をそのまま)
 *       ax..az : 加速度 [G]
 *
 *    起動時と1秒ごとに識別行を出す:
 *    H,<micros>,<n_samples>,<rate_hz>
 *
 *  PC側は受信時刻(ホスト時計)を各行に付けて保存する。
 *  micros とホスト時刻の両方を残すのは、
 *    ・micros  → サンプル間隔が正確(USBのバッファ遅れの影響を受けない)
 *    ・ホスト時刻 → レーダーと共通の時計
 *  の良いとこ取りをして、後から1次式で対応づけるため。
 *
 *  --------------------------------------------------------------------
 *  書き込み方法は3通り:
 *
 *   A. ブラウザから書き込む（推奨。ツール不要）
 *        https://sat0esus.github.io/imu1/  を Chrome / Edge で開いて
 *        「ファームウェアを書き込む」を押すだけ。
 *
 *   B. PlatformIO
 *        cd firmware && pio run -t upload
 *
 *   C. Arduino IDE
 *        src/main.cpp の中身を .ino にコピーし、
 *        ボード: M5CoreS3 / USB CDC On Boot: Enabled / ライブラリ: M5Unified
 *
 *  --------------------------------------------------------------------
 *  操作(画面下部をタッチ):
 *    左   : バイアス測定をやり直す(静止させてから押す)
 *    中央 : 画面表示のヨー角をゼロに戻す(記録には影響しない)
 *    右   : 出力のON/OFF
 */

#include <Arduino.h>
#include <M5Unified.h>

// ---------------------------------------------------------------- 設定
static const uint32_t SERIAL_BAUD = 921600;   // USB CDCでは実質無視されるが一応
static const uint32_t DISP_INTERVAL_MS = 200; // 画面更新(読みやすさ優先で遅く)
static const uint32_t HEARTBEAT_MS = 1000;    // H行の間隔
static const uint16_t BIAS_SAMPLES = 400;     // バイアス測定に使うサンプル数

// ---------------------------------------------------------------- 状態
static bool streaming = true;
static uint32_t n_total = 0;        // 送信したサンプル数
static uint32_t n_window = 0;       // レート計算用
static uint32_t t_last_disp = 0;
static uint32_t t_last_hb = 0;
static float rate_hz = 0.0f;

// 画面表示用(記録には使わない参考値)
static float bias_z = 0.0f;
static bool  bias_done = false;
static uint16_t bias_count = 0;
static float bias_accum = 0.0f;
static float yaw_disp = 0.0f;
static uint32_t t_prev_us = 0;

static float last_gx = 0, last_gy = 0, last_gz = 0;
static float last_ax = 0, last_ay = 0, last_az = 0;


// ---------------------------------------------------------------- 画面
static void drawStatic() {
  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.setTextSize(2);
  M5.Display.setCursor(8, 6);
  M5.Display.print("IMU LOGGER (BMI270)");

  M5.Display.setTextSize(1);
  M5.Display.setTextColor(TFT_DARKGREY, TFT_BLACK);
  M5.Display.setCursor(8, 212);
  M5.Display.print(" [BIAS]        [ZERO]         [OUT] ");
}

static void drawValues() {
  M5.Display.setTextSize(2);

  M5.Display.setTextColor(TFT_CYAN, TFT_BLACK);
  M5.Display.setCursor(8, 40);
  M5.Display.printf("gz %+8.3f dps ", last_gz);

  M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Display.setCursor(8, 66);
  M5.Display.printf("gx %+8.3f      ", last_gx);
  M5.Display.setCursor(8, 88);
  M5.Display.printf("gy %+8.3f      ", last_gy);

  M5.Display.setTextColor(TFT_YELLOW, TFT_BLACK);
  M5.Display.setCursor(8, 118);
  M5.Display.printf("yaw %+8.2f deg ", yaw_disp);

  M5.Display.setTextColor(bias_done ? TFT_GREEN : TFT_ORANGE, TFT_BLACK);
  M5.Display.setCursor(8, 144);
  if (bias_done) M5.Display.printf("bias %+7.4f dps", bias_z);
  else           M5.Display.printf("bias measuring %3d%%", (bias_count * 100) / BIAS_SAMPLES);

  M5.Display.setTextColor(streaming ? TFT_GREEN : TFT_RED, TFT_BLACK);
  M5.Display.setCursor(8, 170);
  M5.Display.printf("%s  %5.1f Hz  n=%lu   ",
                    streaming ? "OUT ON " : "OUT OFF", rate_hz, (unsigned long)n_total);
}


// ---------------------------------------------------------------- setup
void setup() {
  auto cfg = M5.config();
  cfg.internal_imu = true;
  M5.begin(cfg);

  Serial.begin(SERIAL_BAUD);
  delay(300);

  M5.Display.setRotation(1);
  drawStatic();

  if (!M5.Imu.isEnabled()) {
    M5.Display.setTextColor(TFT_RED, TFT_BLACK);
    M5.Display.setTextSize(2);
    M5.Display.setCursor(8, 100);
    M5.Display.print("IMU NOT FOUND");
    Serial.println("E,imu_not_found");
    while (true) delay(1000);
  }

  // 起動メッセージ(PC側がデバイスを識別するのに使う)
  Serial.println("#,cores3_imu,v1,gyro_dps,accel_g");

  t_prev_us = micros();
  t_last_disp = millis();
  t_last_hb = millis();
}


// ---------------------------------------------------------------- loop
void loop() {
  M5.update();

  // --- ボタン(画面下部のタッチ)
  if (M5.BtnA.wasPressed()) {          // バイアス測定やり直し
    bias_done = false;
    bias_count = 0;
    bias_accum = 0.0f;
    yaw_disp = 0.0f;
    Serial.printf("M,%lu,bias_restart\n", (unsigned long)micros());
  }
  if (M5.BtnB.wasPressed()) {          // 表示ヨーをゼロに
    yaw_disp = 0.0f;
    Serial.printf("M,%lu,yaw_zero\n", (unsigned long)micros());
  }
  if (M5.BtnC.wasPressed()) {          // 出力ON/OFF
    streaming = !streaming;
    Serial.printf("M,%lu,stream_%s\n", (unsigned long)micros(),
                  streaming ? "on" : "off");
  }

  // --- IMU読み出し(新しいデータが来たときだけ)
  auto mask = M5.Imu.update();
  if (mask) {
    uint32_t t_us = micros();
    float gx, gy, gz, ax, ay, az;
    M5.Imu.getGyro(&gx, &gy, &gz);
    M5.Imu.getAccel(&ax, &ay, &az);

    if (streaming) {
      // %.4f で十分。BMI270の分解能より細かく出しても意味がない
      Serial.printf("I,%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    (unsigned long)t_us, gx, gy, gz, ax, ay, az);
      n_total++;
      n_window++;
    }

    last_gx = gx; last_gy = gy; last_gz = gz;
    last_ax = ax; last_ay = ay; last_az = az;

    // --- 画面表示用のバイアス測定と積分(参考値。解析はPC側で行う)
    if (!bias_done) {
      bias_accum += gz;
      if (++bias_count >= BIAS_SAMPLES) {
        bias_z = bias_accum / BIAS_SAMPLES;
        bias_done = true;
        Serial.printf("M,%lu,bias_z=%.5f\n", (unsigned long)t_us, bias_z);
      }
    } else {
      float dt = (t_us - t_prev_us) * 1e-6f;
      if (dt > 0 && dt < 0.5f) yaw_disp += (gz - bias_z) * dt;
    }
    t_prev_us = t_us;
  }

  // --- ハートビート(受信側で欠落を検出するため)
  uint32_t now = millis();
  if (now - t_last_hb >= HEARTBEAT_MS) {
    rate_hz = n_window * 1000.0f / (now - t_last_hb);
    Serial.printf("H,%lu,%lu,%.1f\n",
                  (unsigned long)micros(), (unsigned long)n_total, rate_hz);
    n_window = 0;
    t_last_hb = now;
  }

  if (now - t_last_disp >= DISP_INTERVAL_MS) {
    drawValues();
    t_last_disp = now;
  }
}
