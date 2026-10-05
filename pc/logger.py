# -*- coding: utf-8 -*-
"""
logger.py -- レーダー(IWR6843AOP)とIMU(M5 CoreS3)を1つのプロセスで記録する

TI mmWave Demo Visualizer の置き換え。Visualizerは
  ・CFGポート(115200)に .cfg の各行を送る
  ・DATAポート(921600)から来たバイト列をそのまま .dat に落とす
ということしかしていないので、同じことをして、
さらに「ホスト時計での受信時刻」を一緒に記録する。

出力(すべて同じ時計 = PCのtime.perf_counter()基準):
  <prefix>.dat        Visualizerと同一形式。parse_dat() がそのまま使える
  <prefix>_frames.csv t_host, frame, n_bytes     レーダー各フレームの受信時刻
  <prefix>_imu.csv    t_host, t_dev_us, gx,gy,gz, ax,ay,az
  <prefix>_meta.json  起動時刻、ポート、cfgの内容など

使い方:
  python logger.py --cfg xwr68xx_AOP_profile.cfg \
                   --radar-cfg COM5 --radar-data COM6 --imu COM7 \
                   --out run2 --duration 600

ポートが分からないときは:
  python logger.py --list
"""
import argparse
import json
import os
import struct
import sys
import threading
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial がありません:  pip install pyserial")


MAGIC = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])


# ---------------------------------------------------------------- ポート一覧
def list_serial_ports():
    print("接続されているシリアルポート:")
    for p in list_ports.comports():
        print(f"  {p.device:12s}  {p.description}")
        print(f"               hwid: {p.hwid}")
    print()
    print("目安:")
    print("  レーダー  : 'XDS110 Class Application/User UART'  -> --radar-cfg")
    print("              'XDS110 Class Auxiliary Data Port'    -> --radar-data")
    print("  CoreS3    : 'USB Serial' / 'M5Stack' / ESP32-S3   -> --imu")


# ---------------------------------------------------------------- cfg送信
def send_cfg(port, cfg_path, verbose=True):
    """.cfg の各行をCFGポートへ送る。Visualizerがやっているのと同じ処理。"""
    with open(cfg_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]

    sent = []
    with serial.Serial(port, 115200, timeout=1.0) as s:
        time.sleep(0.2)
        s.reset_input_buffer()
        for ln in lines:
            if not ln or ln.startswith("%"):
                continue
            s.write((ln + "\n").encode())
            time.sleep(0.03)
            resp = s.read_all().decode(errors="ignore").strip()
            sent.append(ln)
            if verbose:
                mark = "ok" if "Done" in resp or "Skipped" in resp else resp[:40]
                print(f"  > {ln:<60s} {mark}")
            if ln == "sensorStart":
                break
    return sent


def send_stop(port):
    """計測終了時にセンサを止める。"""
    try:
        with serial.Serial(port, 115200, timeout=1.0) as s:
            s.write(b"sensorStop\n")
            time.sleep(0.1)
    except Exception as e:
        print(f"  sensorStop 送信に失敗: {e}")


# ---------------------------------------------------------------- レーダー受信
class RadarReader(threading.Thread):
    """DATAポートを読み、.dat にそのまま落としつつフレーム境界に時刻を打つ。"""

    def __init__(self, port, dat_path, frames_path, t0, stop_evt):
        super().__init__(daemon=True)
        self.port = port
        self.dat_path = dat_path
        self.frames_path = frames_path
        self.t0 = t0
        self.stop_evt = stop_evt
        self.n_bytes = 0
        self.n_frames = 0
        self.last_frame = None
        self.error = None

    def run(self):
        buf = bytearray()
        try:
            ser = serial.Serial(self.port, 921600, timeout=0.05)
        except Exception as e:
            self.error = f"DATAポートを開けません: {e}"
            return

        with ser, open(self.dat_path, "wb") as fdat, \
                open(self.frames_path, "w", encoding="utf-8") as ffr:
            ffr.write("t_host,frame,n_bytes\n")
            while not self.stop_evt.is_set():
                chunk = ser.read(8192)
                if not chunk:
                    continue
                t = time.perf_counter() - self.t0
                fdat.write(chunk)          # ← Visualizerと同じ生バイト列
                self.n_bytes += len(chunk)

                # フレーム境界を探して時刻を記録する。
                # チャンク到着時刻をそのフレームの時刻として使う。
                # 921600bps では 8KB ≒ 90ms 相当だが、実際は細切れで届くので
                # 誤差は数ms程度に収まる。
                buf += chunk
                while True:
                    i = buf.find(MAGIC)
                    if i < 0 or len(buf) - i < 40:
                        break
                    hdr = buf[i + 8:i + 40]
                    _, tl, _, frame, _, _, _, _ = struct.unpack("<8I", hdr)
                    ffr.write(f"{t:.6f},{frame},{tl}\n")
                    self.n_frames += 1
                    self.last_frame = frame
                    buf = buf[i + 8:]      # このマジックを消費して次を探す
                # バッファが育ちすぎないように切り詰める
                if len(buf) > 65536:
                    buf = buf[-4096:]
                ffr.flush()


# ---------------------------------------------------------------- IMU受信
class ImuReader(threading.Thread):
    """CoreS3 からの行を読み、受信時刻を付けてCSVに落とす。"""

    def __init__(self, port, csv_path, t0, stop_evt, baud=921600):
        super().__init__(daemon=True)
        self.port = port
        self.csv_path = csv_path
        self.t0 = t0
        self.stop_evt = stop_evt
        self.baud = baud
        self.n = 0
        self.rate = 0.0
        self.markers = []
        self.error = None

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.1)
        except Exception as e:
            self.error = f"IMUポートを開けません: {e}"
            return

        with ser, open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("t_host,t_dev_us,gx,gy,gz,ax,ay,az\n")
            ser.reset_input_buffer()
            while not self.stop_evt.is_set():
                try:
                    raw = ser.readline()
                except Exception:
                    continue
                if not raw:
                    continue
                t = time.perf_counter() - self.t0
                line = raw.decode(errors="ignore").strip()
                if line.startswith("I,"):
                    p = line.split(",")
                    if len(p) == 8:
                        f.write(f"{t:.6f},{p[1]},{p[2]},{p[3]},{p[4]},"
                                f"{p[5]},{p[6]},{p[7]}\n")
                        self.n += 1
                elif line.startswith("H,"):
                    p = line.split(",")
                    if len(p) == 4:
                        try:
                            self.rate = float(p[3])
                        except ValueError:
                            pass
                    f.flush()
                elif line.startswith(("M,", "#,", "E,")):
                    self.markers.append((t, line))
                    print(f"  [IMU] {line}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="レーダーとIMUを同一時計で記録する")
    ap.add_argument("--list", action="store_true", help="シリアルポートを一覧表示して終了")
    ap.add_argument("--cfg", help="レーダーに送る .cfg ファイル")
    ap.add_argument("--radar-cfg", help="レーダーのCFGポート (115200)")
    ap.add_argument("--radar-data", help="レーダーのDATAポート (921600)")
    ap.add_argument("--imu", help="CoreS3 のポート")
    ap.add_argument("--imu-baud", type=int, default=921600)
    ap.add_argument("--out", default="run", help="出力ファイル名の接頭辞")
    ap.add_argument("--outdir", default=".", help="出力先ディレクトリ")
    ap.add_argument("--duration", type=float, default=0,
                    help="記録時間[秒]。0ならCtrl-Cまで")
    ap.add_argument("--no-radar", action="store_true", help="IMUだけ記録(D-1実験用)")
    args = ap.parse_args()

    if args.list:
        list_serial_ports()
        return

    if not args.no_radar and not (args.radar_data and args.cfg and args.radar_cfg):
        sys.exit("レーダーを使うには --cfg --radar-cfg --radar-data が必要です\n"
                 "(IMUだけ記録するなら --no-radar)")
    if not args.imu:
        sys.exit("--imu が必要です")

    os.makedirs(args.outdir, exist_ok=True)
    prefix = os.path.join(args.outdir, args.out)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{prefix}_{stamp}"

    stop_evt = threading.Event()
    t0 = time.perf_counter()
    wall0 = time.time()

    # --- IMUを先に開く(CoreS3は起動メッセージを出すので接続確認になる)
    imu = ImuReader(args.imu, prefix + "_imu.csv", t0, stop_evt, args.imu_baud)
    imu.start()
    time.sleep(1.0)
    if imu.error:
        sys.exit(imu.error)
    print(f"IMU    : {args.imu}  受信開始 ({imu.n} サンプル)")

    radar = None
    cfg_lines = []
    if not args.no_radar:
        radar = RadarReader(args.radar_data, prefix + ".dat",
                            prefix + "_frames.csv", t0, stop_evt)
        radar.start()
        time.sleep(0.3)
        if radar.error:
            stop_evt.set()
            sys.exit(radar.error)
        print(f"レーダー: {args.radar_data}  受信開始")
        print(f"設定送信: {args.cfg}")
        cfg_lines = send_cfg(args.radar_cfg, args.cfg)

    meta = dict(start_wall=wall0, start_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
                cfg=args.cfg, cfg_lines=cfg_lines,
                port_radar_cfg=args.radar_cfg, port_radar_data=args.radar_data,
                port_imu=args.imu, prefix=os.path.basename(prefix))
    with open(prefix + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print()
    print("記録中。止めるには Ctrl-C")
    print("-" * 64)
    try:
        while True:
            time.sleep(1.0)
            el = time.perf_counter() - t0
            if radar is not None:
                print(f"\r {el:7.1f}s | radar {radar.n_frames:6d} frames "
                      f"({radar.n_bytes/1024:8.1f} kB)  frame#={radar.last_frame} "
                      f"| imu {imu.n:7d} ({imu.rate:5.1f} Hz)   ", end="")
            else:
                print(f"\r {el:7.1f}s | imu {imu.n:7d} ({imu.rate:5.1f} Hz)   ",
                      end="")
            if args.duration and el >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print()
        stop_evt.set()
        time.sleep(0.4)
        if args.radar_cfg:
            send_stop(args.radar_cfg)

    print("-" * 64)
    print("保存しました:")
    if radar is not None:
        print(f"  {prefix}.dat          ({radar.n_bytes/1024:.1f} kB, "
              f"{radar.n_frames} frames)")
        print(f"  {prefix}_frames.csv")
    print(f"  {prefix}_imu.csv       ({imu.n} samples)")
    print(f"  {prefix}_meta.json")

    if radar is not None and radar.n_frames == 0:
        print()
        print("  ! レーダーのフレームが0件です。cfgが通っていないか、")
        print("    DATAポートの指定が違う可能性があります。")
    if imu.n == 0:
        print()
        print("  ! IMUのサンプルが0件です。CoreS3のUSB CDC設定を確認してください。")


if __name__ == "__main__":
    main()
