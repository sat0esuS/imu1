# -*- coding: utf-8 -*-
"""logger.py の打刻精度を、偽のシリアルポートを使って測る。

実機では「レーダーがいつ送ったか」が分からないので、打刻の誤差を測れない。
測りたいものを使って測ることになってしまう。
そこでレーダーの代わりに自分で書いた送信プログラムを置き、
「このフレームを t=X 秒に送った」という正解を自分で持つ。
logger.py の RadarReader が「t=Y 秒に届いた」と書いたら、|Y-X| が打刻誤差になる。

偽のポートは OS の疑似端末 (pty) で作る。master に書くと slave から出てくる
ただの筒で、slave 側は /dev/pts/3 のようなパスを持ち、pyserial から実機の
ポートと同じように開ける。Windows には pty が無いので WSL か Linux で動かす。

  python test_timestamp.py                # 200フレーム（約13秒）
  python test_timestamp.py --frames 500 --naive

測れるもの:
  ・OS のバッファにフレームが寝てしまう分
  ・バイト位置から先頭時刻へ巻き戻す補正が合っているか
  ・マジックワードの取りこぼし、二重検出

測れないもの:
  ・USBシリアル変換チップ (CP2105) の内部遅れ  ← 偽ポートに実物のチップが無い
  ・レーダー内部の処理遅れ                      ← 送信時刻を自分で決めているので
  これらは実機の timeCpuCycles との突き合わせで別に確かめる。
"""
import argparse
import os
import pty
import struct
import sys
import threading
import time

import numpy as np

import logger
from logger import RadarReader, MAGIC, BYTES_PER_SEC

FRAME_PERIOD = 1.0 / 15.0          # cfg から決まるフレーム周期 [s]
FRAME_BYTES = 1471                 # 実測した 1 フレームのバイト数
CHUNK = 128                        # 何バイトずつ書き込むか（伝送を刻んで真似る）


def build_frame(n):
    """TI の mmWave デモと同じ形のヘッダを持つフレームを 1 つ作る。

    RadarReader はマジックワードの直後 32 バイトを 8 個の uint32 として読み、
    2 番目を全長、4 番目をフレーム番号として使う。そこだけ合っていればよい。
    """
    hdr = struct.pack(
        "<8I",
        0x03060000,                      # version
        FRAME_BYTES,                     # totalPacketLen  ← 読まれる
        0x000A6843,                      # platform
        n,                               # frameNumber     ← 読まれる
        int(n * FRAME_PERIOD * 200e6),   # timeCpuCycles (200MHz カウンタ)
        3,                               # numDetectedObj
        2,                               # numTLVs
        0,                               # subFrameNumber
    )
    body = bytes(FRAME_BYTES - len(MAGIC) - len(hdr))   # 中身は 0 埋めでよい
    return MAGIC + hdr + body


def wait_until(t):
    """perf_counter が t を越えるまで待つ。

    time.sleep は 1ms 以下の精度が出ないので、残りが少なくなったら回して待つ。
    ここが甘いと、測りたい誤差より送信側のばらつきが大きくなってしまう。
    """
    while True:
        rest = t - time.perf_counter()
        if rest <= 0:
            return
        if rest > 0.002:
            time.sleep(rest - 0.002)


def sender(fd, t0, n_frames, truth):
    """フレームを 921600 bps の速さで書き込み、先頭を書いた時刻を正解として残す。

    1 フレーム 1471 バイトは 921600 bps で 15.96 ms かかる。
    一度に書くと伝送時間が消えてしまい、バイト位置からの巻き戻し補正が
    検証できないので、128 バイトずつに刻んで実際の速さで送る。
    """
    for n in range(n_frames):
        t_send = t0 + n * FRAME_PERIOD          # このフレームを出す予定時刻
        wait_until(t_send)
        frame = build_frame(n)
        # 先頭バイトを書いた瞬間が正解。これと frames.csv の値を比べる
        truth[n] = time.perf_counter() - t0
        for i in range(0, len(frame), CHUNK):
            os.write(fd, frame[i:i + CHUNK])
            # 次のかたまりを書くまで、そのバイト数の伝送にかかる時間だけ待つ
            wait_until(t_send + (i + CHUNK) / BYTES_PER_SEC)


ap = argparse.ArgumentParser(description="logger.py の打刻精度を偽シリアルで測る")
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--naive", action="store_true",
                help="工夫を入れていない読み方も試して比べる")
ap.add_argument("--out", default="/tmp/ts_test")
a = ap.parse_args()

if sys.platform.startswith("win"):
    sys.exit("pty が無いので Windows では動きません。WSL か Linux で実行してください。")


def run(naive):
    """偽ポートを用意し、送信と受信を同時に走らせて誤差を返す。"""
    master, slave = pty.openpty()                  # master に書くと slave から出る
    port = os.ttyname(slave)                       # /dev/pts/N のようなパス
    t0 = time.perf_counter()                       # 送信側と受信側で同じ時計を使う
    truth = {}                                     # フレーム番号 → 送った時刻
    stop = threading.Event()

    rd = RadarReader(port, f"{a.out}.dat", f"{a.out}_frames.csv", t0, stop)

    saved_serial, saved_rate = logger.serial.Serial, logger.BYTES_PER_SEC
    if naive:
        # 工夫を 2 つとも外して、何も考えずに書いた場合の打刻に戻す。
        # ① 届いている分だけ取るのをやめ、8192 バイト溜まるまで待つ読み方にする。
        #    レーダーは 9 kB/s しか吐かないので必ずタイムアウト待ちになり、
        #    フレームが OS のバッファで最大 50 ms 寝てから打刻される。
        # ② バイト位置からの巻き戻しを無効にする。レートを巨大にすると
        #    補正項 (バイト数 / レート) が 0 になり、補正なしと同じになる。
        class Blocking:
            def __init__(self, p, baud, timeout=None):
                self._s = saved_serial(p, baud, timeout=0.05)

            @property
            def in_waiting(self):
                return 0          # 「まだ来ていない」と嘘をついて read 側へ流す

            def read(self, n):
                return self._s.read(8192) if n else b""

            def __enter__(self):
                return self

            def __exit__(self, *e):
                self._s.close()

        logger.serial.Serial = Blocking
        logger.BYTES_PER_SEC = 1e18

    th = threading.Thread(target=sender, args=(master, t0, a.frames, truth),
                          daemon=True)
    try:
        rd.start()
        th.start()
        th.join()
        time.sleep(0.3)                            # 残りが読まれるのを待つ
        stop.set()
        rd.join(timeout=2.0)
    finally:
        # 差し替えたものを必ず戻す。戻し忘れると次の測定が汚れる
        logger.serial.Serial, logger.BYTES_PER_SEC = saved_serial, saved_rate
        os.close(master)
        os.close(slave)
    if rd.error:
        sys.exit(rd.error)

    # frames.csv を読んで、同じフレーム番号の正解との差を取る
    got = {}
    with open(f"{a.out}_frames.csv", encoding="utf-8") as f:
        next(f)
        for line in f:
            t_host, frame, _ = line.split(",")
            got[int(frame)] = float(t_host)
    common = sorted(set(truth) & set(got))
    err = np.array([abs(got[n] - truth[n]) for n in common]) * 1e3   # ms
    return err, len(common), len(truth)


print(f"偽シリアルで {a.frames} フレーム送ります（約 {a.frames*FRAME_PERIOD:.0f} 秒）\n")

err, n_got, n_sent = run(naive=False)
print("=== 工夫あり（届いている分だけ即座に読む + バイト位置からの巻き戻し）===")
print(f"受信フレーム   {n_got}/{n_sent}")
print(f"打刻誤差       平均 {err.mean():.2f} ms / 中央値 {np.median(err):.2f} ms "
      f"/ 最大 {err.max():.2f} ms")
print(f"換算（56 deg/s の旋回時の向きの誤差）  {err.mean()/1e3*56:.3f} deg")

if a.naive:
    err2, n2, _ = run(naive=True)
    print("\n=== 工夫なし（タイムアウト待ちで読み、巻き戻しをしない）===")
    print(f"受信フレーム   {n2}/{n_sent}")
    print(f"打刻誤差       平均 {err2.mean():.2f} ms / 中央値 {np.median(err2):.2f} ms "
          f"/ 最大 {err2.max():.2f} ms")
    # 平均と最大は OS の都合で時々飛ぶ外れ値に引っ張られるので、
    # 中央値での比を主に見る
    print(f"\n改善           中央値 {np.median(err2)/np.median(err):.0f} 倍 "
          f"/ 平均 {err2.mean()/err.mean():.0f} 倍")
