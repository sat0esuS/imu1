# -*- coding: utf-8 -*-
"""mcl.py（pyloc を土台にしたミリ波用 MCL）に、記録したデータを流す。

  python run_mcl.py data/run3_*.dat --map map_corridor.npz \
      --imu data/run3_*_imu.csv --frames data/run3_*_frames.csv \
      --init 0 0 90 --true-route route_corridor.csv --out mcl_run3.png
"""
import argparse
import glob
import time

import numpy as np

from preprocess import parse_dat, to_scan, merge_scans, FRAME_PERIOD
from egomotion import estimate_velocity
from mcl import LikelihoodField, MCL
import imu as I

INK, SUB, BLUE, GREEN, GRID = "#1F1F28", "#5C5A6E", "#2E5FA3", "#2F7A5E", "#DDE3EC"


def expand(pat):
    """PowerShell は * を展開しないので自前で展開する。"""
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(pat)
    return hits[0]


ap = argparse.ArgumentParser(description="ミリ波用 MCL を実データで動かす")
ap.add_argument("dat")
ap.add_argument("--map", default="map_corridor.npz")
ap.add_argument("--imu", default=None, help="省略すると回転なしで走る")
ap.add_argument("--frames", default=None)
ap.add_argument("--init", nargs=3, type=float, required=True,
                metavar=("X", "Y", "DEG"))
ap.add_argument("--true-route", default=None, help="正解ルートの頂点 CSV (x,y)")
ap.add_argument("--particles", type=int, default=4000)
ap.add_argument("--step", type=int, default=3, help="何フレームずつ1更新にするか")
ap.add_argument("--floor", type=float, default=None, help="一様床（既定 1/max_dist）")
ap.add_argument("--odom-floor", type=float, default=0.12, help="動作ノイズの下限 [m]")
ap.add_argument("--meas-var", type=float, default=0.01, help="観測分散 [m^2]")
ap.add_argument("--no-snr", action="store_true", help="SNRの重みを使わない")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="軌跡の図を書き出す PNG")
ap.add_argument("--traj", default=None, help="軌跡を書き出す CSV")
a = ap.parse_args()

np.random.seed(a.seed)

# --- データを読む
scans = to_scan(parse_dat(expand(a.dat)))
frames = sorted(scans)
use_imu = bool(a.imu and a.frames)
if use_imu:
    fr = I.load_frames(expand(a.frames))
    d = I.load_imu(expand(a.imu))
    bias = I.estimate_bias(d)["bias"]
    tb = I.dtheta_per_frame(d, fr, bias)
    dth = dict(zip(tb.frame.values, tb.dtheta.values))   # フレーム番号 → 回転量
    print(f"IMU: {len(d)} サンプル, バイアス {bias:+.5f} deg/s")
else:
    dth = {}
    print("IMU なし（回転は観測しない）")

# --- MCL を用意する
lf = LikelihoodField(a.map)
print(f"地図: {lf.W}x{lf.H} セル ({lf.res*100:.0f} cm), 壁 {lf.occ.sum()} セル")
m = MCL(lf, a.init[0], a.init[1], np.radians(a.init[2]))
m.set_particle_num(a.particles)
m.set_odom_noise_floor(a.odom_floor)
m.set_measurement_variance(a.meas_var)
if a.floor is not None:
    m.set_measurement_floor(a.floor)
m.use_snr = not a.no_snr
m.reset_particles(0.2, 0.2, np.radians(8.0))
print(f"一様床 {m.measurement_floor:g} / 動作ノイズ下限 {m.odom_floor_fw:g} m / "
      f"SNR重み {'あり' if m.use_snr else 'なし'}")

# --- 1更新ずつ回す
t0 = time.time()
traj, n_skip = [], 0
for i in range(0, len(frames) - a.step, a.step):
    fs = frames[i:i + a.step]
    sc = merge_scans(scans, fs)              # 数フレームぶんの点群をまとめる
    if len(sc["r"]) == 0:
        n_skip += 1
        continue
    # 前進量: レーダーのドップラーから自己速度を解いて、経過時間を掛ける
    e = estimate_velocity(sc["r"], sc["th"], sc["doppler"])
    dist = (e["vx"] if e["ok"] else 0.0) * a.step * FRAME_PERIOD
    # 回転量: IMU の積分をこのフレーム区間で切り出す
    rot = float(sum(dth.get(f, 0.0) for f in fs)) if use_imu else 0.0

    m.update_particles(dist, rot)            # 動作モデル
    m.calculate_weights(sc["r"], sc["th"], sc["snr"])   # 観測モデル
    m.estimate_robot_pose()                  # 姿勢推定
    m.resample_particles()                   # 必要ならリサンプリング
    x, y, yaw = m.pose()
    traj.append((x, y, yaw, m.effective_sample_size, m.spread()[0]))

traj = np.array(traj)
print(f"\n処理ステップ数 {len(traj)}（点が無くて飛ばした: {n_skip}）"
      f"   {time.time() - t0:.1f} 秒")
print(f"最終姿勢 ({traj[-1,0]:+.2f}, {traj[-1,1]:+.2f}, {np.degrees(traj[-1,2]):+.1f}°)"
      f"   ESS {traj[-1,3]:.0f}/{a.particles}   広がり {traj[-1,4]*100:.0f} cm")

TRUE = None
if a.true_route:
    TRUE = np.loadtxt(a.true_route, delimiter=",", skiprows=1)

    def dist_to_route(P):
        """各推定点から、折れ線としての正解ルートまでの最短距離。"""
        out = []
        for x, y in P:
            dmin = 1e9
            for k in range(len(TRUE) - 1):
                p, q = TRUE[k], TRUE[k + 1]
                v = q - p
                # 線分上に射影し、0〜1 に丸めてから距離を測る
                u = np.clip(np.dot(np.array([x, y]) - p, v) / np.dot(v, v), 0, 1)
                dmin = min(dmin, float(np.hypot(*(np.array([x, y]) - (p + u * v)))))
            out.append(dmin)
        return np.array(out)

    dd = dist_to_route(traj[:, :2])
    end = float(np.hypot(traj[-1, 0] - TRUE[-1, 0], traj[-1, 1] - TRUE[-1, 1]))
    print(f"\n=== 誤差評価 ===")
    print(f"終点の誤差       {end:.2f} m   （正解 {TRUE[-1,0]:+.2f}, {TRUE[-1,1]:+.2f}）")
    print(f"ルートからの距離  平均 {dd.mean():.2f} / 中央値 {np.median(dd):.2f} "
          f"/ 最大 {dd.max():.2f} m")

if a.traj:
    np.savetxt(a.traj, traj, delimiter=",", comments="",
               header="x,y,yaw_rad,ess,spread_m")
    print(f"軌跡を書き出しました: {a.traj}")

if a.out:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [lf.origin[0], lf.origin[0] + lf.W * lf.res,
           lf.origin[1], lf.origin[1] + lf.H * lf.res]
    fig, ax = plt.subplots(figsize=(7.4, 8.2), dpi=160)
    ax.imshow(np.where(lf.occ, 1.0, np.nan), origin="lower", extent=ext,
              cmap="gray", vmin=0, vmax=2, interpolation="nearest", zorder=1)
    if TRUE is not None:
        ax.plot(TRUE[:, 0], TRUE[:, 1], "-o", color=GREEN, lw=3.0, ms=8,
                alpha=0.75, label="true route", zorder=3)
    ax.plot(traj[:, 0], traj[:, 1], "-", color=BLUE, lw=2.2,
            label="MCL estimate", zorder=4)
    ax.plot([a.init[0]], [a.init[1]], "^", color=INK, ms=13, zorder=6)
    ax.plot([traj[-1, 0]], [traj[-1, 1]], "X", color=BLUE, ms=13, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]", color=SUB)
    ax.set_ylabel("Y [m]", color=SUB)
    ax.grid(color=GRID, lw=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"図を書き出しました: {a.out}")
