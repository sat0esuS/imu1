# -*- coding: utf-8 -*-
"""移植した pyloc の MCL に、このリポジトリで記録したデータを流す。

  python run_pyloc.py ../data/run3_*.dat --map ../map_corridor.npz \
      --imu ../data/run3_*_imu.csv --frames ../data/run3_*_frames.csv \
      --init 0 0 90 --true-route ../route_corridor.csv --out pyloc_run3.png

一様床 (--floor) だけは原典の既定値 10e-6 から変えてある。理由は README を参照。
"""
import argparse
import time

import numpy as np

from common import load_run, steps, landmarks_from_map
from pyloc_mcl import MCL

INK, SUB, BLUE, ORANGE, GREEN, RED, GRID = (
    "#1F1F28", "#5C5A6E", "#2E5FA3", "#C77A15", "#2F7A5E", "#B5484E", "#DDE3EC")

ap = argparse.ArgumentParser(description="pyloc の MCL を実データで動かす")
ap.add_argument("dat")
ap.add_argument("--map", default="../map_corridor.npz")
ap.add_argument("--imu", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--init", nargs=3, type=float, required=True,
                metavar=("X", "Y", "DEG"))
ap.add_argument("--true-route", default=None,
                help="正解ルートの頂点 CSV (x,y)。誤差評価と作図に使う")
ap.add_argument("--particles", type=int, default=4000)
ap.add_argument("--floor", type=float, default=0.1,
                help="一様床。原典は 10e-6 固定だったが、ミリ波の疎な点群では "
                     "外れ値1個で正解が却下されるため既定を上げてある")
ap.add_argument("--meas-var", type=float, default=0.01,
                help="等方の観測分散 [m^2]。pyloc は1つしか持てない")
ap.add_argument("--odom-noise", nargs=4, type=float,
                default=[20.0, 0.5, 2.0, 0.3],
                help="原典の odom_noise1..4")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="軌跡の図を書き出す PNG")
ap.add_argument("--traj", default=None, help="軌跡を書き出す CSV")
a = ap.parse_args()

np.random.seed(a.seed)
scans, dth, frames = load_run(a.dat, a.imu, a.frames)
lm = landmarks_from_map(a.map)
print(f"占有セル {len(lm)} 個をランドマークとして使用")
print(f"一様床 {a.floor:g}（原典の既定値は 1e-05）")

m = MCL(a.init[0], a.init[1], np.radians(a.init[2]))
m.set_particle_num(a.particles)
m.set_odom_noises(*a.odom_noise)
m.set_landmarks(lm)
m.set_measurement_variance(a.meas_var)
m.set_measurement_resolution(0.1)
m.set_measurement_model_coefficients(0.9, 0.1)
m.set_measurement_floor(a.floor)
m.set_resample_threshold(0.5)
m.reset_particles(0.2, 0.2, np.radians(8.0))

t0 = time.time()
traj = []
for dist, rot, r, th, snr in steps(scans, dth, frames):
    m.update_particles(dist, rot)
    m.calculate_weights(np.stack([r, th], axis=1))
    m.estimate_robot_pose()
    m.resample_particles()
    traj.append((m.robot_pose_x, m.robot_pose_y, m.robot_pose_yaw,
                 m.effective_sample_size))
traj = np.array(traj)

print(f"処理ステップ数: {len(traj)}   {time.time() - t0:.1f} 秒")
print(f"最終姿勢: ({traj[-1, 0]:.2f}, {traj[-1, 1]:.2f}, "
      f"{np.degrees(traj[-1, 2]):+.1f}°)   ESS {traj[-1, 3]:.0f}/{a.particles}")

TRUE = None
if a.true_route:
    TRUE = np.loadtxt(a.true_route, delimiter=",", skiprows=1)

    def dist_to_route(P):
        out = []
        for x, y in P:
            dmin = 1e9
            for i in range(len(TRUE) - 1):
                p, q = TRUE[i], TRUE[i + 1]
                v = q - p
                u = np.clip(np.dot(np.array([x, y]) - p, v) / np.dot(v, v), 0, 1)
                dmin = min(dmin, float(np.hypot(*(np.array([x, y]) - (p + u * v)))))
            out.append(dmin)
        return np.array(out)

    d = dist_to_route(traj[:, :2])
    end_err = float(np.hypot(traj[-1, 0] - TRUE[-1, 0], traj[-1, 1] - TRUE[-1, 1]))
    print(f"\n=== 誤差評価 ===")
    print(f"終点の誤差      : {end_err:.2f} m   "
          f"（正解 {TRUE[-1, 0]:+.2f}, {TRUE[-1, 1]:+.2f}）")
    print(f"ルートからの距離: 平均 {d.mean():.2f} m / 中央値 {np.median(d):.2f} m "
          f"/ 最大 {d.max():.2f} m")

if a.traj:
    np.savetxt(a.traj, traj, delimiter=",", comments="",
               header="x,y,yaw_rad,ess")
    print(f"軌跡を書き出しました: {a.traj}")

if a.out:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        font_manager.fontManager.addfont(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False

    z = np.load(a.map)
    occ, res, org = z["occ"], float(z["resolution"]), z["origin"]
    ext = [org[0], org[0] + occ.shape[1] * res,
           org[1], org[1] + occ.shape[0] * res]
    fig, ax = plt.subplots(figsize=(7.4, 8.2), dpi=160)
    ax.imshow(np.where(occ, 1.0, np.nan), origin="lower", extent=ext,
              cmap="gray", vmin=0, vmax=2, interpolation="nearest", zorder=1)
    if TRUE is not None:
        ax.plot(TRUE[:, 0], TRUE[:, 1], "-o", color=GREEN, lw=3.0, ms=8,
                alpha=0.75, label="実測ルート（正解）", zorder=3)
    ax.plot(traj[:, 0], traj[:, 1], "-", color=BLUE, lw=2.2,
            label="pyloc の推定", zorder=4)
    ax.plot([a.init[0]], [a.init[1]], "^", color=INK, ms=13, zorder=6)
    ax.plot([traj[-1, 0]], [traj[-1, 1]], "X", color=BLUE, ms=13, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]", color=SUB)
    ax.set_ylabel("Y [m]", color=SUB)
    ax.grid(color=GRID, lw=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=SUB, labelsize=9.5)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax.set_title(f"pyloc の MCL（一様床 {a.floor:g}、{a.particles} パーティクル）",
                 fontsize=12.5, color=INK, fontweight="bold")
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"図を書き出しました: {a.out}")
