# -*- coding: utf-8 -*-
"""MCL が動いている様子を動画にする。

パーティクルが散って収束していく過程、観測点が壁に貼り付く様子、
曲がり角での挙動が目で追える。

  python animate_mcl.py data/run3_*.dat --map map_corridor.npz \
      --imu data/run3_*_imu.csv --frames data/run3_*_frames.csv \
      --init 0 0 90 --true-route route_corridor.csv --out mcl_run3.mp4
"""
import argparse
import glob
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from preprocess import parse_dat, to_scan, merge_scans, FRAME_PERIOD
from egomotion import estimate_velocity
from mcl import LikelihoodField, MCL
import imu as I

INK, SUB, BLUE, GREEN, ORANGE, RED, GRID = (
    "#1F1F28", "#5C5A6E", "#2E5FA3", "#2F7A5E", "#C77A15", "#B5484E", "#DDE3EC")


def expand(pat):
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(pat)
    return hits[0]


ap = argparse.ArgumentParser(description="MCL の動作を動画にする")
ap.add_argument("dat")
ap.add_argument("--map", default="map_corridor.npz")
ap.add_argument("--imu", default=None)
ap.add_argument("--frames", default=None)
ap.add_argument("--init", nargs=3, type=float, required=True,
                metavar=("X", "Y", "DEG"))
ap.add_argument("--true-route", default=None)
ap.add_argument("--particles", type=int, default=4000)
ap.add_argument("--step", type=int, default=3)
ap.add_argument("--out", default="mcl.mp4", help=".mp4 か .gif")
ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--every", type=int, default=1, help="何更新ごとに1コマ描くか")
ap.add_argument("--show-particles", type=int, default=700,
                help="描くパーティクル数（全部描くと重いので間引く）")
ap.add_argument("--zoom", type=float, default=2.2,
                help="追従パネルが映す範囲 [m]。0 で全体のみ")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

np.random.seed(a.seed)

# ---------------------------------------------------------- データとMCLの用意
scans = to_scan(parse_dat(expand(a.dat)))
frames = sorted(scans)
use_imu = bool(a.imu and a.frames)
if use_imu:
    fr = I.load_frames(expand(a.frames))
    d = I.load_imu(expand(a.imu))
    tb = I.dtheta_per_frame(d, fr, I.estimate_bias(d)["bias"])
    dth = dict(zip(tb.frame.values, tb.dtheta.values))
    ftime = dict(zip(fr.frame.values, fr.t.values))    # フレーム番号 → 時刻[s]
else:
    dth, ftime = {}, {}

lf = LikelihoodField(a.map)
m = MCL(lf, a.init[0], a.init[1], np.radians(a.init[2]))
m.set_particle_num(a.particles)
m.reset_particles(0.2, 0.2, np.radians(8.0))

TRUE = np.loadtxt(expand(a.true_route), delimiter=",", skiprows=1) \
    if a.true_route else None

# ---------------------------------------------------------- 先に全部回して記録
# 描画のたびにMCLを進めると巻き戻せないので、1回通してコマを貯めておく
print("MCL を回しています...")
t0 = time.time()
snaps = []
for i in range(0, len(frames) - a.step, a.step):
    fs = frames[i:i + a.step]
    sc = merge_scans(scans, fs)
    if len(sc["r"]) == 0:
        continue
    e = estimate_velocity(sc["r"], sc["th"], sc["doppler"])
    dist = (e["vx"] if e["ok"] else 0.0) * a.step * FRAME_PERIOD
    rot = float(sum(dth.get(f, 0.0) for f in fs)) if use_imu else 0.0

    m.update_particles(dist, rot)
    m.calculate_weights(sc["r"], sc["th"], sc["snr"])
    m.estimate_robot_pose()
    x, y, yaw = m.pose()
    # 重みの大きい順に間引く（薄い粒子ばかり描いても見えないため）
    idx = np.argsort(m.weights)[-a.show_particles:]
    snaps.append(dict(
        P=m.particles[idx].copy(), w=m.weights[idx].copy(),
        pose=(x, y, yaw), ess=m.effective_sample_size, spread=m.spread()[0],
        # 観測点を推定姿勢で地図座標へ写したもの（MCLが「こう見えている」図）
        sx=x + np.cos(yaw) * (sc["r"] * np.cos(sc["th"]))
           - np.sin(yaw) * (sc["r"] * np.sin(sc["th"])),
        sy=y + np.sin(yaw) * (sc["r"] * np.cos(sc["th"]))
           + np.cos(yaw) * (sc["r"] * np.sin(sc["th"])),
        t=ftime.get(fs[0], np.nan)))
    m.resample_particles()
print(f"  {len(snaps)} 更新 / {time.time() - t0:.1f} 秒")

snaps = snaps[::a.every]
traj = np.array([s["pose"][:2] for s in snaps])

# ---------------------------------------------------------- 描画の下地
ext = [lf.origin[0], lf.origin[0] + lf.W * lf.res,
       lf.origin[1], lf.origin[1] + lf.H * lf.res]
two = a.zoom > 0
fig, axes = plt.subplots(1, 2 if two else 1, figsize=(12.6 if two else 7.2, 8.6),
                         dpi=110, gridspec_kw={"width_ratios": [1, 1]} if two else None)
axes = list(np.atleast_1d(axes))

art = []     # パネルごとの描画オブジェクトをまとめて持つ
for j, ax in enumerate(axes):
    ax.imshow(np.where(lf.occ, 1.0, np.nan), origin="lower", extent=ext,
              cmap="gray", vmin=0, vmax=2, interpolation="nearest", zorder=1)
    if TRUE is not None:
        ax.plot(TRUE[:, 0], TRUE[:, 1], "-", color=GREEN, lw=3.0, alpha=0.5,
                zorder=2, label="true route")
    if j == 0:                     # 左 = 全体
        ax.set_xlim(-8.0, 3.2); ax.set_ylim(-1.5, 14.2)
        ps, pa = 6, 0.30           # パーティクルの大きさと濃さ
        ss, hw = 26, 0.8
    else:                          # 右 = 推定姿勢を追いかける拡大
        ps, pa = 26, 0.45
        ss, hw = 60, 0.45
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]", color=SUB)
    ax.grid(color=GRID, lw=0.7, alpha=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors=SUB, labelsize=9)
    art.append(dict(
        ax=ax, hw=hw,
        pts=ax.scatter([], [], s=ps, c=BLUE, alpha=pa, zorder=3,
                       label="particles"),
        scan=ax.scatter([], [], s=ss, c=ORANGE, edgecolor="white",
                        linewidth=0.5, zorder=5, label="scan (projected)"),
        trail=ax.plot([], [], "-", color=BLUE, lw=2.0, zorder=4,
                      label="estimate")[0],
        head=ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=6,
                         arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2))))
axes[0].set_ylabel("Y [m]", color=SUB)
axes[0].legend(fontsize=9, loc="lower left", framealpha=0.95)
if two:
    # 日本語フォントは環境依存なので、図の中の文字は ASCII に揃える
    axes[1].set_title(f"follow the estimate  (+/- {a.zoom:.1f} m)",
                      fontsize=11, color=INK)
fig.tight_layout()
txt = axes[0].text(0.02, 0.985, "", transform=axes[0].transAxes, va="top",
                   fontsize=10, color=INK, family="monospace", zorder=7,
                   bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))


def draw(k):
    s = snaps[k]
    x, y, yaw = s["pose"]
    for j, A in enumerate(art):
        A["pts"].set_offsets(s["P"][:, :2])
        A["scan"].set_offsets(np.stack([s["sx"], s["sy"]], axis=1))
        A["trail"].set_data(traj[:k + 1, 0], traj[:k + 1, 1])
        A["head"].set_position((x, y))
        A["head"].xy = (x + A["hw"] * np.cos(yaw), y + A["hw"] * np.sin(yaw))
        if j == 1:                      # 追従パネルは毎コマ窓を動かす
            A["ax"].set_xlim(x - a.zoom, x + a.zoom)
            A["ax"].set_ylim(y - a.zoom, y + a.zoom)
    # --frames を渡していないと時刻が無いので、そのときはステップだけ出す
    tt = f"t = {s['t']:5.1f} s   " if np.isfinite(s["t"]) else ""
    txt.set_text(f"{tt}step {k+1}/{len(snaps)}"
                 f"{'' if use_imu else '   (no IMU: rotation unobserved)'}\n"
                 f"pose ({x:+5.2f}, {y:+5.2f}) {np.degrees(yaw):+6.1f} deg\n"
                 f"ESS {s['ess']:4.0f}/{a.particles}   spread {s['spread']*100:4.0f} cm")
    return []


print(f"{len(snaps)} コマを書き出しています...")
anim = FuncAnimation(fig, draw, frames=len(snaps), interval=1000 // a.fps,
                     blit=False)
if a.out.lower().endswith(".gif"):
    anim.save(a.out, writer=PillowWriter(fps=a.fps))
else:
    anim.save(a.out, writer=FFMpegWriter(fps=a.fps, bitrate=2400))
plt.close(fig)
print(f"保存しました: {a.out}  ({len(snaps)/a.fps:.1f} 秒)")
