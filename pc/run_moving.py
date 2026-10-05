# -*- coding: utf-8 -*-
"""
移動データ(台車走行)でのMCL

使い方:
    # ドップラーで自己速度を推定して動作モデルに使う(推奨)
    python3 run_moving.py 走行.dat --init 0.9 1.0 90 --motion doppler

    # 一定速度を仮定する場合
    python3 run_moving.py 走行.dat --init 0.9 1.0 90 --motion const --speed 0.3

    # 正解軌跡がある場合(テープ通過時刻から作ったCSV: t,x,y)
    python3 run_moving.py 走行.dat --init 0.9 1.0 90 --gt truth.csv
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import os
from matplotlib import font_manager
for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
           "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc",
           "C:/Windows/Fonts/msgothic.ttc",
           "/System/Library/Fonts/ttf/HiraginoSans-W3.ttc",
           "/System/Library/Fonts/Hiragino Sans GB.ttc"):
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
            matplotlib.rcParams["font.family"] = \
                font_manager.FontProperties(fname=_p).get_name()
        except Exception:
            pass
        break
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from mcl import LikelihoodField, MCL
from preprocess import parse_dat, to_scan, merge_scans, FRAME_PERIOD
from egomotion import estimate_velocity

ap = argparse.ArgumentParser()
ap.add_argument("dat")
ap.add_argument("--map", default="map.npz")
ap.add_argument("--init", nargs=3, type=float, required=True,
                metavar=("X", "Y", "DEG"), help="出発点の姿勢")
ap.add_argument("--motion", choices=["doppler", "const", "none"], default="doppler")
ap.add_argument("--speed", type=float, default=0.3, help="const時の速度 [m/s]")
ap.add_argument("--vmax", type=float, default=0.61, help="cfgのv_max")
ap.add_argument("--step", type=int, default=3, help="何フレームずつ処理するか")
ap.add_argument("--particles", type=int, default=4000)
ap.add_argument("--gt", default=None, help="正解軌跡CSV (t,x,y)")
ap.add_argument("--out", default="mcl_moving.png")
a = ap.parse_args()

lf = LikelihoodField(a.map)
df = parse_dat(a.dat)
scans = to_scan(df)
frames = sorted(scans)
print(f"読み込み: {len(frames)} フレーム ({len(frames)*FRAME_PERIOD:.1f} 秒), {len(df)} 点")

m = MCL(lf, n_particles=a.particles, motion_noise=(0.06, 0.04, 0.06))
m.init_around(a.init[0], a.init[1], a.init[2], s_xy=0.2, s_deg=8)

traj, vels = [], []
for i in range(0, len(frames) - a.step, a.step):
    fs = frames[i:i + a.step]
    sc = merge_scans(scans, fs)
    if len(sc["r"]) == 0:
        continue
    dt = a.step * FRAME_PERIOD

    # --- 予測 (動作モデル)
    if a.motion == "doppler":
        est = estimate_velocity(sc["r"], sc["th"], sc["doppler"], v_max=a.vmax)
        v = est["vx"] if est["ok"] else 0.0
        vels.append((i * FRAME_PERIOD, v, est["ok"], est["ratio"]))
        m.predict(dx=v * dt)
    elif a.motion == "const":
        m.predict(dx=a.speed * dt)
    else:
        m.predict()

    # --- 更新
    m.update(sc["r"], sc["th"], sc["snr"])
    e, mass = m.estimate_mode()
    traj.append((i * FRAME_PERIOD, e[0], e[1], e[2], m.spread()[0], mass))

traj = np.array(traj)
print(f"処理ステップ数: {len(traj)}")
if len(vels):
    V = np.array([(t, v) for t, v, ok, _ in vels if ok])
    if len(V):
        print(f"ドップラー速度推定: 有効 {len(V)}/{len(vels)} ステップ, "
              f"中央値 {np.median(V[:,1]):.2f} m/s, 範囲 [{V[:,1].min():.2f}, {V[:,1].max():.2f}]")
print(f"最終姿勢: ({traj[-1,1]:.2f}, {traj[-1,2]:.2f}, {np.degrees(traj[-1,3]):+.1f}°), "
      f"広がり {traj[-1,4]*100:.0f} cm")

# --- 正解軌跡との比較
gt = None
if a.gt:
    gt = np.loadtxt(a.gt, delimiter=",", skiprows=1)
    gx = np.interp(traj[:, 0], gt[:, 0], gt[:, 1])
    gy = np.interp(traj[:, 0], gt[:, 0], gt[:, 2])
    err = np.hypot(traj[:, 1] - gx, traj[:, 2] - gy)
    # 進行方向(縦)と横方向に分解
    dxy = np.gradient(np.stack([gx, gy], axis=1), axis=0)
    nrm = np.linalg.norm(dxy, axis=1, keepdims=True)
    u = dxy / np.maximum(nrm, 1e-9)
    dv = np.stack([traj[:, 1] - gx, traj[:, 2] - gy], axis=1)
    e_long = np.abs((dv * u).sum(axis=1))
    e_lat = np.abs(dv[:, 0] * (-u[:, 1]) + dv[:, 1] * u[:, 0])
    print(f"\n=== 誤差評価 ===")
    print(f"全体   : 平均 {err.mean()*100:5.1f} cm, 中央値 {np.median(err)*100:5.1f} cm, "
          f"最大 {err.max()*100:5.1f} cm")
    print(f"縦方向 : 平均 {e_long.mean()*100:5.1f} cm  (廊下方向・縮退が出る軸)")
    print(f"横方向 : 平均 {e_lat.mean()*100:5.1f} cm  (壁で拘束される軸)")

# --- 描画
ext = [lf.origin[0], lf.origin[0] + lf.W * lf.res,
       lf.origin[1], lf.origin[1] + lf.H * lf.res]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
ax = axes[0]
ax.imshow(np.where(lf.occ, 0, np.nan), origin="lower", extent=ext, cmap="gray_r", vmin=0, vmax=1)
sc_ = ax.scatter(traj[:, 1], traj[:, 2], c=traj[:, 0], cmap="viridis", s=14)
if gt is not None:
    ax.plot(gt[:, 1], gt[:, 2], "r--", lw=1.5, label="正解軌跡")
    ax.legend(fontsize=8)
ax.plot(a.init[0], a.init[1], "k^", ms=10)
ax.set_aspect("equal"); ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_title("推定軌跡 (色 = 時刻)")
plt.colorbar(sc_, ax=ax, shrink=0.8, label="時刻 [s]")

ax = axes[1]
if gt is not None:
    ax.plot(traj[:, 0], err * 100, color="#D85A30", label="全体誤差")
    ax.plot(traj[:, 0], e_long * 100, color="#534AB7", ls="--", label="縦方向")
    ax.plot(traj[:, 0], e_lat * 100, color="#1D9E75", ls=":", label="横方向")
    ax.set_ylabel("誤差 [cm]"); ax.legend(fontsize=8)
else:
    ax.plot(traj[:, 0], traj[:, 4] * 100, color="#534AB7")
    ax.set_ylabel("パーティクルの広がり [cm]")
ax.set_xlabel("時刻 [s]"); ax.grid(alpha=0.3); ax.set_title("誤差 / 収束の推移")

ax = axes[2]
if len(vels):
    V = np.array([(t, v, float(ok)) for t, v, ok, _ in vels])
    ax.plot(V[:, 0], V[:, 1], ".", color="#1D9E75", label="ドップラー速度")
    bad = V[V[:, 2] < 0.5]
    if len(bad):
        ax.plot(bad[:, 0], bad[:, 1], "x", color="#D85A30", label="棄却")
    ax.axhline(a.vmax, color="gray", ls=":", label="v_max")
    ax.set_ylabel("速度 [m/s]"); ax.legend(fontsize=8)
else:
    ax.plot(traj[:, 0], traj[:, 5], color="#EF9F27")
    ax.set_ylabel("最大クラスタの重み")
ax.set_xlabel("時刻 [s]"); ax.grid(alpha=0.3); ax.set_title("自己速度推定")

fig.suptitle(f"移動MCL: {a.dat.split('/')[-1]} (動作モデル={a.motion})", fontsize=12)
fig.tight_layout(); fig.savefig(a.out, dpi=115)
print(f"\nsaved: {a.out}")
