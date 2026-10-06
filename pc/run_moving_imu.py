# -*- coding: utf-8 -*-
"""
移動データ(台車走行)でのMCL  ---  IMU対応版

run_moving.py との違いは、動作モデルの回転 dtheta を
IMU(CoreS3のBMI270)の積分値から与えること。
レーダーは原理的に回転を測れない(v_d = -v*cosθ に回転の項がない)ため、
曲がり角で姿勢が失われるのを防ぐのがこのスクリプトの目的。

分業:
    回転 dtheta  <- IMUのジャイロZを積分
    並進 dx      <- レーダーのドップラー(または一定速度)

使い方:
    # logger.py で取ったデータ一式を使う
    python3 run_moving_imu.py run_20261012_143000.dat \
        --map map_corridor.npz --init 0.9 1.0 90 \
        --imu    run_20261012_143000_imu.csv \
        --frames run_20261012_143000_frames.csv \
        --gyro-scale 0.982 --gyro-sign +1

    # IMU無しの挙動と比べる(--no-imu で回転をゼロにする)
    python3 run_moving_imu.py ... --no-imu

gyro-scale / gyro-sign は analyze_d1.py が出す値をそのまま入れる。
"""
import argparse
import glob
import os
import sys

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
import imu as IMU

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
ap.add_argument("--imu", default=None, help="logger.py が出した *_imu.csv")
ap.add_argument("--frames", default=None, help="logger.py が出した *_frames.csv")
ap.add_argument("--gyro-scale", type=float, default=1.0,
                help="D-1c で求めたスケール補正係数")
ap.add_argument("--gyro-sign", type=float, default=1.0,
                help="ジャイロZの符号 (+1: 左回りが正)")
ap.add_argument("--gyro-bias", type=float, default=None,
                help="バイアス[deg/s]。省略時はログの静止区間から自動推定")
ap.add_argument("--bias-track", action="store_true",
                help="静止区間ごとにバイアスを測り直す(長時間走行向け)")
ap.add_argument("--s-th", type=float, default=None,
                help="動作モデルの回転ノイズ[rad/フレーム]。IMU使用時は小さくできる")
ap.add_argument("--no-imu", action="store_true",
                help="IMUを読み込んでも回転に使わない(比較用)")
ap.add_argument("--out", default="mcl_moving_imu.png")
a = ap.parse_args()


def _expand(pat, label, required=True):
    """PowerShell は * を展開しないので、ここで展開する。"""
    if not pat:
        return None
    hits = sorted(glob.glob(pat))
    if not hits:
        if os.path.exists(pat):
            return pat
        if required:
            sys.exit(f"{label}: '{pat}' に一致するファイルがありません\n"
                     f"  いまのフォルダ: {os.getcwd()}")
        return None
    if len(hits) > 1:
        print(f"  ! {label}: {len(hits)} 件見つかりました。最新の "
              f"{os.path.basename(hits[-1])} を使います")
    return hits[-1]


a.dat = _expand(a.dat, "dat")
a.imu = _expand(a.imu, "--imu", required=False)
a.frames = _expand(a.frames, "--frames", required=False)
a.gt = _expand(a.gt, "--gt", required=False)

lf = LikelihoodField(a.map)
df = parse_dat(a.dat)
scans = to_scan(df)
frames = sorted(scans)
print(f"読み込み: {len(frames)} フレーム ({len(frames)*FRAME_PERIOD:.1f} 秒), {len(df)} 点")

# ------------------------------------------------------------------ IMU
# dth[frame] = そのフレームまでの回転量 [rad]
dth = {}
use_imu = bool(a.imu) and not a.no_imu
if a.imu:
    d_imu = IMU.load_imu(a.imu, sign_z=a.gyro_sign)
    print()
    if a.frames:
        fr = IMU.load_frames(a.frames)
    else:
        # frames.csv が無い場合はフレーム番号×周期で代用する(精度は落ちる)
        import pandas as _pd
        fr = _pd.DataFrame({"frame": frames,
                            "t": d_imu.t.values[0] +
                                 (np.array(frames) - frames[0]) * FRAME_PERIOD})
        print("  ! --frames が無いので frame×66.7ms で代用します。")
        print("    曲がり角での精度が落ちるので、可能なら frames.csv を使ってください。")
    print(IMU.summary(d_imu, fr))

    if a.bias_track:
        b0 = IMU.estimate_bias(d_imu)["bias"]
        bias = IMU.bias_track(d_imu, b0)
        print(f"\nバイアス追従: {bias.min():+.4f} 〜 {bias.max():+.4f} deg/s")
    else:
        bias = a.gyro_bias if a.gyro_bias is not None \
            else IMU.estimate_bias(d_imu)["bias"]
        print(f"\nバイアス: {float(np.mean(bias)):+.5f} deg/s")

    tab = IMU.dtheta_per_frame(d_imu, fr, bias, scale=a.gyro_scale)
    dth = dict(zip(tab.frame.values, tab.dtheta.values))
    tot = np.degrees(tab.dtheta.sum())
    mx = np.degrees(tab.dtheta.abs().max())
    print(f"回転量   : 合計 {tot:+.1f}°, 1フレーム最大 {mx:.2f}°, "
          f"スケール {a.gyro_scale:.4f}")
    if a.no_imu:
        print("  (--no-imu が指定されたので、この値は動作モデルに使いません)")
    print()

# 回転が観測できるようになったぶん、回転ノイズを絞れる
s_th = a.s_th if a.s_th is not None else (0.012 if use_imu else 0.04)
s_cross = 0.02 if use_imu else 0.06
m = MCL(lf, n_particles=a.particles, motion_noise=(0.06, s_th, s_cross))
m.init_around(a.init[0], a.init[1], a.init[2], s_xy=0.2, s_deg=8)
print(f"動作モデル: s_xy=0.06, s_th={s_th:.3f} rad ({np.degrees(s_th):.1f}°), "
      f"s_cross={s_cross:.3f}" + ("  [IMUあり]" if use_imu else "  [IMUなし]"))

traj, vels, yaws = [], [], []
for i in range(0, len(frames) - a.step, a.step):
    fs = frames[i:i + a.step]
    sc = merge_scans(scans, fs)
    if len(sc["r"]) == 0:
        continue
    dt = a.step * FRAME_PERIOD

    # --- 回転 (IMU)
    dtheta = float(sum(dth.get(f, 0.0) for f in fs)) if use_imu else 0.0
    yaws.append((i * FRAME_PERIOD, dtheta))

    # --- 並進 (レーダー)
    if a.motion == "doppler":
        est = estimate_velocity(sc["r"], sc["th"], sc["doppler"], v_max=a.vmax)
        v = est["vx"] if est["ok"] else 0.0
        vels.append((i * FRAME_PERIOD, v, est["ok"], est["ratio"]))
    elif a.motion == "const":
        v = a.speed
    else:
        v = 0.0

    m.predict(dx=v * dt, dtheta=dtheta)

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
if use_imu and len(yaws):
    Y = np.array(yaws)
    # IMUは出発時の向きを起点にした絶対方位として描く
    imu_deg = a.init[2] + np.degrees(np.cumsum(Y[:, 1]))
    # MCLの向きは多峰分布から取るため跳ぶことがある。unwrapすると誤差が
    # 累積して実態より大きく見えるので、±180に畳んだ生の値を点で描く
    mcl_deg = (np.degrees(traj[:, 3]) - imu_deg[:len(traj)] + 180) % 360 - 180
    ax.plot(Y[:, 0], np.zeros(len(Y)), color="#2E5FA3", lw=1.6,
            label="IMU積分ヨー角（基準）")
    ax.plot(traj[:, 0], mcl_deg, ".", ms=3, color="#C77A15",
            label="MCL推定の向き（IMUとの差）")
    for lim in (13.2, -13.2):
        ax.axhline(lim, color="#2F7A5E", lw=1.0, ls="--")
    ax.text(traj[-1, 0], 13.2, "±13.2°(ビーム半値)", fontsize=7,
            color="#2F7A5E", ha="right", va="bottom")
    ax.set_ylabel("IMUを基準にした向きの差 [deg]")
    ax.set_ylim(-190, 190)
    ax.legend(fontsize=8)
    ax.set_title("向きの比較（IMU基準）")
elif len(vels):
    V = np.array([(t, v, float(ok)) for t, v, ok, _ in vels])
    ax.plot(V[:, 0], V[:, 1], ".", color="#1D9E75", label="ドップラー速度")
    bad = V[V[:, 2] < 0.5]
    if len(bad):
        ax.plot(bad[:, 0], bad[:, 1], "x", color="#D85A30", label="棄却")
    ax.axhline(a.vmax, color="gray", ls=":", label="v_max")
    ax.set_ylabel("速度 [m/s]"); ax.legend(fontsize=8)
    ax.set_title("自己速度推定")
else:
    ax.plot(traj[:, 0], traj[:, 5], color="#EF9F27")
    ax.set_ylabel("最大クラスタの重み")
    ax.set_title("収束度")
ax.set_xlabel("時刻 [s]"); ax.grid(alpha=0.3)

fig.suptitle(f"移動MCL: {a.dat.split('/')[-1]} "
             f"(並進={a.motion}, 回転={'IMU' if use_imu else 'なし'})", fontsize=12)
fig.tight_layout(); fig.savefig(a.out, dpi=115)
print(f"\nsaved: {a.out}")
