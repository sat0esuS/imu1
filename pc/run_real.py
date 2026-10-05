# -*- coding: utf-8 -*-
"""
実データでのMCL検証

使い方:
    python3 run_real.py <録画.dat> --true 0.91 1.70 90
        --true は正解姿勢(x[m] y[m] heading[deg])。省略可(その場合は誤差を出さない)

段階的に3つを実行する:
    1) 地図の上に観測点群を重ねる (地図・前処理・座標系の検証)
    2) グリッド探索による尤度地形と最尤姿勢
    3) MCL(初期化→追跡)の収束
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
from preprocess import parse_dat, to_scan, merge_scans

ap = argparse.ArgumentParser()
ap.add_argument("dat")
ap.add_argument("--map", default="map.npz")
ap.add_argument("--true", nargs=3, type=float, default=None,
                metavar=("X", "Y", "DEG"))
ap.add_argument("--merge", type=int, default=15, help="統合するフレーム数")
ap.add_argument("--zband", nargs=2, type=float, default=[-0.5, 0.5])
ap.add_argument("--particles", type=int, default=3000)
ap.add_argument("--out", default="mcl_result.png")
a = ap.parse_args()

lf = LikelihoodField(a.map)
df = parse_dat(a.dat)
scans = to_scan(df, z_band=tuple(a.zband))
frames = sorted(scans)
print(f"読み込み: {len(df)} 点 / {len(frames)} フレーム")
print(f"高さフィルタ後の点数: {sum(len(s['r']) for s in scans.values())}")

merged = merge_scans(scans, frames[:a.merge])
print(f"統合スキャン({a.merge}フレーム): {len(merged['r'])} 点")

# --- 2) グリッド探索
m = MCL(lf, n_particles=a.particles)
res = m.global_search(merged["r"], merged["th"], merged["snr"],
                      step=0.10, headings_deg=np.arange(0, 360, 5))
p_best, ll_best = res[0]
print(f"\n最尤姿勢: ({p_best[0]:.2f}, {p_best[1]:.2f}, {np.degrees(p_best[2]):+.1f}°)  "
      f"logL={ll_best:.0f}  (2位との差 {ll_best - res[1][1]:.0f})")
if a.true:
    tx, ty, tdeg = a.true
    e = np.hypot(p_best[0] - tx, p_best[1] - ty)
    dth = abs((np.degrees(p_best[2]) - tdeg + 180) % 360 - 180)
    print(f"  → 正解との誤差: {e*100:.1f} cm, {dth:.1f}°")

# --- 3) MCL 追跡
m.init_around(p_best[0], p_best[1], np.degrees(p_best[2]))
hist = []
step = max(1, a.merge // 3)
for i in range(0, min(len(frames), 300) - step, step):
    sc = merge_scans(scans, frames[i:i + step])
    if len(sc["r"]) == 0:
        continue
    m.predict(0, 0, 0)
    m.update(sc["r"], sc["th"], sc["snr"])
    est, _ = m.estimate_mode()
    hist.append((est[0], est[1], est[2], m.spread()[0]))
hist = np.array(hist)
if len(hist):
    fin = hist[-1]
    print(f"MCL最終推定: ({fin[0]:.2f}, {fin[1]:.2f}, {np.degrees(fin[2]):+.1f}°), "
          f"広がり {fin[3]*100:.0f} cm")
    if a.true:
        e = np.hypot(fin[0] - a.true[0], fin[1] - a.true[1])
        print(f"  → 正解との誤差: {e*100:.1f} cm")

# --- 描画
ext = [lf.origin[0], lf.origin[0] + lf.W * lf.res,
       lf.origin[1], lf.origin[1] + lf.H * lf.res]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))


def draw_map(ax):
    ax.imshow(np.where(lf.occ, 0, np.nan), origin="lower", extent=ext,
              cmap="gray_r", vmin=0, vmax=1)
    ax.set_aspect("equal"); ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")


# ① 正解姿勢(または最尤姿勢)で点群を地図に重ねる
ax = axes[0]; draw_map(ax)
ref = np.array([a.true[0], a.true[1], np.radians(a.true[2])]) if a.true else p_best
gx = ref[0] + np.cos(ref[2]) * merged["r"] * np.cos(merged["th"]) - np.sin(ref[2]) * merged["r"] * np.sin(merged["th"])
gy = ref[1] + np.sin(ref[2]) * merged["r"] * np.cos(merged["th"]) + np.cos(ref[2]) * merged["r"] * np.sin(merged["th"])
sc = ax.scatter(gx, gy, c=merged["snr"], s=8, cmap="viridis", alpha=0.7)
ax.plot(ref[0], ref[1], "r*", ms=15)
plt.colorbar(sc, ax=ax, shrink=0.8, label="SNR [dB]")
ax.set_title("① 地図と観測点群の重ね合わせ")

# ② 尤度地形(最尤姿勢の向きで固定)
ax = axes[1]; draw_map(ax)
xs = np.arange(ext[0], ext[1], 0.10); ys = np.arange(ext[2], ext[3], 0.10)
XX, YY = np.meshgrid(xs, ys)
okm = lf.is_free(XX.ravel(), YY.ravel())
from mcl import measurement_log_likelihood
LL = np.full(XX.size, np.nan)
cand = np.stack([XX.ravel()[okm], YY.ravel()[okm], np.full(okm.sum(), p_best[2])], axis=1)
LL[okm] = measurement_log_likelihood(cand, merged["r"], merged["th"], merged["snr"], lf)
LL = LL.reshape(XX.shape)
v = LL[np.isfinite(LL)]
pc = ax.pcolormesh(XX, YY, LL, cmap="viridis", alpha=0.85,
                   vmin=np.percentile(v, 70), vmax=v.max())
ax.plot(p_best[0], p_best[1], "wo", ms=9, mec="k", label="最尤")
if a.true:
    ax.plot(a.true[0], a.true[1], "r*", ms=15, label="正解")
ax.legend(fontsize=8); ax.set_title("② 尤度地形(グリッド探索)")
plt.colorbar(pc, ax=ax, shrink=0.8, label="対数尤度")

# ③ パーティクルと推定軌跡
ax = axes[2]; draw_map(ax)
ax.scatter(m.particles[:, 0], m.particles[:, 1], s=2, alpha=0.25, color="#534AB7")
if len(hist):
    ax.plot(hist[:, 0], hist[:, 1], "-", color="#EF9F27", lw=1.5, label="推定の推移")
    ax.plot(hist[-1, 0], hist[-1, 1], "o", color="#EF9F27", ms=9, mec="k")
if a.true:
    ax.plot(a.true[0], a.true[1], "r*", ms=15, label="正解")
ax.legend(fontsize=8); ax.set_title("③ MCLパーティクル分布")

fig.suptitle(f"MCL検証: {a.dat.split('/')[-1]}", fontsize=12)
fig.tight_layout(); fig.savefig(a.out, dpi=115)
print(f"\nsaved: {a.out}")
