# -*- coding: utf-8 -*-
"""
74号館3階 Z字廊下の占有グリッド地図を実測寸法から生成する。

座標系: 原点 = P1の台車先端, +Y = 進行方向(P1→P2), +X = 進行方向の右手
実測値:
  P1→P2      6.354 m   (4900 + 1454)
  P2→P4      3.000 m
  P4→P6      5.400 m
  廊下1 幅    1.800 m   (P1は南壁から0.900)
  廊下A 幅    2.200 m   (P4は南壁から0.870)
  廊下2-1 幅  2.710 m   (P6は西壁から1.500)
  P4→CAD西壁 1.454 m
  EVブロック  X -6.20〜-3.87, Y 7.50〜9.40
  西へ延びる廊下 幅1.850 (Y 4.87〜6.72)

使い方:
    python build_corridor_map.py            → map_corridor.npz を生成
    python build_corridor_map.py --png      → 確認用の map_corridor.png も出す
"""
import argparse
import os

import numpy as np
from scipy.ndimage import binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))

RES = 0.05
Y_CADW   = 4.900        # CADサーバ室 西壁
Y_EPSE   = 9.354        # EPS 東壁 (P2 + 3.000)
Y_R21_E  = 10.544       # 廊下2-1 東壁
Y_KYODO  = 13.254       # 共同実験室東壁 = 廊下2-1 西壁
X_N1     = 0.900        # 廊下1 北壁
X_CADN   = -0.900       # CADサーバ室 北壁 (= 廊下1 南壁)
X_EVN    = -3.870       # EV北壁 = 廊下A 南壁
X_EPSS   = -1.670       # EPS南壁 = 廊下A 北壁
X_FAR    = 13.0         # 廊下2-1が続く範囲(片側)

POSES = {   # 台車先端の姿勢 (x, y, heading[deg])
    "P1": (0.0, 0.0, 90.0), "P2": (0.0, 6.354, 90.0), "P3": (0.0, 6.354, 180.0),
    "P4": (-3.0, 6.354, 180.0), "P5": (-3.0, 6.354, 90.0), "P6": (-3.0, 11.754, 90.0),
}
SENSOR_BACK, SENSOR_RIGHT = 0.10, 0.15   # 台車先端から後方 / 右へのオフセット


def sensor_pose(x, y, h_deg):
    """台車先端の姿勢から、センサ本体の実位置を求める"""
    a = np.radians(h_deg)
    fwd = np.array([np.cos(a), np.sin(a)])
    right = np.array([np.sin(a), -np.cos(a)])
    p = np.array([x, y]) - fwd * SENSOR_BACK + right * SENSOR_RIGHT
    return float(p[0]), float(p[1]), h_deg


def build(res=RES, wall=0.10, pad=1.5):
    lo = np.array([-X_FAR - pad, -3.0 - pad])
    hi = np.array([X_FAR + pad, Y_KYODO + pad])
    W = int(np.ceil((hi[0] - lo[0]) / res))
    H = int(np.ceil((hi[1] - lo[1]) / res))
    xs = lo[0] + (np.arange(W) + 0.5) * res
    ys = lo[1] + (np.arange(H) + 0.5) * res
    XX, YY = np.meshgrid(xs, ys)

    def box(x0, x1, y0, y1):
        return (XX >= x0) & (XX <= x1) & (YY >= y0) & (YY <= y1)

    free = np.zeros((H, W), bool)
    free |= box(X_CADN, X_N1, -3.0, Y_CADW)              # 廊下1 (P1→P2)
    free |= box(X_EVN, X_N1, Y_CADW, Y_EPSE)             # 廊下2-2 (P2→P4)
    free |= box(X_EVN, X_EPSS, Y_EPSE, Y_R21_E)          # 廊下A (P4→P6)
    free |= box(-X_FAR, X_FAR, Y_R21_E, Y_KYODO)         # 廊下2-1 (東西に長い)
    free |= box(-X_FAR, X_EVN, 4.87, 6.72)               # 西へ延びる廊下(幅1850)
    free &= ~box(-6.20, X_EVN, 7.50, 9.40)               # EVブロックは占有

    r = max(1, int(round(wall / res)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    occ = binary_dilation(free, (xx ** 2 + yy ** 2) <= r ** 2) & ~free
    # 地図の外周(廊下が続く方向)は壁にしない
    edge = (XX < -X_FAR + 0.3) | (XX > X_FAR - 0.3) | (YY < -3.0 + 0.3)
    occ &= ~edge
    return dict(occ=occ, free=free, origin=lo, resolution=res, width=W, height=H)


def _use_cjk_font():
    """日本語が出るフォントを探す。見つからなくても落ちない。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/msgothic.ttc",
                 "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"):
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                plt.rcParams["font.family"] = \
                    font_manager.FontProperties(fname=path).get_name()
                break
            except Exception:
                pass
    else:
        for name in ("Noto Sans CJK JP", "Yu Gothic", "MS Gothic", "Hiragino Sans"):
            if any(f.name == name for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = name
                break
    plt.rcParams["axes.unicode_minus"] = False


def draw(m, png_path):
    """占有格子を確認用の図にする。左=全体、右=走行範囲の拡大。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _use_cjk_font()

    INK, SUB, GREEN, RED, ORANGE, GRID = (
        "#1F1F28", "#5C5A6E", "#2F7A5E", "#B5484E", "#C77A15", "#DDE3EC")
    occ, free, res, org = m["occ"], m["free"], m["resolution"], m["origin"]
    H, W = occ.shape
    ext = [org[0], org[0] + W * res, org[1], org[1] + H * res]
    route = np.array([POSES[k][:2] for k in ("P1", "P2", "P4", "P6")], float)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 8.4), dpi=160,
                             gridspec_kw={"width_ratios": [1.45, 1]})
    for ax, lim, ttl in (
            (axes[0], (ext[0], ext[1], ext[2], ext[3]),
             "地図の全体（廊下2-1は東西に長い）"),
            (axes[1], (-7.6, 2.6, -1.6, 13.4), "走行範囲の拡大")):
        ax.imshow(np.where(free, 1.0, np.nan), origin="lower", extent=ext,
                  cmap="Blues", vmin=0, vmax=6, interpolation="nearest", zorder=1)
        ax.imshow(np.where(occ, 1.0, np.nan), origin="lower", extent=ext,
                  cmap="gray", vmin=0, vmax=1.6, interpolation="nearest", zorder=2)
        ax.plot(route[:, 0], route[:, 1], "-", color=GREEN, lw=3.2, alpha=0.85,
                zorder=4, label="走行ルート P1→P6")
        ax.plot(route[:, 0], route[:, 1], "o", color=GREEN, ms=9, zorder=5)
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
        ax.set_aspect("equal")
        ax.set_xlabel("X [m]", color=SUB, fontsize=10.5)
        ax.grid(color=GRID, lw=0.7, alpha=0.7); ax.set_axisbelow(True)
        ax.tick_params(colors=SUB, labelsize=9.5)
        ax.set_title(ttl, fontsize=12.5, color=INK, fontweight="bold", loc="left")
    axes[0].set_ylabel("Y [m]", color=SUB, fontsize=10.5)
    axes[0].legend(fontsize=10, loc="lower left", framealpha=0.95)

    ax = axes[1]
    for k, (x, y, h) in POSES.items():
        sx, sy, _ = sensor_pose(x, y, h)
        ax.plot([sx], [sy], "x", color=ORANGE, ms=8, mew=2, zorder=6)
        a = np.radians(h)
        ax.annotate("", xy=(x + 0.75 * np.cos(a), y + 0.75 * np.sin(a)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6), zorder=6)
    for k, (dx, dy) in {"P1": (0.42, -0.55), "P2": (0.42, 0.30),
                        "P3": (0.95, -0.62), "P4": (-0.55, 0.35),
                        "P5": (-0.95, -0.30), "P6": (0.40, 0.20)}.items():
        x, y, _ = POSES[k]
        ax.annotate(k, xy=(x + dx, y + dy), fontsize=11.5, color=GREEN,
                    fontweight="bold", ha="center", zorder=7)
    for (p0, p1, lab, tx, rot, ha) in (
            ((0.52, 0.0), (0.52, 6.354), "6.354 m", (0.72, 3.2), 90, "center"),
            ((0.0, 5.85), (-3.0, 5.85), "3.000 m", (-1.5, 5.60), 0, "center"),
            ((-3.52, 6.354), (-3.52, 11.754), "5.400 m", (-3.70, 9.05), 90, "right")):
        ax.annotate("", xy=p0, xytext=p1,
                    arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
        ax.annotate(lab, xy=tx, fontsize=10.5, color=RED, rotation=rot,
                    va="center", ha=ha, fontweight="bold")
    ax.annotate("EV ブロック\n（占有）", xy=(-5.0, 8.45), fontsize=10, color=SUB,
                ha="center", fontweight="bold", zorder=7)
    ax.annotate("西へ延びる廊下\n（幅 1.85 m）", xy=(-6.3, 5.8), fontsize=9.5,
                color=SUB, ha="center", fontweight="bold", zorder=7)
    ax.annotate("× = センサ実位置\n（台車先端から後方 %.2f / 右 %.2f）"
                % (SENSOR_BACK, SENSOR_RIGHT), xy=(-5.6, 12.5), fontsize=9.5,
                color=ORANGE, ha="center", fontweight="bold", zorder=7)

    fig.suptitle("74号館3階 Z字廊下の占有格子地図 — 実測寸法から生成"
                 f"（{res*100:.0f} cm 刻み、壁 {int(occ.sum())} セル）",
                 fontsize=14, color=INK, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="廊下の占有格子地図を実測寸法から作る")
    ap.add_argument("--npz", default=os.path.join(HERE, "map_corridor.npz"),
                    help="出力する .npz（既定はこのスクリプトと同じ場所）")
    ap.add_argument("--png", nargs="?", const=os.path.join(HERE, "map_corridor.png"),
                    default=None, help="確認用の画像も出す（既定 map_corridor.png）")
    ap.add_argument("--res", type=float, default=RES, help="セルの大きさ [m]")
    a = ap.parse_args()

    m = build(res=a.res)
    np.savez(a.npz, occ=m["occ"], free=m["free"],
             origin=m["origin"], resolution=m["resolution"])
    print(f"保存しました: {a.npz}")
    print(f"  {m['width']}x{m['height']} セル ({m['resolution']*100:.0f} cm), "
          f"通行可能 {m['free'].sum()} セル = {m['free'].sum()*a.res**2:.1f} m2, "
          f"壁 {m['occ'].sum()} セル")
    for k, v in POSES.items():
        sp = sensor_pose(*v)
        print(f"  {k}: 先端{v} → センサ({sp[0]:+.2f},{sp[1]:+.2f},{sp[2]:.0f}°)")

    if a.png:
        draw(m, a.png)
        print(f"画像を保存しました: {a.png}")
