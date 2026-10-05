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
    python3 build_corridor_map.py      → map_corridor.npz を生成
"""
import numpy as np
from scipy.ndimage import binary_dilation

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


if __name__ == "__main__":
    m = build()
    np.savez("map_corridor.npz", occ=m["occ"], free=m["free"],
             origin=m["origin"], resolution=m["resolution"])
    print(f"saved map_corridor.npz  {m['width']}x{m['height']} cells "
          f"({m['resolution']*100:.0f} cm), free={m['free'].sum()}, occ={m['occ'].sum()}")
    for k, v in POSES.items():
        sp = sensor_pose(*v)
        print(f"  {k}: 先端{v} → センサ({sp[0]:+.2f},{sp[1]:+.2f},{sp[2]:.0f}°)")
