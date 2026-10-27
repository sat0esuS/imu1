# -*- coding: utf-8 -*-
"""74号館3階 Z字廊下の占有グリッド地図を実測寸法から生成する。

SLAMで作った地図ではなく、巻尺で測った寸法から幾何的に組み立てている。
地図側に誤差が入らないので、自己位置推定で出た誤差を全部推定側のものだと
言い切れる。

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
  センサは台車先端から後方0.10 / 右0.15（地図には関係しない。解析側で扱う）

使い方:
    python build_corridor_map.py            → map_corridor.npz
    python build_corridor_map.py --png      → map_corridor.png も
"""
import argparse
import os

import numpy as np
from scipy.ndimage import binary_dilation

# このスクリプト自身が置いてあるディレクトリ。出力先をここに固定することで、
# どのディレクトリから実行しても .npz が pc/ に入る
HERE = os.path.dirname(os.path.abspath(__file__))

RES = 0.05              # 1セルの大きさ [m]。5cm 刻み

# --- 実測した壁の位置 [m]。Y_* は南北方向、X_* は東西方向の座標
Y_CADW = 4.900          # CADサーバ室 西壁
Y_EPSE = 9.354          # EPS 東壁 (P2 + 3.000)
Y_R21_E = 10.544        # 廊下2-1 東壁
Y_KYODO = 13.254        # 共同実験室東壁 = 廊下2-1 西壁
X_N1 = 0.900            # 廊下1 北壁
X_CADN = -0.900         # CADサーバ室 北壁 (= 廊下1 南壁)
X_EVN = -3.870          # EV北壁 = 廊下A 南壁
X_EPSS = -1.670         # EPS南壁 = 廊下A 北壁
X_FAR = 13.0            # 廊下2-1は実際にはもっと続くが、この範囲までを地図にする


def build(res=RES, wall=0.10, pad=1.5):
    """通行可能領域を矩形で組み立て、その縁を厚さ wall の壁にする。

    壁を直接描くのではなく「通行できる場所」を先に決め、その輪郭を
    壁とする。こうすると壁と床が食い違うことが原理的に起きない。
    """
    # --- ① 地図が覆う範囲を決める
    # 廊下の端から pad だけ余白を取る。余白が無いと端のセルで壁が切れる
    lo = np.array([-X_FAR - pad, -3.0 - pad])       # 左下隅の座標 (x, y)
    hi = np.array([X_FAR + pad, Y_KYODO + pad])     # 右上隅の座標 (x, y)
    W = int(np.ceil((hi[0] - lo[0]) / res))         # 横に何セル必要か
    H = int(np.ceil((hi[1] - lo[1]) / res))         # 縦に何セル必要か

    # 各セルの「中心」の座標を作る。+0.5 はセルの角ではなく中心を指すため。
    # XX, YY はどちらも (H, W) の配列で、XX[i,j] がそのセルの x 座標になる
    XX, YY = np.meshgrid(lo[0] + (np.arange(W) + 0.5) * res,
                         lo[1] + (np.arange(H) + 0.5) * res)

    def box(x0, x1, y0, y1):
        """矩形の中に入るセルだけ True にした (H, W) の真偽値配列を返す"""
        return (XX >= x0) & (XX <= x1) & (YY >= y0) & (YY <= y1)

    # --- ② 通行可能領域を矩形の足し算で組み立てる
    free = np.zeros((H, W), bool)                        # 最初は全部「通行不可」
    free |= box(X_CADN, X_N1, -3.0, Y_CADW)              # 廊下1 (P1→P2)
    free |= box(X_EVN, X_N1, Y_CADW, Y_EPSE)             # 廊下2-2 (P2→P4)
    free |= box(X_EVN, X_EPSS, Y_EPSE, Y_R21_E)          # 廊下A (P4→P6)
    free |= box(-X_FAR, X_FAR, Y_R21_E, Y_KYODO)         # 廊下2-1 (東西に長い)
    free |= box(-X_FAR, X_EVN, 4.87, 6.72)               # 西へ延びる廊下
    # EVブロックは廊下の中にある障害物なので、足した領域から引く
    free &= ~box(-6.20, X_EVN, 7.50, 9.40)

    # --- ③ 通行可能領域の縁を壁にする
    r = max(1, int(round(wall / res)))                   # 壁の厚さをセル数に直す (10cm → 2セル)
    # 半径 r セルの円盤を作る。これを「膨らませる形」として使う。
    # 正方形ではなく円盤にするのは、角の膨らみ方を揃えるため
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]                # -r..+r の格子
    occ = binary_dilation(free, (xx ** 2 + yy ** 2) <= r ** 2) & ~free
    # ↑ free を r セルぶん太らせてから、元の free を引く。
    #   残るのは free の外側に r セルだけはみ出した「縁」= 壁

    # --- ④ 地図の外へ廊下が続く方向には壁を置かない
    # ③ は地図の端でも縁を作ってしまうが、そこは実際には壁ではなく
    # 廊下の続きなので、壁を置くと存在しない壁にパーティクルが引き寄せられる
    edge = (XX < -X_FAR + 0.3) | (XX > X_FAR - 0.3) | (YY < -3.0 + 0.3)
    occ &= ~edge                                         # 端の壁を消す

    return dict(occ=occ, free=free, origin=lo, resolution=res, width=W, height=H)


def save_png(m, path, scale=3):
    """npz に入る配列をそのまま画像にする。1セル = scale ピクセル。

    軸も文字も枠も付けない。occ と free を色分けして並べるだけなので、
    画像のピクセルと配列の要素が 1 対 1 で対応する。
    """
    import matplotlib.image as mpimg

    occ, free = m["occ"], m["free"]
    H, W = occ.shape
    img = np.full((H, W, 3), 255, np.uint8)      # まず全セルを白で埋める
    img[free] = (214, 230, 245)                  # free が True のセル → 薄い水色
    img[occ] = (110, 110, 110)                   # occ が True のセル → グレー
    # 整数倍に引き伸ばす。np.repeat なので補間は入らず、1セルが scale 角の正方形になる
    img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    # 配列は行0が下（y が小さい側）だが、画像は行0が上に出るので上下を反転して書く
    mpimg.imsave(path, img[::-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="廊下の占有格子地図を実測寸法から作る")
    ap.add_argument("--npz", default=os.path.join(HERE, "map_corridor.npz"))
    # nargs="?" + const で、--png だけ書けば既定名、--png 別名.png で名前指定になる
    ap.add_argument("--png", nargs="?", const=os.path.join(HERE, "map_corridor.png"),
                    default=None, help="画像も出す（既定 map_corridor.png）")
    ap.add_argument("--res", type=float, default=RES, help="セルの大きさ [m]")
    ap.add_argument("--wall", type=float, default=0.10, help="壁の厚さ [m]")
    ap.add_argument("--scale", type=int, default=3, help="画像で1セルを何ピクセルにするか")
    a = ap.parse_args()

    m = build(res=a.res, wall=a.wall)
    # occ と free の両方を保存する。free は「見えていないだけで通れる場所」を
    # 区別するのに使う（パーティクルを撒ける範囲など）
    np.savez(a.npz, occ=m["occ"], free=m["free"],
             origin=m["origin"], resolution=m["resolution"])
    print(f"保存しました: {a.npz}")
    print(f"  {m['width']}x{m['height']} セル ({m['resolution'] * 100:.0f} cm), "
          f"通行可能 {m['free'].sum()} セル = {m['free'].sum() * a.res ** 2:.1f} m2, "
          f"壁 {m['occ'].sum()} セル")
    if a.png:
        save_png(m, a.png, a.scale)
        print(f"画像を保存しました: {a.png}")
