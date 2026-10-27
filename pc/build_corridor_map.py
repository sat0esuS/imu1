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
  西へ延びる廊下 幅1.850 (Y 4.870〜6.720)
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
Y_WCOR_S = 4.870        # 西へ延びる廊下 南壁
Y_WCOR_N = 6.720        # 西へ延びる廊下 北壁（幅 = N - S = 1.850）
X_EV_W = -6.200         # EVブロック 西端 (東端は X_EVN)
Y_EV_S = 7.500          # EVブロック 南端
Y_EV_N = 9.400          # EVブロック 北端


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
    free |= box(-X_FAR, X_EVN, Y_WCOR_S, Y_WCOR_N)       # 西へ延びる廊下
    # EVブロックは廊下の中にある障害物なので、足した領域から引く。
    # ただし現在の矩形の組み方では、この範囲はもともと free になっていないので
    # 実際には何も変わらない（確認済み: 範囲内1786セルのうち free は0）。
    # 廊下の矩形を広げたときに効いてくるので、保険として残してある
    free &= ~box(X_EV_W, X_EVN, Y_EV_S, Y_EV_N)

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


def _use_cjk_font():
    """日本語が出るフォントを探す。見つからなくても落ちない。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc",
                 "C:/Windows/Fonts/msgothic.ttc"):
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = \
                font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_labeled_png(m, path):
    """どの壁がどの定数で、幅がいくつかを書き込んだ確認用の図。

    巻尺で測った値と図を突き合わせるためのもの。素の --png とは別物で、
    ここに描かれている線や寸法は npz には入っていない。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _use_cjk_font()

    SUB, RED, BLUE, ORANGE = "#5C5A6E", "#B5484E", "#2E5FA3", "#C77A15"
    # ラベルの下に白を敷く。敷かないと破線が文字を貫いて読めなくなる
    BOX = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.88)
    occ, free, res, org = m["occ"], m["free"], m["resolution"], m["origin"]
    H, W = occ.shape
    ext = [org[0], org[0] + W * res, org[1], org[1] + H * res]

    fig, ax = plt.subplots(figsize=(12.6, 10.4), dpi=160)
    ax.imshow(np.where(free, 1.0, np.nan), origin="lower", extent=ext,
              cmap="Blues", vmin=0, vmax=8, interpolation="nearest", zorder=1)
    ax.imshow(np.where(occ, 1.0, np.nan), origin="lower", extent=ext,
              cmap="gray", vmin=0, vmax=1.8, interpolation="nearest", zorder=2)

    # --- 縦線 = X_* の定数。上端にラベルを出す
    # 近い線どうしでラベルがぶつかるので、高さを交互にずらす
    for x, name, note, ytop in ((X_N1, "X_N1", "廊下1 北壁", 15.4),
                                (X_CADN, "X_CADN", "廊下1 南壁", 14.2),
                                (X_EPSS, "X_EPSS", "廊下A 北壁", 15.4),
                                (X_EVN, "X_EVN", "廊下A 南壁", 14.2)):
        ax.axvline(x, color=BLUE, lw=1.0, ls="--", alpha=0.8, zorder=5)
        ax.annotate(f"{name} = {x:+.3f}\n{note}", xy=(x, ytop), fontsize=9,
                    color=BLUE, ha="center", va="bottom", fontweight="bold",
                    zorder=7, bbox=BOX)

    # --- 横線 = Y_* の定数。右端にラベルを出す
    for y, name, note in ((Y_CADW, "Y_CADW", "CADサーバ室 西壁"),
                          (Y_EPSE, "Y_EPSE", "EPS 東壁"),
                          (Y_R21_E, "Y_R21_E", "廊下2-1 東壁"),
                          (Y_KYODO, "Y_KYODO", "廊下2-1 西壁")):
        ax.axhline(y, color=ORANGE, lw=1.0, ls="--", alpha=0.8, zorder=5)
        ax.annotate(f"{name} = {y:.3f}  ({note})", xy=(4.3, y), fontsize=9,
                    color=ORANGE, ha="left", va="center", fontweight="bold",
                    zorder=7, bbox=BOX)

    # --- 寸法線。(始点, 終点, ラベル, ラベル位置, 回転, 横位置揃え)
    for p0, p1, lab, tp, rot, ha in (
            ((X_CADN, 1.6), (X_N1, 1.6), f"廊下1 幅 {X_N1 - X_CADN:.3f} m",
             (0.0, 1.15), 0, "center"),
            ((X_EVN, 10.05), (X_EPSS, 10.05),
             f"廊下A 幅 {X_EPSS - X_EVN:.3f} m", (-2.77, 9.62), 0, "center"),
            ((2.6, Y_R21_E), (2.6, Y_KYODO),
             f"廊下2-1 幅 {Y_KYODO - Y_R21_E:.3f} m", (2.35, 11.9), 90, "center"),
            ((-7.4, Y_WCOR_S), (-7.4, Y_WCOR_N),
             f"西へ延びる廊下 幅 {Y_WCOR_N - Y_WCOR_S:.3f} m",
             (-7.65, (Y_WCOR_S + Y_WCOR_N) / 2), 90, "center")):
        ax.annotate("", xy=p0, xytext=p1,
                    arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6),
                    zorder=6)
        ax.annotate(lab, xy=tp, fontsize=9.5, color=RED, rotation=rot,
                    ha=ha, va="center", fontweight="bold", zorder=7, bbox=BOX)

    ax.annotate(f"EV ブロック（占有）\n"
                f"X {X_EV_W:.2f}〜{X_EVN:.2f} / Y {Y_EV_S:.2f}〜{Y_EV_N:.2f}",
                xy=((X_EV_W + X_EVN) / 2, (Y_EV_S + Y_EV_N) / 2),
                fontsize=9, color=SUB, ha="center",
                va="center", fontweight="bold", zorder=7, bbox=BOX)
    ax.annotate("← 廊下2-1 は X ±13 まで続く →", xy=(-2.0, 12.0), fontsize=9,
                color=SUB, ha="center", fontweight="bold", zorder=7, bbox=BOX)

    ax.set_xlim(-9.0, 9.0)
    ax.set_ylim(-1.2, 16.6)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]", color=SUB)
    ax.set_ylabel("Y [m]", color=SUB)
    ax.grid(color="#DDE3EC", lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="廊下の occupancy grid を実測寸法から作る")
    ap.add_argument("--npz", default=os.path.join(HERE, "map_corridor.npz"))
    # nargs="?" + const で、--png だけ書けば既定名、--png 別名.png で名前指定になる
    ap.add_argument("--png", nargs="?", const=os.path.join(HERE, "map_corridor.png"),
                    default=None, help="画像も出す（既定 map_corridor.png）")
    ap.add_argument("--res", type=float, default=RES, help="セルの大きさ [m]")
    ap.add_argument("--wall", type=float, default=0.10, help="壁の厚さ [m]")
    ap.add_argument("--scale", type=int, default=3, help="画像で1セルを何ピクセルにするか")
    ap.add_argument("--labels", nargs="?",
                    const=os.path.join(HERE, "map_corridor_labels.png"),
                    default=None,
                    help="どの壁がどの定数で幅がいくつかを書いた確認用の図")
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
    if a.labels:
        save_labeled_png(m, a.labels)
        print(f"ラベル付きの図を保存しました: {a.labels}")
