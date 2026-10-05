# -*- coding: utf-8 -*-
"""
analyze_d1.py -- D-1実験(ジャイロの素性を測る)の解析

A系列でレーダーの癖を測ったのと同じことを、IMUに対して行う。
MCLに組み込む前に必ずこれを通す。

  D-1a バイアス     : 静止60秒 × 3回     -> バイアス値と再現性
  D-1b ドリフト     : 静止5分             -> 放置したときの累積誤差 [deg/分]
  D-1c スケール     : 正確な90°旋回 × 10回 -> 倍率と1回あたりの誤差
  D-1d 周回         : ルート1周(合計360°)  -> 実走行での累積誤差

使い方:
  python analyze_d1.py --bias  d1a_1_imu.csv d1a_2_imu.csv d1a_3_imu.csv
  python analyze_d1.py --drift d1b_imu.csv
  python analyze_d1.py --scale d1c_imu.csv --expect 90 --turns 10
  python analyze_d1.py --loop  d1d_imu.csv --expect 360

出力:
  d1_report.txt  と 図(d1_bias.png / d1_drift.png / d1_scale.png)
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import imu as I

for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/meiryo.ttc"):
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=_p).get_name()
        except Exception:
            pass
        break
plt.rcParams["axes.unicode_minus"] = False

INK, SUB, BLUE, ORANGE, GREEN, GRID = (
    "#1F1F28", "#5C5A6E", "#2E5FA3", "#C77A15", "#2F7A5E", "#DDE3EC")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=SUB, labelsize=10)
    ax.grid(color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)


# ============================================================ D-1a バイアス
def run_bias(paths, outdir, sign_z=1.0):
    lines = ["=" * 64, "D-1a  バイアス測定", "=" * 64]
    res = []
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150)
    for i, p in enumerate(paths):
        d = I.load_imu(p, sign_z=sign_z)
        b = I.estimate_bias(d, use_stationary=False)   # 全区間が静止のはず
        res.append(b["bias"])
        lines.append(f"  {os.path.basename(p):<34s} "
                     f"bias = {b['bias']:+.5f} deg/s   "
                     f"ばらつき(1σ) = {b['std']:.5f}   n={b['n']:,}")
        t = d.t.values - d.t.values[0]
        ax.plot(t, d.wz.values, lw=0.6, alpha=0.75, label=f"#{i+1}")
        ax.axhline(b["bias"], color=ORANGE, lw=1.0, ls="--")

    res = np.array(res)
    lines.append("")
    lines.append(f"  3回の平均        : {res.mean():+.5f} deg/s")
    lines.append(f"  3回のばらつき    : {res.std():.5f} deg/s  "
                 f"(幅 {res.max()-res.min():.5f})")
    lines.append("")
    lines.append("  >> この平均値を imu.estimate_bias() の代わりに使ってもよいが、")
    lines.append("     温度で動くので走行ログ中の静止区間から測り直すのが確実。")
    lines.append(f"  >> 3回のばらつきが大きい(>0.05 deg/s)なら、毎回測り直すこと。")

    ax.set_xlabel("時間 [s]", color=SUB)
    ax.set_ylabel("ジャイロ Z [deg/s]", color=SUB)
    ax.set_title("D-1a 静止時のジャイロZ出力（破線が各回のバイアス）",
                 color=INK, fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "d1_bias.png"), facecolor="white")
    plt.close(fig)
    return lines, float(res.mean())


# ============================================================ D-1b ドリフト
def run_drift(path, outdir, bias=None, sign_z=1.0):
    lines = ["", "=" * 64, "D-1b  ドリフト（静止したまま積分し続ける）", "=" * 64]
    d = I.load_imu(path, sign_z=sign_z)
    if bias is None:
        bias = I.estimate_bias(d, use_stationary=False)["bias"]
    t, yaw = I.yaw_series(d, bias)
    t = t - t[0]
    deg = np.degrees(yaw)
    dur = t[-1]

    lines.append(f"  記録時間          : {dur:.1f} s ({dur/60:.1f} 分)")
    lines.append(f"  使ったバイアス    : {bias:+.5f} deg/s")
    lines.append(f"  終端での累積誤差  : {deg[-1]:+.2f} deg")
    lines.append(f"  平均ドリフト率    : {deg[-1]/(dur/60):+.2f} deg/分")

    # 区間ごとの誤差の伸び方(ランダムウォークかどうかの目安)
    lines.append("")
    lines.append("  経過時間ごとの誤差:")
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = int(len(t) * frac) - 1
        lines.append(f"    {t[k]/60:5.2f} 分 : {deg[k]:+7.2f} deg")
    lines.append("")
    one_turn_s = 2.0
    lines.append(f"  >> 曲がり角1回（約{one_turn_s:.0f}秒）あたりに換算すると "
                 f"{abs(deg[-1])/dur*one_turn_s:.3f} deg。")
    lines.append("     これが『回転中に溜まるドリフト』の目安で、")
    lines.append("     D-1c で出る旋回誤差より十分小さければ、ドリフトは問題にならない。")

    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150)
    ax.plot(t / 60, deg, color=BLUE, lw=1.6)
    ax.axhline(0, color=GRID, lw=1)
    for lim, c, lab in ((5, GREEN, "±5°"), (13.2, ORANGE, "±13.2°(ビーム半値)")):
        ax.axhline(lim, color=c, lw=1.0, ls="--")
        ax.axhline(-lim, color=c, lw=1.0, ls="--")
        ax.text(t[-1] / 60 * 0.99, lim, lab, fontsize=9, color=c,
                ha="right", va="bottom")
    ax.set_xlabel("経過時間 [分]", color=SUB)
    ax.set_ylabel("積分したヨー角 [deg]", color=SUB)
    ax.set_title("D-1b 静止させたまま積分したときの累積誤差",
                 color=INK, fontsize=11.5, fontweight="bold")
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "d1_drift.png"), facecolor="white")
    plt.close(fig)
    return lines


# ============================================================ D-1c スケール
def detect_turns(d, bias, w_th=5.0, min_dur=0.3):
    """|ヨーレート| が閾値を超えた区間を1回の旋回として切り出す。"""
    t = d.t.values
    w = d.wz.values - bias
    act = np.abs(w) > w_th
    idx = np.where(act)[0]
    if not len(idx):
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, splits + 1)
    turns = []
    for g in groups:
        if t[g[-1]] - t[g[0]] < min_dur:
            continue
        # 前後に少し余白を足して取りこぼしを防ぐ
        t0 = t[max(0, g[0] - 5)]
        t1 = t[min(len(t) - 1, g[-1] + 5)]
        turns.append((t0, t1))
    return turns


def run_scale(path, outdir, expect_deg=90.0, n_turns=None, bias=None,
              sign_z=1.0, w_th=5.0):
    lines = ["", "=" * 64,
             f"D-1c  スケールファクタ（{expect_deg:.0f}° 旋回の繰り返し）", "=" * 64]
    d = I.load_imu(path, sign_z=sign_z)
    if bias is None:
        bias = I.estimate_bias(d)["bias"]       # 旋回の合間の静止から推定
    lines.append(f"  使ったバイアス    : {bias:+.5f} deg/s")

    turns = detect_turns(d, bias, w_th=w_th)
    lines.append(f"  検出した旋回      : {len(turns)} 回"
                 + (f"（期待 {n_turns} 回）" if n_turns else ""))
    if not turns:
        lines.append("  ! 旋回が検出できません。--wth を下げてみてください。")
        return lines, 1.0, np.nan

    meas = []
    for (t0, t1) in turns:
        meas.append(np.degrees(I.integrate_yaw(d, bias, t0=t0, t1=t1)))
    meas = np.array(meas)
    sgn = np.sign(np.median(meas))
    amag = np.abs(meas)

    lines.append("")
    lines.append("   #   積分値[deg]   誤差[deg]   所要[s]")
    for i, ((t0, t1), m) in enumerate(zip(turns, meas)):
        lines.append(f"  {i+1:2d}   {m:+9.2f}   {abs(m)-expect_deg:+8.2f}   "
                     f"{t1-t0:6.2f}")

    scale = expect_deg / amag.mean()
    lines.append("")
    lines.append(f"  積分値の平均      : {amag.mean():.2f} deg "
                 f"(真値 {expect_deg:.0f} deg)")
    lines.append(f"  スケール補正係数  : {scale:.4f}")
    lines.append(f"  補正前の誤差      : 平均 {amag.mean()-expect_deg:+.2f} deg")
    lines.append(f"  補正後のばらつき  : ±{(amag*scale).std():.2f} deg (1σ)")
    lines.append(f"  最大のはずれ      : {np.abs(amag*scale-expect_deg).max():.2f} deg")
    lines.append("")
    lines.append(f"  >> MCL では  dtheta_per_frame(..., scale={scale:.4f})  と指定する。")
    sd = (amag * scale).std()
    lines.append(f"  >> 動作モデルのノイズ s_th は、1回の旋回で ±{sd:.2f}° なので")
    lines.append(f"     旋回にかかるフレーム数で割って設定する。")
    lines.append(f"     例: 旋回が約{np.median([t1-t0 for t0,t1 in turns]):.1f}秒 "
                 f"= 約{np.median([t1-t0 for t0,t1 in turns])*15:.0f}フレームなら")
    per = np.radians(sd) / max(np.sqrt(np.median([t1-t0 for t0, t1 in turns]) * 15), 1)
    per = max(per, 0.005)   # 下限。これ以上絞るとパーティクルが追従できなくなる
    lines.append(f"     s_th ≒ {per:.4f} rad ({np.degrees(per):.2f}°) / フレーム")
    lines.append(f"     （現在の既定値 0.03 rad = 1.7° から下げられる見込み）")

    if sgn < 0:
        lines.append("")
        lines.append("  ! 積分値が負です。左回りを正にしたいなら sign_z=-1.0 を指定。")

    # 図
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=150)
    ax = axes[0]
    t = d.t.values - d.t.values[0]
    ax.plot(t, d.wz.values - bias, color=BLUE, lw=0.8)
    for (t0, t1) in turns:
        ax.axvspan(t0 - d.t.values[0], t1 - d.t.values[0],
                   color=ORANGE, alpha=0.18)
    ax.axhline(w_th, color=ORANGE, ls="--", lw=1)
    ax.axhline(-w_th, color=ORANGE, ls="--", lw=1)
    ax.set_xlabel("時間 [s]", color=SUB)
    ax.set_ylabel("ヨーレート [deg/s]", color=SUB)
    ax.set_title("切り出した旋回区間", color=INK, fontsize=11, fontweight="bold")
    _style(ax)

    ax = axes[1]
    ax.bar(np.arange(1, len(amag) + 1), amag, color=BLUE, width=0.6)
    ax.axhline(expect_deg, color=GREEN, lw=1.8, label=f"真値 {expect_deg:.0f}°")
    ax.axhline(amag.mean(), color=ORANGE, lw=1.4, ls="--",
               label=f"平均 {amag.mean():.1f}°")
    ax.set_xlabel("旋回の回数", color=SUB)
    ax.set_ylabel("積分したヨー角 [deg]", color=SUB)
    ax.set_title("1回ごとの積分値", color=INK, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "d1_scale.png"), facecolor="white")
    plt.close(fig)
    return lines, float(scale), float((amag * scale).std())


# ============================================================ D-1d 周回
def run_loop(path, outdir, expect_deg=360.0, bias=None, scale=1.0, sign_z=1.0):
    lines = ["", "=" * 64, "D-1d  周回（ルート1周）", "=" * 64]
    d = I.load_imu(path, sign_z=sign_z)
    if bias is None:
        b = I.bias_track(d, I.estimate_bias(d)["bias"])
    else:
        b = bias
    t, yaw = I.yaw_series(d, b, scale=scale)
    deg = np.degrees(yaw)
    lines.append(f"  記録時間          : {t[-1]-t[0]:.1f} s")
    lines.append(f"  使ったスケール    : {scale:.4f}")
    lines.append(f"  終端の積分値      : {deg[-1]:+.2f} deg （真値 {expect_deg:+.0f}）")
    lines.append(f"  誤差              : {deg[-1]-expect_deg:+.2f} deg "
                 f"({abs(deg[-1]-expect_deg)/expect_deg*100:.1f}%)")
    lines.append("")
    if abs(deg[-1] - expect_deg) < 13.2:
        lines.append("  >> ビーム半値幅(13.2°)以内。MCLに入れて問題ない水準。")
    else:
        lines.append("  >> 13.2°を超えている。スケール補正かバイアス追従を見直すこと。")

    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150)
    ax.plot(t - t[0], deg, color=BLUE, lw=1.6)
    ax.axhline(expect_deg, color=GREEN, lw=1.4, ls="--",
               label=f"真値 {expect_deg:.0f}°")
    ax.set_xlabel("時間 [s]", color=SUB)
    ax.set_ylabel("累積ヨー角 [deg]", color=SUB)
    ax.set_title("D-1d ルート1周での累積ヨー角", color=INK,
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, labelcolor=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "d1_loop.png"), facecolor="white")
    plt.close(fig)
    return lines


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="D-1実験の解析")
    ap.add_argument("--bias", nargs="+", help="D-1a: 静止ログ(複数)")
    ap.add_argument("--drift", help="D-1b: 長時間静止ログ")
    ap.add_argument("--scale", help="D-1c: 繰り返し旋回ログ")
    ap.add_argument("--loop", help="D-1d: 周回ログ")
    ap.add_argument("--loop-expect", type=float, default=360.0,
                    help="D-1d で1周したときの合計回転角[deg]")
    ap.add_argument("--expect", type=float, default=90.0, help="1回の旋回角[deg]")
    ap.add_argument("--turns", type=int, default=None, help="期待する旋回回数")
    ap.add_argument("--wth", type=float, default=5.0, help="旋回検出の閾値[deg/s]")
    ap.add_argument("--sign", type=float, default=1.0, help="ジャイロZの符号 (+1/-1)")
    ap.add_argument("--outdir", default=".", help="出力先")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    report, bias, scale = [], None, 1.0

    if args.bias:
        ls, bias = run_bias(args.bias, args.outdir, args.sign)
        report += ls
    if args.drift:
        report += run_drift(args.drift, args.outdir, bias, args.sign)
    if args.scale:
        ls, scale, sd = run_scale(args.scale, args.outdir, args.expect,
                                  args.turns, bias, args.sign, args.wth)
        report += ls
    if args.loop:
        report += run_loop(args.loop, args.outdir, args.loop_expect,
                           bias, scale, args.sign)

    if not report:
        ap.print_help()
        return

    report += ["", "=" * 64, "まとめ", "=" * 64,
               f"  bias  = {bias if bias is not None else '(未測定)'}",
               f"  scale = {scale:.4f}",
               "",
               "  run_moving.py には次のように渡す:",
               f"    --imu <...>_imu.csv --frames <...>_frames.csv "
               f"--gyro-scale {scale:.4f} --gyro-sign {args.sign:+.0f}"]

    text = "\n".join(report)
    print(text)
    path = os.path.join(args.outdir, "d1_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n保存: {path}")


if __name__ == "__main__":
    main()
