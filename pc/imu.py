# -*- coding: utf-8 -*-
"""
imu.py -- CoreS3(BMI270)のログを読み、レーダーフレームに合わせたヨー角を作る

設計方針:
  ・姿勢推定(クォータニオン等)はしない。使うのはジャイロZの積分だけ。
  ・加速度は「静止しているか」の判定にしか使わない。
    2回積分で位置を出そうとすると数秒でメートル級に発散するため。
  ・地磁気は使わない。屋内は鉄筋・EV・配電盤で方位が出ないため。

時刻の扱い:
  CoreS3の micros() は間隔が正確だがPCと時計が違う。
  ホスト受信時刻は時計は共通だがUSBのバッファで数msの揺らぎがある。
  そこで t_host ≈ a * t_dev + b を最小二乗で当てはめ、
  デバイス時刻を共通時計に写像して使う(両方の良いところを取る)。
"""
import numpy as np
import pandas as pd

# numpy 2.0 で trapz -> trapezoid に改名された。どちらでも動くようにしておく。
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------- 読み込み
def load_imu(csv_path, sign_z=+1.0):
    """
    imu.csv を読み、共通時計に写像した DataFrame を返す。

    sign_z: ジャイロZの符号。MCLのθは「前方=0, 左が正(反時計回り)」なので、
            左に回したとき gz が負になる取り付けなら -1 を渡す。
            check_sign() で確認できる。

    戻り値の列:
      t        共通時計での時刻 [s]   ← レーダーの frames.csv と同じ基準
      wz       ヨーレート [deg/s]     (符号適用済み、バイアス未除去)
      wx, wy   参考
      ax,ay,az 加速度 [G]
      amag     加速度の大きさ [G]     静止判定に使う
    """
    d = pd.read_csv(csv_path)
    t_dev = d["t_dev_us"].values.astype(np.float64) * 1e-6

    # micros() の32bit折り返し(約71.6分)を展開
    wrap = np.where(np.diff(t_dev) < -1000.0)[0]
    for i in wrap:
        t_dev[i + 1:] += 2 ** 32 * 1e-6

    t_host = d["t_host"].values.astype(np.float64)

    # t_host ≈ a*t_dev + b
    if len(t_dev) >= 10:
        a, b = np.polyfit(t_dev, t_host, 1)
    else:
        a, b = 1.0, (t_host.mean() - t_dev.mean() if len(t_dev) else 0.0)

    out = pd.DataFrame({
        "t": a * t_dev + b,
        "wx": d["gx"].values.astype(np.float64),
        "wy": d["gy"].values.astype(np.float64),
        "wz": d["gz"].values.astype(np.float64) * sign_z,
        "ax": d["ax"].values.astype(np.float64),
        "ay": d["ay"].values.astype(np.float64),
        "az": d["az"].values.astype(np.float64),
    })
    out["amag"] = np.sqrt(out.ax ** 2 + out.ay ** 2 + out.az ** 2)
    out.attrs["clock_a"] = float(a)
    out.attrs["clock_b"] = float(b)
    out.attrs["n"] = len(out)
    out.attrs["rate_hz"] = float(len(out) / max(out.t.values[-1] - out.t.values[0], 1e-9)) \
        if len(out) > 1 else 0.0
    return out


def load_frames(csv_path):
    """frames.csv を読む。戻り値: DataFrame(t, frame, n_bytes)"""
    d = pd.read_csv(csv_path)
    d = d.rename(columns={"t_host": "t"})
    # 同じフレーム番号が複数行に出た場合は最初の時刻を採る
    d = d.drop_duplicates(subset="frame", keep="first").reset_index(drop=True)
    return d


# ---------------------------------------------------------------- 静止判定
def stationary_mask(imu, w_th=1.0, a_tol=0.05, win=0.5):
    """
    静止している区間を True にする。
      w_th  : ジャイロ3軸の大きさがこの値[deg/s]未満
      a_tol : |加速度| が 1G からこれ以上ずれない
      win   : 判定の移動窓 [s]
    """
    wmag = np.sqrt(imu.wx ** 2 + imu.wy ** 2 + imu.wz ** 2).values
    quiet = (wmag < w_th) & (np.abs(imu.amag.values - 1.0) < a_tol)

    # 窓内がすべて quiet のときだけ静止とみなす(ノイズで点滅させない)
    t = imu.t.values
    n = len(t)
    out = np.zeros(n, bool)
    if n == 0:
        return out
    dt = np.median(np.diff(t)) if n > 1 else 0.01
    k = max(1, int(win / max(dt, 1e-6)))
    cs = np.concatenate([[0], np.cumsum(quiet.astype(int))])
    for i in range(n):
        lo = max(0, i - k // 2)
        hi = min(n, lo + k)
        out[i] = (cs[hi] - cs[lo]) == (hi - lo)
    return out


# ---------------------------------------------------------------- バイアス
def estimate_bias(imu, t0=None, t1=None, use_stationary=True):
    """
    ジャイロZのバイアス [deg/s] を推定する。

    t0, t1 を指定すればその区間、しなければ静止と判定された全区間を使う。
    戻り値: dict(bias, std, n, t_span)
    """
    t = imu.t.values
    if t0 is not None or t1 is not None:
        m = np.ones(len(t), bool)
        if t0 is not None:
            m &= t >= t0
        if t1 is not None:
            m &= t <= t1
    elif use_stationary:
        m = stationary_mask(imu)
    else:
        m = np.ones(len(t), bool)

    if m.sum() < 10:
        return dict(bias=0.0, std=np.nan, n=int(m.sum()), t_span=0.0, ok=False)

    w = imu.wz.values[m]
    return dict(bias=float(w.mean()), std=float(w.std()),
                n=int(m.sum()),
                t_span=float(t[m].max() - t[m].min()), ok=True)


def bias_track(imu, bias0, min_gap=1.0):
    """
    静止区間ごとにバイアスを測り直し、時刻に対する区分線形のバイアス列を返す。
    温度でバイアスが動くため、長時間の走行ではこれを使う。

    戻り値: 各サンプルに対応する bias の配列 (len(imu),)
    """
    t = imu.t.values
    st = stationary_mask(imu)
    # 静止区間を塊に分ける
    idx = np.where(st)[0]
    knots_t, knots_b = [], []
    if len(idx):
        splits = np.where(np.diff(idx) > 1)[0]
        groups = np.split(idx, splits + 1)
        for g in groups:
            if t[g[-1]] - t[g[0]] < min_gap:
                continue
            knots_t.append(t[g].mean())
            knots_b.append(imu.wz.values[g].mean())

    if not knots_t:
        return np.full(len(t), bias0)
    if len(knots_t) == 1:
        return np.full(len(t), knots_b[0])
    return np.interp(t, knots_t, knots_b)


# ---------------------------------------------------------------- 積分
def integrate_yaw(imu, bias, scale=1.0, t0=None, t1=None):
    """区間 [t0, t1) のヨー角変化 [rad] を台形積分で返す。"""
    t = imu.t.values
    m = np.ones(len(t), bool)
    if t0 is not None:
        m &= t >= t0
    if t1 is not None:
        m &= t < t1
    if m.sum() < 2:
        return 0.0
    w = (imu.wz.values[m] - bias) * scale        # [deg/s]
    return float(np.radians(_trapz(w, t[m])))


def yaw_series(imu, bias, scale=1.0, t_start=None):
    """
    全区間の累積ヨー角 [rad] を返す。戻り値: (t, yaw)
    軌跡をプロットして確かめるときに使う。
    """
    t = imu.t.values
    w = np.radians((imu.wz.values - np.asarray(bias)) * scale)
    yaw = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
    if t_start is not None:
        yaw -= np.interp(t_start, t, yaw)
    return t, yaw


def dtheta_per_frame(imu, frames, bias, scale=1.0):
    """
    レーダーの各フレーム間のヨー角変化 [rad] を返す。
    MCL の predict(dtheta=...) にそのまま渡せる。

    frames: load_frames() の DataFrame (列 t, frame)
    戻り値: DataFrame(frame, t, dt, dtheta)
            先頭フレームの dtheta は 0
    """
    ft = frames.t.values
    t = imu.t.values
    b = np.asarray(bias)
    if b.ndim == 0:
        b = np.full(len(t), float(b))
    w = np.radians((imu.wz.values - b) * scale)

    # 累積積分を作っておき、フレーム時刻で差分を取る(都度積分より速く正確)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
    y = np.interp(ft, t, cum)

    out = pd.DataFrame({
        "frame": frames.frame.values,
        "t": ft,
        "dt": np.concatenate([[0.0], np.diff(ft)]),
        "dtheta": np.concatenate([[0.0], np.diff(y)]),
    })
    # IMUの記録範囲外は 0 にしておく(外挿しない)
    bad = (ft < t[0]) | (ft > t[-1])
    out.loc[bad, "dtheta"] = 0.0
    return out


# ---------------------------------------------------------------- 符号確認
def check_sign(imu, t0, t1, expect_left=True):
    """
    [t0,t1] で左(反時計回り)に回した記録を渡すと、sign_z をどちらにすべきか答える。
    """
    d = integrate_yaw(imu, bias=0.0, t0=t0, t1=t1)
    deg = np.degrees(d)
    good = (deg > 0) == expect_left
    return dict(integrated_deg=float(deg),
                sign_z=+1.0 if good else -1.0,
                message=("そのまま sign_z=+1 でよい" if good
                         else "load_imu(..., sign_z=-1.0) を指定してください"))


# ---------------------------------------------------------------- 要約
def summary(imu, frames=None):
    s = [f"IMU サンプル数 : {len(imu):,}",
         f"記録時間       : {imu.t.values[-1] - imu.t.values[0]:.1f} s",
         f"平均レート     : {imu.attrs.get('rate_hz', 0):.1f} Hz",
         f"時計の係数 a   : {imu.attrs.get('clock_a', 1.0):.9f}  "
         f"(1からのずれが大きいならデバイス側の時計が狂っている)"]
    st = stationary_mask(imu)
    s.append(f"静止と判定     : {st.sum():,} / {len(imu):,} サンプル "
             f"({100*st.mean():.1f}%)")
    b = estimate_bias(imu)
    if b["ok"]:
        s.append(f"バイアス(Z)    : {b['bias']:+.4f} deg/s  "
                 f"(ばらつき {b['std']:.4f}, {b['t_span']:.1f}秒分)")
    if frames is not None and len(frames):
        s.append(f"レーダーframe  : {len(frames):,} 件  "
                 f"({frames.t.values[0]:.2f}s 〜 {frames.t.values[-1]:.2f}s)")
        dt = np.diff(frames.t.values)
        if len(dt):
            s.append(f"frame間隔      : 中央値 {np.median(dt)*1000:.1f} ms "
                     f"(設定値 66.7 ms)")
    return "\n".join(s)
