# -*- coding: utf-8 -*-
"""
ドップラーによる自己速度推定 (radar ego-motion)

原理:
    センサが速度 v で前進し、静止物体が方位角 θ にあるとき、
    観測されるドップラーは  v_d = -v * cos(θ)
    複数の静止点から最小二乗で v を解く。
    横方向速度 vy も同時に解く場合: v_d = -(vx*cos θ + vy*sin θ)

注意 (C-1の実測より):
    - v_max を超えると折り返すので、|v| < v_max*0.8 の範囲でのみ信頼する
    - 歩行者など動体が混ざると外れ値になるため RANSAC で除外する
"""
import numpy as np


def estimate_velocity(scan_r, scan_th, doppler, v_max=0.61,
                      n_iter=50, inlier_th=0.06, min_points=4,
                      solve_lateral=False, rng=None):
    """
    戻り値: dict(vx, vy, n_inliers, ratio, ok)
        vx: 前進方向の速度 [m/s] (正 = 前進)
        ok: 推定が信頼できるか
    """
    rng = rng or np.random.default_rng(0)
    th = np.asarray(scan_th, float)
    d = np.asarray(doppler, float)
    n = len(d)
    if n < min_points:
        return dict(vx=0.0, vy=0.0, n_inliers=0, ratio=0.0, ok=False)

    # 設計行列: v_d = -(vx cosθ + vy sinθ)
    if solve_lateral:
        A = -np.stack([np.cos(th), np.sin(th)], axis=1)
    else:
        A = -np.cos(th)[:, None]

    best_in, best_x = None, None
    k = A.shape[1]
    for _ in range(n_iter):
        idx = rng.choice(n, size=max(k, 2), replace=False)
        try:
            x, *_ = np.linalg.lstsq(A[idx], d[idx], rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = np.abs(A @ x - d)
        inl = resid < inlier_th
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best_x = inl, x

    if best_in is None or best_in.sum() < min_points:
        return dict(vx=0.0, vy=0.0, n_inliers=0, ratio=0.0, ok=False)

    # インライアで再推定
    x, *_ = np.linalg.lstsq(A[best_in], d[best_in], rcond=None)
    vx = float(x[0])
    vy = float(x[1]) if solve_lateral else 0.0
    ratio = float(best_in.mean())
    speed = np.hypot(vx, vy)
    ok = (ratio > 0.5) and (speed < v_max * 0.8)
    return dict(vx=vx, vy=vy, n_inliers=int(best_in.sum()), ratio=ratio, ok=ok)


def velocity_series(scans, frames, v_max=0.61, window=3, **kw):
    """
    フレーム列に対して速度推定を行い、時系列を返す。
    window: 前後何フレームを束ねて推定するか(点数を稼ぐ)
    """
    out = []
    for i, f in enumerate(frames):
        lo = max(0, i - window // 2)
        hi = min(len(frames), lo + window)
        r = np.concatenate([scans[frames[j]]["r"] for j in range(lo, hi)])
        th = np.concatenate([scans[frames[j]]["th"] for j in range(lo, hi)])
        dp = np.concatenate([scans[frames[j]]["doppler"] for j in range(lo, hi)])
        est = estimate_velocity(r, th, dp, v_max=v_max, **kw)
        est["frame"] = f
        out.append(est)
    return out
