# -*- coding: utf-8 -*-
"""mcl.py の観測モデルから要素を1つずつ抜いて、どれが効いているかを測る。

  python ablate.py data/run3_*.dat --map ../map_corridor.npz \
      --imu data/run3_*_imu.csv --frames data/run3_*_frames.csv \
      --init 0 0 90 --true-end -3 11.754 --seeds 3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcl as M                                        # noqa: E402
from common import load_run, steps                     # noqa: E402

ap = argparse.ArgumentParser(description="観測モデルの要素別の寄与を測る")
ap.add_argument("dat")
ap.add_argument("--map", default="../map_corridor.npz")
ap.add_argument("--imu", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--init", nargs=3, type=float, required=True,
                metavar=("X", "Y", "DEG"))
ap.add_argument("--true-end", nargs=2, type=float, required=True,
                metavar=("X", "Y"))
ap.add_argument("--particles", type=int, default=4000)
ap.add_argument("--s-x", type=float, default=0.12, help="並進の動作ノイズ")
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()

ORIG = M.measurement_log_likelihood


def make_ll(const_sigma=None, flat_weights=False, tiny_floor=False):
    """元の尤度関数の σ・点の重み・一様床だけを差し替えた版を返す。"""
    def ll(particles, scan_r, scan_th, snr_db, lf, use_snr=True,
           sigma_min=0.06):
        N, Mn = particles.shape[0], scan_r.shape[0]
        if Mn == 0:
            return np.zeros(N)
        if const_sigma is None:
            st = scan_r * np.tan(lf.half_beam)
            sig = np.sqrt(np.maximum(lf.sigma_r, sigma_min)
                          * np.maximum(st, sigma_min))
        else:
            sig = np.full(Mn, const_sigma)
        wp = (np.ones(Mn) / Mn if flat_weights
              else M.point_weights(scan_r, snr_db, lf, use_snr))
        ct = np.cos(particles[:, 2])[:, None]
        st_ = np.sin(particles[:, 2])[:, None]
        px, py = scan_r * np.cos(scan_th), scan_r * np.sin(scan_th)
        gx = particles[:, 0:1] + ct * px - st_ * py
        gy = particles[:, 1:2] + st_ * px + ct * py
        d = lf.query(gx.ravel(), gy.ravel()).reshape(N, Mn)
        g = np.exp(-0.5 * (d / sig) ** 2) / (np.sqrt(2 * np.pi) * sig)
        floor = 1e-5 if tiny_floor else lf.z_rand / lf.max_dist
        p = lf.z_hit * g + floor
        return (wp * np.log(np.maximum(p, 1e-30))).sum(axis=1) * Mn
    return ll


lf = M.LikelihoodField(a.map)
scans, dth, frames = load_run(a.dat, a.imu, a.frames)
TRUE = np.array(a.true_end, float)


def run(seed, temperature=6.0, estimator="mode"):
    np.random.seed(seed)
    m = M.MCL(lf, n_particles=a.particles,
              motion_noise=(a.s_x, 0.012, 0.02),
              temperature=temperature, random_inject=0.0)
    m.init_around(a.init[0], a.init[1], a.init[2], s_xy=0.2, s_deg=8)
    for dist, rot, r, th, snr in steps(scans, dth, frames):
        m.predict(dx=dist, dtheta=rot)
        m.update(r, th, snr)
    if estimator == "mode":
        est, _ = m.estimate_mode()
        x, y = est[0], est[1]
    else:                                   # pyloc 流の重み付き平均
        w = m.weights
        x = float(np.sum(m.particles[:, 0] * w))
        y = float(np.sum(m.particles[:, 1] * w))
    return float(np.hypot(x - TRUE[0], y - TRUE[1]))


CASES = [
    ("いまの mcl.py のまま",            dict(), 6.0, "mode"),
    ("一様床を小さくする (1e-5)",       dict(tiny_floor=True), 6.0, "mode"),
    ("SNR重みを外す",                   dict(flat_weights=True), 6.0, "mode"),
    ("σ を定数 0.10 に固定",            dict(const_sigma=0.10), 6.0, "mode"),
    ("σ を定数 0.27 に固定",            dict(const_sigma=0.27), 6.0, "mode"),
    ("temperature を外す (1.0)",        dict(), 1.0, "mode"),
    ("推定を重み付き平均にする",        dict(), 6.0, "mean"),
]
print(f"{'条件':<32}{'誤差 [m]':<26}{'中央値':>8}")
print("-" * 68)
for lab, kw, temp, estm in CASES:
    M.measurement_log_likelihood = make_ll(**kw)
    es = [run(s, temp, estm) for s in range(a.seeds)]
    print(f"{lab:<32}{' '.join(f'{e:6.2f}' for e in es):<26}"
          f"{np.median(es):>8.2f}")
M.measurement_log_likelihood = ORIG
