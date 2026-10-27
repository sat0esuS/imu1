# -*- coding: utf-8 -*-
"""動作ノイズの下限を入れると y の不足が直るかを測る"""
import argparse
import numpy as np
from common import load_run, steps, landmarks_from_map
from pyloc_mcl import MCL

ap = argparse.ArgumentParser()
ap.add_argument("dat"); ap.add_argument("--map", default="../map_corridor.npz")
ap.add_argument("--imu", required=True); ap.add_argument("--frames", required=True)
ap.add_argument("--init", nargs=3, type=float, required=True)
ap.add_argument("--true-end", nargs=2, type=float, required=True)
ap.add_argument("--particles", type=int, default=4000)
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
scans, dth, frames = load_run(a.dat, a.imu, a.frames)
lm = landmarks_from_map(a.map)
TRUE = np.array(a.true_end, float)


def run(seed, fw, lat):
    np.random.seed(seed)
    m = MCL(a.init[0], a.init[1], np.radians(a.init[2]))
    m.set_particle_num(a.particles)
    m.set_odom_noises(20.0, 0.5, 2.0, 0.3)
    m.set_odom_noise_floor(fw, lat)
    m.set_landmarks(lm)
    m.set_measurement_variance(0.01)
    m.set_measurement_resolution(0.1)
    m.set_measurement_model_coefficients(0.9, 0.1)
    m.set_measurement_floor(0.1)
    m.set_resample_threshold(0.5)
    m.reset_particles(0.2, 0.2, np.radians(8.0))
    for dist, rot, r, th, snr in steps(scans, dth, frames):
        m.update_particles(dist, rot)
        m.calculate_weights(np.stack([r, th], axis=1))
        m.estimate_robot_pose()
        m.resample_particles()
    return (np.hypot(m.robot_pose_x - TRUE[0], m.robot_pose_y - TRUE[1]),
            m.robot_pose_x, m.robot_pose_y, np.degrees(m.robot_pose_yaw))


print(f"正解 ({TRUE[0]:+.2f}, {TRUE[1]:+.2f})   一様床 0.1 / 分散 0.01 で固定")
print(f"{'前進σ下限':>10}{'横σ':>8}  {'誤差 [m]':<22}{'中央値':>8}{'終点の例':>24}")
print("-" * 76)
for fw, lat in ((0.00, 0.00), (0.06, 0.00), (0.12, 0.00),
                (0.12, 0.06), (0.12, 0.12), (0.20, 0.12)):
    es, last = [], None
    for s in range(a.seeds):
        e, x, y, th = run(s, fw, lat); es.append(e); last = (x, y, th)
    tag = " 原典" if fw == 0 and lat == 0 else ""
    print(f"{fw:>10.2f}{lat:>8.2f}  {' '.join(f'{e:6.2f}' for e in es):<22}"
          f"{np.median(es):>8.2f}  ({last[0]:+6.2f},{last[1]:+6.2f}){last[2]:>+7.1f}°{tag}")
