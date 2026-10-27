# -*- coding: utf-8 -*-
"""一様床と観測分散を振って、pyloc が実データで使える設定を探す"""
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


def run(seed, floor, mv):
    np.random.seed(seed)
    m = MCL(a.init[0], a.init[1], np.radians(a.init[2]))
    m.set_particle_num(a.particles)
    m.set_odom_noises(20.0, 0.5, 2.0, 0.3)
    m.set_landmarks(lm)
    m.set_measurement_variance(mv)
    m.set_measurement_resolution(0.1)
    m.set_measurement_model_coefficients(0.9, 0.1)
    m.set_measurement_floor(floor)
    m.set_resample_threshold(0.5)
    m.reset_particles(0.2, 0.2, np.radians(8.0))
    for dist, rot, r, th, snr in steps(scans, dth, frames):
        m.update_particles(dist, rot)
        m.calculate_weights(np.stack([r, th], axis=1))
        m.estimate_robot_pose()
        m.resample_particles()
    return (np.hypot(m.robot_pose_x - TRUE[0], m.robot_pose_y - TRUE[1]),
            m.robot_pose_x, m.robot_pose_y, np.degrees(m.robot_pose_yaw))


print(f"{'一様床':>10}{'分散':>8}  {'誤差 [m]':<22}{'中央値':>8}{'終点の例':>24}")
print("-" * 76)
for floor in (10e-6, 1e-2, 1e-1, 3e-1, 1.0):
    for mv in (0.01, 0.04):
        es, last = [], None
        for s in range(a.seeds):
            e, x, y, th = run(s, floor, mv); es.append(e); last = (x, y, th)
        tag = "原典" if floor == 10e-6 else ""
        print(f"{floor:>10.0e}{mv:>8.2f}  {' '.join(f'{e:6.2f}' for e in es):<22}"
              f"{np.median(es):>8.2f}  ({last[0]:+6.2f},{last[1]:+6.2f}){last[2]:>+7.1f}° {tag}")
