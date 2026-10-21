# -*- coding: utf-8 -*-
"""移植した pyloc の MCL に、このリポジトリで記録したデータを流す。

  python run_pyloc.py data/run3_*.dat --map ../map_corridor.npz \
      --imu data/run3_*_imu.csv --frames data/run3_*_frames.csv \
      --init 0 0 90 --true-end -3 11.754
"""
import argparse
import time

import numpy as np

from common import load_run, steps, landmarks_from_map
from pyloc_mcl import MCL

ap = argparse.ArgumentParser(description="pyloc の MCL を実データで動かす")
ap.add_argument("dat")
ap.add_argument("--map", default="../map_corridor.npz")
ap.add_argument("--imu", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--init", nargs=3, type=float, required=True,
                metavar=("X", "Y", "DEG"))
ap.add_argument("--true-end", nargs=2, type=float, default=None,
                metavar=("X", "Y"), help="正解の終点（誤差を出すため）")
ap.add_argument("--particles", type=int, default=4000)
ap.add_argument("--meas-var", type=float, default=0.01,
                help="等方の観測分散 [m^2]。pyloc は1つしか持てない")
ap.add_argument("--odom-noise", nargs=4, type=float,
                default=[20.0, 0.5, 2.0, 0.3],
                help="原典の odom_noise1..4。既定値は1ステップあたりの実効ノイズが "
                     "mcl.py の (0.12, 0.012) に近くなるよう合わせたもの")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

np.random.seed(a.seed)
scans, dth, frames = load_run(a.dat, a.imu, a.frames)
lm = landmarks_from_map(a.map)
print(f"占有セル {len(lm)} 個をランドマークとして使用")

m = MCL(a.init[0], a.init[1], np.radians(a.init[2]))
m.set_particle_num(a.particles)
m.set_odom_noises(*a.odom_noise)
m.set_landmarks(lm)
m.set_measurement_variance(a.meas_var)
m.set_measurement_resolution(0.1)
m.set_measurement_model_coefficients(0.9, 0.1)
m.set_resample_threshold(0.5)
m.reset_particles(0.2, 0.2, np.radians(8.0))

t0 = time.time()
for dist, rot, r, th, snr in steps(scans, dth, frames):
    m.update_particles(dist, rot)
    m.calculate_weights(np.stack([r, th], axis=1))
    m.estimate_robot_pose()
    m.resample_particles()

x, y, yaw = m.robot_pose_x, m.robot_pose_y, np.degrees(m.robot_pose_yaw)
print(f"終点 ({x:+.2f}, {y:+.2f})  向き {yaw:+.1f}°   {time.time()-t0:.1f} 秒")
if a.true_end:
    print(f"正解 ({a.true_end[0]:+.2f}, {a.true_end[1]:+.2f})  "
          f"誤差 {np.hypot(x-a.true_end[0], y-a.true_end[1]):.2f} m")
