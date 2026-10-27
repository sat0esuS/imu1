'''
Copyright (C) 2019 Naoki Akai.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at https://mozilla.org/MPL/2.0/.
'''

# -*- coding: utf-8 -*-
"""
NaokiAkai/pyloc の mcl.py を Python 3 に移植したもの。
  原典: https://github.com/NaokiAkai/pyloc
  解説: https://qiita.com/NaokiAkai/items/44b2a160e47a8de44abf

変えたのは3点だけで、アルゴリズムの数式は原典のまま:
  1. print 文 → print 関数（Python 2 → 3）
  2. 最近傍ランドマーク探索の線形走査 → KD木（答えは同一、速度のみ）
  3. パーティクル方向のループ → numpy のベクトル化（同上）

3841個のランドマーク × 4000パーティクルでも 1走行 18 秒で回る。
原典のまま純Pythonの三重ループだと数時間かかる。
"""
import numpy as np
from scipy.spatial import cKDTree

PI, PI2 = np.pi, 2.0 * np.pi


def mod_yaw(yaw):
    return (np.asarray(yaw) + PI) % PI2 - PI


class MCL:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.robot_pose_x, self.robot_pose_y, self.robot_pose_yaw = x, y, yaw
        # --- 原典の既定値
        self.odom_noise1, self.odom_noise2 = 1.0, 0.5
        self.odom_noise3, self.odom_noise4 = 0.5, 2.0
        self.measurement_variance = 0.1 * 0.1
        self.measurement_resolution = 0.1
        self.z_hit, self.z_rand = 0.9, 0.1
        self.resample_threshold = 0.5
        # 原典では 10e-6 のハードコード。ミリ波の疎な点群では外れ値1個で
        # 正解のパーティクルが却下されてしまうため、設定可能にした。
        # 既定値は原典のままなので、触らなければ挙動は変わらない。
        self.measurement_floor = 10e-6
        self.particle_num = 100
        self.particles = None          # (N,3) 姿勢
        self.weights = None            # (N,)
        self.landmarks = []
        self._tree = None

    # ---- 設定（原典と同名）
    def set_particle_num(self, n): self.particle_num = n
    def set_odom_noises(self, n1, n2, n3, n4):
        self.odom_noise1, self.odom_noise2 = n1, n2
        self.odom_noise3, self.odom_noise4 = n3, n4
    def set_measurement_variance(self, v): self.measurement_variance = v
    def set_measurement_resolution(self, r): self.measurement_resolution = r
    def set_measurement_floor(self, f):
        """一様床の高さ。原典は 10e-6 固定。

        尤度の下限。観測点が壁からどれだけ離れていても、この値より下には
        ならない。小さいほど外れ値1個の罰が重くなる（床 f の点1個が
        log(z_rand*f*resolution) を叩き込む）。ミリ波レーダーのように
        1フレームの点が2〜8個でゴーストが混ざる場合は大きめに取る。
        """
        self.measurement_floor = f
    def set_measurement_model_coefficients(self, z_hit, z_rand):
        if abs(z_hit + z_rand - 1.0) > 1e-12:
            raise ValueError("sum of z_hit and z_rand must be one.")
        self.z_hit, self.z_rand = z_hit, z_rand
    def set_resample_threshold(self, t): self.resample_threshold = t

    def add_landmark(self, x, y):
        self.landmarks.append([x, y]); self._tree = None

    def set_landmarks(self, pts):
        self.landmarks = [list(p) for p in np.asarray(pts)]; self._tree = None

    def _ensure_tree(self):
        if self._tree is None:
            self._tree = cKDTree(np.asarray(self.landmarks, dtype=float))
        return self._tree

    # ---- 原典 reset_particles
    def reset_particles(self, var_x, var_y, var_yaw):
        N = self.particle_num
        self.particles = np.stack([
            np.random.normal(self.robot_pose_x,   var_x,   N),
            np.random.normal(self.robot_pose_y,   var_y,   N),
            mod_yaw(np.random.normal(self.robot_pose_yaw, var_yaw, N))], axis=1)
        self.weights = np.full(N, 1.0 / N)

    # ---- 原典 update_particles（動作モデル）
    def update_particles(self, delta_dist, delta_yaw):
        N = self.particle_num
        d2, y2 = delta_dist ** 2, delta_yaw ** 2
        s_d = self.odom_noise1 * d2 + self.odom_noise2 * y2
        s_y = self.odom_noise3 * d2 + self.odom_noise4 * y2
        del_dist = np.random.normal(delta_dist, max(s_d, 1e-12), N)
        del_yaw  = np.random.normal(delta_yaw,  max(s_y, 1e-12), N)
        th = self.particles[:, 2]
        self.particles[:, 0] += del_dist * np.cos(th)
        self.particles[:, 1] += del_dist * np.sin(th)
        self.particles[:, 2] = mod_yaw(th + del_yaw)

    # ---- 原典 calculate_weights（観測モデル）
    def calculate_weights(self, measurements):
        m = np.asarray(measurements, dtype=float)       # (M,2) = [距離, 角度]
        if m.size == 0:
            return
        N, M = self.particle_num, len(m)
        th = self.particles[:, 2][:, None] + m[None, :, 1]    # (N,M)
        mx = m[None, :, 0] * np.cos(th) + self.particles[:, 0][:, None]
        my = m[None, :, 0] * np.sin(th) + self.particles[:, 1][:, None]
        min_dl, _ = self._ensure_tree().query(
            np.stack([mx.ravel(), my.ravel()], axis=1), k=1)
        min_dl = min_dl.reshape(N, M)

        var = self.measurement_variance
        norm_coef = 1.0 / np.sqrt(2.0 * PI * var)
        # 原典のまま: 指数の分母が 2*(2*var) になっている（通常のガウスの√2倍広い）
        prob = self.z_hit * norm_coef * np.exp(-0.5 * min_dl ** 2 / (2.0 * var)) \
            + self.z_rand * self.measurement_floor
        prob = np.minimum(prob * self.measurement_resolution, 1.0)
        total_log_prob = np.log(prob).sum(axis=1)                 # (N,)

        w = self.weights * np.exp(total_log_prob - total_log_prob.max())
        s = w.sum()
        self.weights = np.full(N, 1.0 / N) if s < 1e-300 else w / s
        self.effective_sample_size = 1.0 / np.sum(self.weights ** 2)
        self.total_weight = float(s)

    # ---- 原典 estimate_robot_pose（重み付き平均）
    def estimate_robot_pose(self):
        w = self.weights
        self.robot_pose_x = float(np.sum(self.particles[:, 0] * w))
        self.robot_pose_y = float(np.sum(self.particles[:, 1] * w))
        tmp = self.robot_pose_yaw
        dyaw = mod_yaw(tmp - self.particles[:, 2])
        self.robot_pose_yaw = float(mod_yaw(tmp - np.sum(dyaw * w)))

    # ---- 原典 resample_particles（ESS閾値つきルーレット）
    def resample_particles(self):
        N = self.particle_num
        if self.effective_sample_size > N * self.resample_threshold:
            return
        board = np.cumsum(self.weights)
        darts = np.random.random(N)
        idx = np.searchsorted(board, darts)          # 原典の線形探索と同じ結果
        np.clip(idx, 0, N - 1, out=idx)
        self.particles = self.particles[idx].copy()
        self.weights = np.full(N, 1.0 / N)
