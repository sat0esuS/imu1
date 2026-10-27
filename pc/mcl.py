'''
Copyright (C) 2019 Naoki Akai.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at https://mozilla.org/MPL/2.0/.
'''

# -*- coding: utf-8 -*-
"""ミリ波レーダー用の MCL（モンテカルロ自己位置推定）。

NaokiAkai/pyloc の mcl.py を土台にしている。
  原典: https://github.com/NaokiAkai/pyloc
  解説: https://qiita.com/NaokiAkai/items/44b2a160e47a8de44abf

5段の骨格（初期化 → 動作モデル → 観測モデル → 姿勢推定 → リサンプリング）、
関数名、ESS閾値つきのリサンプリング、重み付き平均での姿勢推定は原典のまま。
変えたのは次の3点で、どれも実測で必要性を確かめてある（数字は74号館の
本番ルート run3、終点の誤差の中央値、3〜10シード）。

  ① 地図: 点ランドマーク → 占有格子＋尤度場（LikelihoodField）
     計算している量は同じ（最も近い壁までの距離）だが、距離変換で前計算して
     おくと照合が配列参照1回で済む。原典の総当たりだと4000パーティクル ×
     3841ランドマークで1走行18秒かかる。

  ② 観測モデル: 一様床を 1e-7 → 0.1 に上げ、SNRによる点の重みを足した
     一様床を原典の値に戻すと 0.46 m → 5.18 m（11倍）。
     SNR重みを外すと 0.46 m → 0.81 m。
     ミリ波の点群は1フレームに2〜8個しかなく、多重反射やサイドローブによる
     ゴースト点が混ざる。床が低いと外れ値1個で正解のパーティクルが却下される。

  ③ 動作モデル: ノイズに下限を入れた
     原典は σ ∝ Δ² なので止まっている間はパーティクルが散らない。
     ドップラーから積算した距離は 8.87 m（真値 14.75 m）しかないため、
     足りない分は散って地図の尤度に選ばれることでしか埋まらない。
     下限なしだと終点の y が 2.9 m 足りないまま止まる。

逆に、効かなかったので入れていないもの:
  距離依存の観測σ（0.46 → 0.47 m）、尤度を鈍らせる temperature（0.46 →
  0.57 m）、最頻値での姿勢推定（平均のほうが良い）、横方向の動作ノイズ（逆効果）。
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

PI, PI2 = np.pi, 2.0 * np.pi


def mod_yaw(yaw):
    """角度を -π〜+π に畳み込む。原典の mod_yaw を配列対応にしたもの。"""
    return (np.asarray(yaw) + PI) % PI2 - PI


# ---------------------------------------------------------------- 地図
class LikelihoodField:
    """占有格子から「各セル → 最も近い壁までの距離」を前計算して持つ。

    原典は壁を点ランドマークの集合として持ち、観測点ごとに全ランドマークへ
    総当たりで最近傍を探していた。求まる量は同じだが、地図は走行中に変わら
    ないので先に全セルぶん計算しておける。
    """

    def __init__(self, map_npz, max_dist=1.5):
        d = np.load(map_npz)
        self.occ = d["occ"]              # True = 壁
        self.free = d["free"]            # True = 通行可能（部屋の中は False）
        self.origin = d["origin"]        # 地図の左下隅の実座標 (x, y)
        self.res = float(d["resolution"])  # 1セルの大きさ [m]
        self.H, self.W = self.occ.shape
        self.max_dist = max_dist

        # ~occ（壁でないセル）それぞれについて、最も近い壁まで何セルかを返す。
        # × res でメートルに直す。これが「尤度場」の本体
        self.dist = distance_transform_edt(~self.occ) * self.res
        # max_dist より遠いところは頭打ちにする。それ以上離れた点はガウスが
        # ほぼ0になり一様床が効くので、区別しても意味がない
        self.dist = np.minimum(self.dist, max_dist)

        # free なセルの (行, 列)。パーティクルを撒くときに使う
        self._free_rc = np.array(np.where(self.free))

    def world_to_idx(self, x, y):
        """実座標 → 配列の添字。配列は [行, 列] = [y, x] の順なので入れ替わる。"""
        c = ((x - self.origin[0]) / self.res).astype(int)
        r = ((y - self.origin[1]) / self.res).astype(int)
        ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)  # 地図の外を弾く
        return r, c, ok

    def query(self, x, y):
        """その点から最も近い壁までの距離 [m]。地図の外は max_dist 扱い。"""
        r, c, ok = self.world_to_idx(np.asarray(x), np.asarray(y))
        out = np.full(np.shape(x), self.max_dist, float)
        if ok.any():
            out[ok] = self.dist[r[ok], c[ok]]
        return out

    def is_free(self, x, y):
        """その点が通行可能セルの中か。"""
        r, c, ok = self.world_to_idx(np.asarray(x), np.asarray(y))
        out = np.zeros(np.shape(x), bool)
        if ok.any():
            out[ok] = self.free[r[ok], c[ok]]
        return out

    def sample_free(self, n):
        """通行可能セルの中から n 点を一様に選ぶ（大域推定の初期化用）。"""
        rr, cc = self._free_rc
        pick = np.random.randint(0, rr.size, n)
        # セルの中心に +-半セルの揺らぎを足して、格子に張り付かないようにする
        x = self.origin[0] + (cc[pick] + 0.5) * self.res \
            + (np.random.rand(n) - 0.5) * self.res
        y = self.origin[1] + (rr[pick] + 0.5) * self.res \
            + (np.random.rand(n) - 0.5) * self.res
        return x, y

    def as_landmarks(self):
        """壁セルの中心を点ランドマークとして返す（原典と同じ持ち方の比較用）。"""
        rr, cc = np.where(self.occ)
        return np.stack([self.origin[0] + (cc + 0.5) * self.res,
                         self.origin[1] + (rr + 0.5) * self.res], axis=1)


def p_detect(snr_db, s50=13.0, width=1.6):
    """SNR から検出確率を返すロジスティック曲線（A-2実験の実測に当てはめた）。

    SNR が s50 のとき 0.5 になり、それより高いほど 1 に近づく。
    """
    return 1.0 / (1.0 + np.exp(-(snr_db - s50) / width))


# ---------------------------------------------------------------- MCL 本体
class MCL:
    """原典 pyloc の MCL。地図と観測モデルをミリ波向けに差し替えてある。"""

    def __init__(self, lf, x=0.0, y=0.0, yaw=0.0):
        self.lf = lf                              # ← 原典には無い（地図）
        self.robot_pose_x = x
        self.robot_pose_y = y
        self.robot_pose_yaw = yaw

        # --- 動作モデル（原典と同じ4係数。σ は移動量の2乗に比例する）
        self.odom_noise1 = 20.0     # 前進方向の σ への、前進量の寄与
        self.odom_noise2 = 0.5      # 前進方向の σ への、回転量の寄与
        self.odom_noise3 = 2.0      # 回転方向の σ への、前進量の寄与
        self.odom_noise4 = 0.3      # 回転方向の σ への、回転量の寄与
        # 原典には無い下限。止まっている間も最低これだけ散らす（改変③）
        self.odom_floor_fw = 0.12   # 前進方向の σ の下限 [m]

        # --- 観測モデル
        self.measurement_variance = 0.1 * 0.1   # 等方の観測分散 [m^2]
        self.measurement_resolution = 0.1       # 原典と同じ、確率に掛ける係数
        self.z_hit, self.z_rand = 0.9, 0.1      # ガウスと一様床の混ぜ方
        # 原典は z_rand * 10e-6 の固定値だった。ミリ波では外れ値1個で正解が
        # 却下されてしまうので上げてある（改変②）
        self.measurement_floor = 1.0 / lf.max_dist
        self.use_snr = True                     # SNRで点に重みを付けるか（改変②）

        self.resample_threshold = 0.5           # ESS がこの割合を切ったら撒き直す
        self.particle_num = 4000
        self.particles = None                   # (N, 3) 各行が [x, y, yaw]
        self.weights = None                     # (N,)
        self.effective_sample_size = float(self.particle_num)

    # ---- 設定（原典と同名のものはそのまま）
    def set_particle_num(self, n):
        self.particle_num = n

    def set_odom_noises(self, n1, n2, n3, n4):
        self.odom_noise1, self.odom_noise2 = n1, n2
        self.odom_noise3, self.odom_noise4 = n3, n4

    def set_odom_noise_floor(self, fw):
        """動作ノイズの下限 [m]。0 にすると原典と同じ挙動になる。"""
        self.odom_floor_fw = fw

    def set_measurement_variance(self, v):
        self.measurement_variance = v

    def set_measurement_resolution(self, r):
        self.measurement_resolution = r

    def set_measurement_model_coefficients(self, z_hit, z_rand):
        if abs(z_hit + z_rand - 1.0) > 1e-12:
            raise ValueError("z_hit と z_rand の和は 1 でなければならない")
        self.z_hit, self.z_rand = z_hit, z_rand

    def set_measurement_floor(self, f):
        """一様床の高さ。小さいほど外れ値1個の罰が重くなる。原典は 1e-5。"""
        self.measurement_floor = f

    def set_resample_threshold(self, t):
        self.resample_threshold = t

    # ---- 原典 reset_particles
    def reset_particles(self, var_x, var_y, var_yaw):
        """推定姿勢のまわりにガウス分布でパーティクルを撒く（追従走行の開始時）。"""
        N = self.particle_num
        self.particles = np.stack([
            np.random.normal(self.robot_pose_x, var_x, N),
            np.random.normal(self.robot_pose_y, var_y, N),
            mod_yaw(np.random.normal(self.robot_pose_yaw, var_yaw, N))], axis=1)
        self.weights = np.full(N, 1.0 / N)       # 重みは最初は全部同じ
        self.effective_sample_size = float(N)

    def reset_particles_global(self, headings_deg=None):
        """地図の通行可能領域全体に撒く（初期位置が分からないとき）。原典には無い。"""
        N = self.particle_num
        x, y = self.lf.sample_free(N)
        if headings_deg is None:
            th = np.random.uniform(-PI, PI, N)               # 向きも分からない
        else:
            # 廊下沿いなど、向きの候補が分かっているときはそこから選ぶ
            th = np.radians(np.random.choice(headings_deg, N)) \
                + np.random.randn(N) * 0.05
        self.particles = np.stack([x, y, mod_yaw(th)], axis=1)
        self.weights = np.full(N, 1.0 / N)
        self.effective_sample_size = float(N)

    # ---- 原典 update_particles（動作モデル）
    def update_particles(self, delta_dist, delta_yaw):
        """移動量にノイズを乗せて全パーティクルを動かす。

        delta_dist : この更新での前進量 [m]（レーダーのドップラーから）
        delta_yaw  : この更新での回転量 [rad]（IMUのジャイロから）
        """
        N = self.particle_num
        d2, y2 = delta_dist ** 2, delta_yaw ** 2
        # 原典の式。移動量が大きいほどノイズも大きい
        s_d = self.odom_noise1 * d2 + self.odom_noise2 * y2
        s_y = self.odom_noise3 * d2 + self.odom_noise4 * y2
        # 下限を入れる（改変③）。これが無いと止まっている間 σ=0 で散らない
        s_d = max(s_d, self.odom_floor_fw, 1e-12)
        s_y = max(s_y, 1e-12)

        del_dist = np.random.normal(delta_dist, s_d, N)   # 各パーティクルの前進量
        del_yaw = np.random.normal(delta_yaw, s_y, N)     # 各パーティクルの回転量
        th = self.particles[:, 2]
        # それぞれ自分の向きの方向へ進む（原典と同じく前後方向のみ）
        self.particles[:, 0] += del_dist * np.cos(th)
        self.particles[:, 1] += del_dist * np.sin(th)
        self.particles[:, 2] = mod_yaw(th + del_yaw)

    # ---- 原典 calculate_weights（観測モデル）
    def calculate_weights(self, scan_r, scan_th, snr_db=None):
        """点群を各パーティクルの姿勢で地図に貼り、壁とどれだけ合うかで重み付け。

        scan_r  : (M,) センサから見た距離 [m]
        scan_th : (M,) センサから見た方位 [rad]（正面が0）
        snr_db  : (M,) 各点のSNR [dB]。None なら全点を等しく扱う
        """
        M = len(scan_r)
        if M == 0:
            return
        N = self.particle_num

        # --- 点ごとの重み。SNRが高い点ほど強く効かせる（改変②）
        if self.use_snr and snr_db is not None:
            wp = p_detect(np.asarray(snr_db))
        else:
            wp = np.ones(M)
        wp = wp / max(wp.sum(), 1e-12)         # 合計1に正規化

        # --- 点群をセンサ座標からパーティクルごとの地図座標へ回して平行移動する
        # px, py はセンサ座標での点の位置（正面が +x）
        px = scan_r * np.cos(scan_th)
        py = scan_r * np.sin(scan_th)
        ct = np.cos(self.particles[:, 2])[:, None]        # (N,1)
        st = np.sin(self.particles[:, 2])[:, None]
        gx = self.particles[:, 0:1] + ct * px - st * py   # (N,M) 地図座標の x
        gy = self.particles[:, 1:2] + st * px + ct * py   # (N,M) 地図座標の y

        # --- 各点が最も近い壁からどれだけ離れているか（改変①：配列参照1回）
        d = self.lf.query(gx.ravel(), gy.ravel()).reshape(N, M)

        var = self.measurement_variance
        norm_coef = 1.0 / np.sqrt(2.0 * PI * var)
        # 指数の分母が 2*(2*var) になっているのは原典のまま。通常のガウスより
        # √2 倍広い。一様床を 1e-7 に戻せば原典の挙動を再現できるよう残してある
        gauss = norm_coef * np.exp(-0.5 * d ** 2 / (2.0 * var))
        # 混合モデル: 壁の近くならガウス、遠ければ一様床。床があることで
        # ゴースト点1個が正解のパーティクルを却下できなくなる（改変②）
        prob = self.z_hit * gauss + self.z_rand * self.measurement_floor
        prob = np.minimum(prob * self.measurement_resolution, 1.0)

        # 点ごとの対数確率を重み付きで足す。確率の積を対数で扱うのは、
        # そのまま掛けると桁が落ちて0になるため
        total_log_prob = (wp * np.log(np.maximum(prob, 1e-300))).sum(axis=1)

        # 最大値を引いてから exp する（オーバーフロー・アンダーフロー対策）
        w = self.weights * np.exp(total_log_prob - total_log_prob.max())
        s = w.sum()
        # 全部0に潰れたら重みを平らに戻す（推定を捨てずに次の観測に賭ける）
        self.weights = np.full(N, 1.0 / N) if s < 1e-300 else w / s
        # 有効サンプル数。重みが1個に集中すると1に、平らなら N に近づく
        self.effective_sample_size = 1.0 / np.sum(self.weights ** 2)

    # ---- 原典 estimate_robot_pose（重み付き平均）
    def estimate_robot_pose(self):
        """パーティクルの重み付き平均を推定姿勢とする。"""
        w = self.weights
        self.robot_pose_x = float(np.sum(self.particles[:, 0] * w))
        self.robot_pose_y = float(np.sum(self.particles[:, 1] * w))
        # 角度はそのまま平均すると ±π の境目で壊れるので、前回の推定値からの
        # 差を平均してから足し戻す（原典と同じやり方）
        tmp = self.robot_pose_yaw
        dyaw = mod_yaw(tmp - self.particles[:, 2])
        self.robot_pose_yaw = float(mod_yaw(tmp - np.sum(dyaw * w)))

    # ---- 原典 resample_particles（ESS閾値つきルーレット）
    def resample_particles(self):
        """重みの偏りが強くなったときだけ、重みに比例して選び直す。"""
        N = self.particle_num
        # 毎回リサンプルすると多様性が失われるので、ESS が閾値を切ったときだけ
        if self.effective_sample_size > N * self.resample_threshold:
            return
        board = np.cumsum(self.weights)          # 累積重み = ルーレットの盤
        darts = np.random.random(N)              # N本の矢を一様に投げる
        idx = np.searchsorted(board, darts)      # どの区間に刺さったか
        np.clip(idx, 0, N - 1, out=idx)          # 丸め誤差で N に出るのを防ぐ
        self.particles = self.particles[idx].copy()
        self.weights = np.full(N, 1.0 / N)       # 選び直したら重みは平らに戻す

    # ---- 便利メソッド（原典には無い）
    def pose(self):
        """推定姿勢を (x, y, yaw[rad]) で返す。"""
        return self.robot_pose_x, self.robot_pose_y, self.robot_pose_yaw

    def spread(self):
        """パーティクルの散らばり具合 (位置[m], 向き[rad])。収束の目安。"""
        w = self.weights
        mx = np.sum(self.particles[:, 0] * w)
        my = np.sum(self.particles[:, 1] * w)
        sxy = np.sqrt(np.sum(w * ((self.particles[:, 0] - mx) ** 2
                                  + (self.particles[:, 1] - my) ** 2)))
        # 向きは単位ベクトルの平均の長さから散らばりを出す（角度の平均対策）
        c = np.sum(w * np.cos(self.particles[:, 2]))
        s = np.sum(w * np.sin(self.particles[:, 2]))
        sth = np.sqrt(max(-2.0 * np.log(max(np.hypot(c, s), 1e-12)), 0.0))
        return float(sxy), float(sth)
