# -*- coding: utf-8 -*-
"""
尤度場モデルによるモンテカルロ自己位置推定 (MCL)

IWR6843AOP向けの適応:
  - 異方性の観測ノイズ: 距離方向 σ_r は小さく(bin量子化)、
    接線方向 σ_t は距離に比例(角度分解能26.4°に由来)
  - SNR依存の重み: A-2で実測した P(検出|SNR) ロジスティック曲線を使用
  - 非検出をfreeの証拠にしない (角度マスキング対策のため尤度場を採用)
  - 一様床 z_rand: ゴースト点が全パーティクルを一様に減点するだけにする
"""
import numpy as np
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------- 尤度場
class LikelihoodField:
    def __init__(self, map_npz, sigma_r=0.06, beam_deg=26.4,
                 z_hit=0.85, z_rand=0.15, max_dist=1.5):
        d = np.load(map_npz, allow_pickle=True)
        self.occ = d["occ"]
        self.free = d["free"]
        self.origin = d["origin"]
        self.res = float(d["resolution"])
        self.H, self.W = self.occ.shape
        # 各セルから最も近い占有セルまでの距離 [m]
        self.dist = distance_transform_edt(~self.occ) * self.res
        self.dist = np.minimum(self.dist, max_dist)
        self.sigma_r = sigma_r
        self.half_beam = np.radians(beam_deg / 2.0)
        self.z_hit = z_hit
        self.z_rand = z_rand
        self.max_dist = max_dist

    def world_to_idx(self, x, y):
        c = ((x - self.origin[0]) / self.res).astype(int)
        r = ((y - self.origin[1]) / self.res).astype(int)
        ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        return r, c, ok

    def query(self, x, y):
        """各点の最近傍壁距離 [m]。地図外は max_dist 扱い。"""
        r, c, ok = self.world_to_idx(np.asarray(x), np.asarray(y))
        out = np.full(np.shape(x), self.max_dist, float)
        if ok.any():
            out[ok] = self.dist[r[ok], c[ok]]
        return out

    def is_free(self, x, y):
        r, c, ok = self.world_to_idx(np.asarray(x), np.asarray(y))
        out = np.zeros(np.shape(x), bool)
        if ok.any():
            out[ok] = self.free[r[ok], c[ok]]
        return out


# ------------------------------------------------- 観測モデル (点群→尤度)
def p_detect(snr_db, s50=13.0, width=1.6):
    """A-2実測の検出確率カーブ。点の信頼度重みとして使う。"""
    return 1.0 / (1.0 + np.exp(-(snr_db - s50) / width))


def point_weights(scan_r, snr_db, lf, use_snr=True):
    """点ごとの重み(合計1に正規化)。SNRが高いほど強く効かせる。"""
    w = p_detect(snr_db) if use_snr else np.ones_like(scan_r)
    return w / max(w.sum(), 1e-12)


def measurement_log_likelihood(particles, scan_r, scan_th, snr_db, lf,
                               use_snr=True, sigma_min=0.06):
    """
    particles: (N,3) [x, y, theta(rad)]
    scan_r, scan_th: (M,) センサ座標系の距離[m]と方位[rad]
    戻り値: (N,) 対数尤度
    """
    N = particles.shape[0]
    M = scan_r.shape[0]
    if M == 0:
        return np.zeros(N)

    # --- 点ごとの異方性σ: 接線方向は距離に比例して広がる
    sigma_t = scan_r * np.tan(lf.half_beam)          # (M,)
    # 距離方向と接線方向の幾何平均を等方σとして近似(計算量削減)
    sigma = np.sqrt(np.maximum(lf.sigma_r, sigma_min) * np.maximum(sigma_t, sigma_min))

    wp = point_weights(scan_r, snr_db, lf, use_snr)  # (M,)

    # --- 各パーティクル姿勢で点群を地図座標へ変換
    cos_t = np.cos(particles[:, 2])[:, None]         # (N,1)
    sin_t = np.sin(particles[:, 2])[:, None]
    px = scan_r * np.cos(scan_th)                    # (M,) センサ前方=+x
    py = scan_r * np.sin(scan_th)
    gx = particles[:, 0:1] + cos_t * px - sin_t * py  # (N,M)
    gy = particles[:, 1:2] + sin_t * px + cos_t * py

    d = lf.query(gx.ravel(), gy.ravel()).reshape(N, M)

    # --- 混合モデル: ガウス(壁に近い) + 一様床(外れ値/ゴースト)
    gauss = np.exp(-0.5 * (d / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    unif = 1.0 / lf.max_dist
    p = lf.z_hit * gauss + lf.z_rand * unif
    return (wp * np.log(np.maximum(p, 1e-30))).sum(axis=1) * M


# ------------------------------------------------------------- 動作モデル
def motion_sample(particles, dx, dy, dtheta, noise):
    """
    ロボット座標系での移動量(dx前進, dy横, dtheta回転)を適用。
    noise = (s_xy, s_theta, s_xy_per_rot)
    """
    N = particles.shape[0]
    s_xy, s_th, s_cross = noise
    c, s = np.cos(particles[:, 2]), np.sin(particles[:, 2])
    ndx = dx + np.random.randn(N) * (s_xy + s_cross * abs(dtheta))
    ndy = dy + np.random.randn(N) * s_xy
    nth = dtheta + np.random.randn(N) * s_th
    out = particles.copy()
    out[:, 0] += c * ndx - s * ndy
    out[:, 1] += s * ndx + c * ndy
    out[:, 2] = (out[:, 2] + nth + np.pi) % (2 * np.pi) - np.pi
    return out


# --------------------------------------------------------- リサンプリング
def low_variance_resample(particles, weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.rand()) / N
    idx = np.searchsorted(np.cumsum(weights), positions)
    idx = np.clip(idx, 0, N - 1)
    return particles[idx]


def effective_sample_size(weights):
    return 1.0 / np.sum(weights ** 2)


# ------------------------------------------------------------------ MCL
class MCL:
    def __init__(self, lf, n_particles=3000, motion_noise=(0.05, 0.03, 0.05),
                 use_snr=True, resample_thresh=0.5, temperature=6.0,
                 random_inject=0.03):
        """
        temperature: 尤度の鋭さを鈍らせる係数。大きいほど慎重(1フレームで
            決めつけない)。疎な点群では対称解への早期収束を防ぐため必須。
        random_inject: 毎更新で入れ替える一様パーティクルの割合。
            グローバル推定時のパーティクル枯渇対策。
        """
        self.lf = lf
        self.N = n_particles
        self.motion_noise = motion_noise
        self.use_snr = use_snr
        self.resample_thresh = resample_thresh
        self.temperature = temperature
        self.random_inject = random_inject
        self.particles = None
        self.weights = None
        self._free_rc = np.array(np.where(lf.free))

    def init_global(self, headings=None):
        """地図の通行可能領域全体に一様にばら撒く(グローバル自己位置推定)。"""
        lf = self.lf
        rr, cc = np.where(lf.free)
        pick = np.random.randint(0, len(rr), self.N)
        x = lf.origin[0] + (cc[pick] + 0.5) * lf.res + (np.random.rand(self.N) - 0.5) * lf.res
        y = lf.origin[1] + (rr[pick] + 0.5) * lf.res + (np.random.rand(self.N) - 0.5) * lf.res
        if headings is None:
            th = np.random.uniform(-np.pi, np.pi, self.N)
        else:
            th = np.radians(np.random.choice(headings, self.N)) + np.random.randn(self.N) * 0.05
        self.particles = np.stack([x, y, th], axis=1)
        self.weights = np.full(self.N, 1.0 / self.N)

    def global_search(self, scan_r, scan_th, snr_db, step=0.10,
                      headings_deg=None, top_k=5):
        """
        地図全体をグリッド探索して尤度最大の姿勢を求める(初期化用)。
        パーティクルの初期収束の運に左右されず、多フレーム統合スキャンと
        組み合わせれば対称解の判別も確実になる。
        戻り値: [(pose, logL), ...] 上位top_k
        """
        lf = self.lf
        xs = np.arange(lf.origin[0], lf.origin[0] + lf.W * lf.res, step)
        ys = np.arange(lf.origin[1], lf.origin[1] + lf.H * lf.res, step)
        XX, YY = np.meshgrid(xs, ys)
        ok = lf.is_free(XX.ravel(), YY.ravel())
        gx, gy = XX.ravel()[ok], YY.ravel()[ok]
        hs = np.radians(headings_deg if headings_deg is not None
                        else np.arange(0, 360, 5))
        best = []
        for h in hs:
            cand = np.stack([gx, gy, np.full(gx.size, h)], axis=1)
            ll = measurement_log_likelihood(cand, scan_r, scan_th, snr_db,
                                            lf, self.use_snr)
            i = int(np.argmax(ll))
            best.append((cand[i].copy(), float(ll[i])))
        best.sort(key=lambda t: -t[1])
        return best[:top_k]

    def init_from_search(self, scan_r, scan_th, snr_db, s_xy=0.25, s_deg=8.0,
                         **kw):
        res = self.global_search(scan_r, scan_th, snr_db, **kw)
        p = res[0][0]
        self.init_around(p[0], p[1], np.degrees(p[2]), s_xy, s_deg)
        return res

    def init_around(self, x, y, heading_deg, s_xy=0.3, s_deg=10.0):
        """初期位置がおおよそ分かっている場合(トラッキング)。"""
        self.particles = np.stack([
            x + np.random.randn(self.N) * s_xy,
            y + np.random.randn(self.N) * s_xy,
            np.radians(heading_deg) + np.random.randn(self.N) * np.radians(s_deg)], axis=1)
        self.weights = np.full(self.N, 1.0 / self.N)

    def predict(self, dx=0.0, dy=0.0, dtheta=0.0):
        self.particles = motion_sample(self.particles, dx, dy, dtheta, self.motion_noise)

    def _sample_free(self, n, headings=None):
        rr, cc = self._free_rc
        pick = np.random.randint(0, rr.size, n)
        lf = self.lf
        x = lf.origin[0] + (cc[pick] + 0.5) * lf.res + (np.random.rand(n) - 0.5) * lf.res
        y = lf.origin[1] + (rr[pick] + 0.5) * lf.res + (np.random.rand(n) - 0.5) * lf.res
        if headings is None:
            th = np.random.uniform(-np.pi, np.pi, n)
        else:
            th = np.radians(np.random.choice(headings, n)) + np.random.randn(n) * 0.05
        return np.stack([x, y, th], axis=1)

    def update(self, scan_r, scan_th, snr_db, headings=None):
        ll = measurement_log_likelihood(self.particles, scan_r, scan_th, snr_db,
                                        self.lf, self.use_snr)
        ll = ll / self.temperature          # 尤度を鈍らせる(過信防止)
        ll -= ll.max()
        w = self.weights * np.exp(ll)
        s = w.sum()
        w = np.full(self.N, 1.0 / self.N) if s < 1e-300 else w / s
        self.weights = w
        if effective_sample_size(w) < self.resample_thresh * self.N:
            self.particles = low_variance_resample(self.particles, w)
            n_inj = int(self.N * self.random_inject)
            if n_inj > 0:
                idx = np.random.choice(self.N, n_inj, replace=False)
                self.particles[idx] = self._sample_free(n_inj, headings)
            self.weights = np.full(self.N, 1.0 / self.N)

    def estimate(self):
        """重み付き平均姿勢(角度は単位ベクトル平均)。"""
        w = self.weights
        x = np.sum(w * self.particles[:, 0])
        y = np.sum(w * self.particles[:, 1])
        c = np.sum(w * np.cos(self.particles[:, 2]))
        s = np.sum(w * np.sin(self.particles[:, 2]))
        return np.array([x, y, np.arctan2(s, c)])

    def estimate_mode(self, bandwidth=0.35):
        """
        多峰分布に対応した推定。最も重みが集まっているクラスタの
        重み付き平均を返す。分布が2山に割れている状況では
        estimate()(全体平均)より正しい。
        戻り値: (pose, mode_weight) mode_weightはそのクラスタの重み和
        """
        P, w = self.particles, self.weights
        # 重み上位の点を種に、近傍の重みを積算して最良の種を選ぶ
        top = np.argsort(w)[-min(60, len(w)):]
        best, best_mass = None, -1.0
        for i in top:
            d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
            dth = np.abs((P[:, 2] - P[i, 2] + np.pi) % (2 * np.pi) - np.pi)
            near = (d < bandwidth) & (dth < np.radians(35))
            mass = w[near].sum()
            if mass > best_mass:
                best_mass, best = mass, near
        ww = w[best] / max(w[best].sum(), 1e-300)
        x = np.sum(ww * P[best, 0]); y = np.sum(ww * P[best, 1])
        c = np.sum(ww * np.cos(P[best, 2])); s = np.sum(ww * np.sin(P[best, 2]))
        return np.array([x, y, np.arctan2(s, c)]), float(best_mass)

    def spread(self):
        """パーティクルの広がり(収束判定用) [m], [deg]"""
        est = self.estimate()
        d = np.hypot(self.particles[:, 0] - est[0], self.particles[:, 1] - est[1])
        dth = np.degrees(np.abs((self.particles[:, 2] - est[2] + np.pi) % (2 * np.pi) - np.pi))
        return float(np.sqrt(np.sum(self.weights * d ** 2))), float(np.sqrt(np.sum(self.weights * dth ** 2)))
