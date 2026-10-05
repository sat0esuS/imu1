# -*- coding: utf-8 -*-
"""
点群からオキュパンシグリッド(占有格子地図)を構築する。

A〜C系列の実測に基づく逆センサーモデル:
  - 点のSNRからA-2実測のロジスティック曲線で確信度を決める
  - 角度不確かさはB-1実測のビーム幅26.4度 (視野端では26.4/cosθ)
  - 距離不確かさはbin量子化に由来して小さい
  - free更新は「点までの経路」のみ。点がない方向は更新しない
    (角度マスキングによる欠測をfreeの証拠にしないため)

使い方:
    from occupancy import build_local_map
    g = build_local_map(scans, sorted(scans)[:3000])
    P = g.prob()          # 0〜1の占有確率マップ
"""
import numpy as np


class OccupancyGrid:
    def __init__(self, size_m=26.0, res=0.05, l_min=-4.0, l_max=6.0):
        self.res = res
        self.n = int(size_m / res)
        self.origin = np.array([-size_m / 2, -size_m / 2])
        self.L = np.zeros((self.n, self.n))     # log-odds
        self.l_min, self.l_max = l_min, l_max

    def w2i(self, x, y):
        c = ((x - self.origin[0]) / self.res).astype(int)
        r = ((y - self.origin[1]) / self.res).astype(int)
        ok = (c >= 0) & (c < self.n) & (r >= 0) & (r < self.n)
        return r, c, ok

    def prob(self):
        """log-odds を 0〜1 の確率に変換"""
        return 1.0 / (1.0 + np.exp(-self.L))

    def integrate(self, r, th, snr, pose=(0.0, 0.0, 0.0),
                  beam_deg=26.4, sigma_r=0.05, free_gain=0.25,
                  occ_gain=1.0, max_range=12.0):
        """1フレーム分の点群をグリッドに統合する。

        r, th, snr : センサ座標系の距離[m]、方位[rad]、SNR[dB]
        pose       : センサの姿勢 (x, y, theta[rad])
        """
        px, py, pth = pose
        keep = (r > 0.2) & (r < max_range) & np.isfinite(snr)
        r, th, snr = r[keep], th[keep], snr[keep]
        if len(r) == 0:
            return

        # 点ごとの確信度 (A-2実測: 50%点13dB, 幅1.6dB)
        conf = 1.0 / (1.0 + np.exp(-(snr - 13.0) / 1.6))

        for ri, ti, ci in zip(r, th, conf):
            ang = pth + ti
            # --- free: センサから点の手前までの経路 (点がある方向のみ)
            n_step = max(int((ri - 2 * sigma_r) / self.res), 0)
            if n_step > 0:
                d = np.arange(n_step) * self.res
                rr, cc, ok = self.w2i(px + d * np.cos(ang), py + d * np.sin(ang))
                if ok.any():
                    self.L[rr[ok], cc[ok]] -= free_gain * ci

            # --- occupied: ビーム幅に沿った円弧状の領域
            half = np.radians(beam_deg / 2) / np.cos(min(abs(ti), np.radians(60)))
            n_a = max(int(2 * half * ri / self.res), 3)
            aa = ang + np.linspace(-half, half, n_a)
            w = np.exp(-0.5 * (np.linspace(-2, 2, n_a)) ** 2)   # 中心ほど重い
            for k in range(-1, 2):     # 距離方向に±1セル
                rr, cc, ok = self.w2i(px + (ri + k * self.res) * np.cos(aa),
                                      py + (ri + k * self.res) * np.sin(aa))
                if ok.any():
                    np.add.at(self.L, (rr[ok], cc[ok]),
                              occ_gain * ci * w[ok] * (1.0 if k == 0 else 0.5))

        np.clip(self.L, self.l_min, self.l_max, out=self.L)


def build_local_map(scans, frames, size_m=26.0, res=0.05, **kw):
    """複数フレームを積分してローカルマップを作る"""
    g = OccupancyGrid(size_m=size_m, res=res)
    for f in frames:
        s = scans[f]
        g.integrate(s["r"], s["th"], s["snr"], **kw)
    return g
