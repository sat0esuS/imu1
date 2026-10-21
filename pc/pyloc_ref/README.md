# pyloc_ref — 標準的なMCL実装との比較

自分の `mcl.py` が何をしているかを確かめるために、広く参照されている MCL 実装を
同じデータで動かし、差分を1つずつ戻して**どの設計が効いているか**を測った記録。

- 原典: **[NaokiAkai/pyloc](https://github.com/NaokiAkai/pyloc)**（Copyright (C) 2019 Naoki Akai、MPL-2.0）
- 解説記事: <https://qiita.com/NaokiAkai/items/44b2a160e47a8de44abf>

`pyloc_mcl.py` は原典 `mcl/mcl.py` の派生物なので **MPL-2.0 のまま**。
このディレクトリの他のファイルはこのリポジトリのライセンス（MIT）に従う。

---

## 結論

| 要素 | 外すと誤差の中央値 |
|---|---|
| いまの `mcl.py` のまま | **0.46 m** |
| **一様床を小さくする (0.1 → 1e-5)** | **5.18 m** ← 決定的 |
| SNR重みを外す | 0.81 m |
| σ を定数 0.10 に固定 | 0.47 m |
| σ を定数 0.27 に固定 | 0.53 m |
| `temperature` を外す (1.0) | 0.57 m |
| 推定を重み付き平均にする | 0.30 m |

移植した pyloc をそのまま動かすと **誤差 3.4 m**（3シードとも同じ解に落ちる）。
等方の観測分散をどう振っても改善しない。

| 測定分散 | σ [m] | 誤差 (seed 0,1) |
|---|---|---|
| 0.01 | 0.10 | 3.41 / 3.46 |
| 0.04 | 0.20 | 5.09 / 5.12 |
| 0.25 | 0.50 | 4.24 / 5.52 |
| 1.00 | 1.00 | 4.36 / 4.43 |
| 4.00 | 2.00 | 4.54 / 4.62 |

**効いていたのは観測ノイズの形でも推定法でもなく、外れ値に対する尤度の下限だった。**

### 一様床とは

観測点が壁からどれだけ離れていても、尤度に下限を設ける仕組み。

```python
# pyloc
prob = z_hit * norm_coef * exp(...) + z_rand * 10e-6
prob *= measurement_resolution              # 床 = 0.1 * 1e-5 * 0.1 = 1e-7

# mcl.py
p = z_hit * gauss + z_rand * (1.0 / max_dist)   # 床 = 0.15 / 1.5 = 0.1
```

床の高さが **100万倍**違う。対数にすると、壁から外れた点1個がパーティクルに与える罰が
**−16.1 対 −2.3** で 7 倍の差になる。

ミリ波レーダーの点群は1フレームに **2〜8個**しかなく、多重反射やサイドローブによる
ゴースト点が混ざる。pyloc の床では**ゴースト点1個で正解のパーティクルが却下される**。
460ステップ続けば正解は残らない。

pyloc は**ランドマーク4点・観測が素直なシミュレータ**向けの実装なので、
その想定の中では何の問題もない。実測のミリ波点群に持ってくると破綻する、というだけ。

---

## 構造の対応

5段の骨格は両者で完全に同じ。

| 段 | pyloc | `mcl.py` |
|---|---|---|
| 初期化 | `reset_particles(var_x, var_y, var_yaw)` | `init_around(x, y, deg, s_xy, s_deg)` |
| 動作モデル | `update_particles(delta_dist, delta_yaw)` | `predict(dx, dtheta)` |
| 観測モデル | `calculate_weights(measurements)` | `update(scan_r, scan_th, snr)` |
| 姿勢推定 | `estimate_robot_pose()` 重み付き平均 | `estimate_mode()` 最頻値 |
| リサンプリング | `resample_particles()` ルーレット | `low_variance_resample()` |
| ESS 閾値 | 0.5 | 0.5（同じ） |
| 地図との照合 | 全ランドマークへ総当たりで最近傍距離 | 尤度場（距離変換で前計算） |

地図との照合は、**計算しているもの自体は同じ**（最も近い壁までの距離）。
距離変換で前計算しておくと配列参照1回で済む、という実装の違い。

---

## 使い方

```bash
cd pc/pyloc_ref

# pyloc を実データで動かす
python run_pyloc.py ../data/run3_*.dat --map ../map_corridor.npz \
    --imu ../data/run3_*_imu.csv --frames ../data/run3_*_frames.csv \
    --init 0 0 90 --true-end -3 11.754

# 要素別の寄与を測る
python ablate.py ../data/run3_*.dat --map ../map_corridor.npz \
    --imu ../data/run3_*_imu.csv --frames ../data/run3_*_frames.csv \
    --init 0 0 90 --true-end -3 11.754 --seeds 3
```

`*` はスクリプト側で展開するので PowerShell でもそのまま打てる。

### ファイル

| | |
|---|---|
| `pyloc_mcl.py` | 原典 `mcl/mcl.py` の Python 3 移植（MPL-2.0） |
| `common.py` | `.dat` / `_imu.csv` / `_frames.csv` を読んで1更新ずつ取り出す |
| `run_pyloc.py` | pyloc を実データで動かす |
| `ablate.py` | `mcl.py` の観測モデルから要素を1つずつ抜いて測る |

### 移植で変えたところ

数式には手を付けていない。

1. `print 'x'` → `print('x')`（Python 2 → 3、3箇所）
2. 最近傍ランドマーク探索の線形走査 → KD木（**答えは同一**、速度のみ）
3. パーティクル方向のループ → numpy のベクトル化（同上）

原典のまま純Pythonの三重ループ（パーティクル × 観測 × ランドマーク）だと、
4000パーティクル × 3841ランドマークで1ステップ十数秒、1走行で数時間かかる。
移植版は **18 秒**。

### 比較の条件

- 地図は `map_corridor.npz` の占有セル **3841個**をそのまま点ランドマークにした（5 cm 刻み）
- `odom_noise1..4` の既定値 `(20.0, 0.5, 2.0, 0.3)` は、1ステップあたりの実効ノイズが
  `mcl.py` の `motion_noise=(0.12, 0.012)` に近くなるよう合わせたもの。
  原典の既定値 `(20, 10, 10, 30)` は彼のシミュレータ用
- `ablate.py` の並進ノイズ既定値は **0.12**。`mcl.py` の既定 0.06 のままだと
  廊下方向で多峰性が出て、シードによって 3 m 級に外れることがある（`--s-x` で変えられる）
