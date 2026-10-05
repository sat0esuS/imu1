# imu1 — CoreS3 IMU Logger

[![ビルド](https://github.com/sat0esuS/imu1/actions/workflows/build-and-deploy.yml/badge.svg)](https://github.com/sat0esuS/imu1/actions/workflows/build-and-deploy.yml)
[![書き込みページ](https://img.shields.io/badge/書き込みページ-sat0esus.github.io%2Fimu1-2e5fa3)](https://sat0esus.github.io/imu1/)

ミリ波レーダー(TI IWR6843AOP)によるSLAMで、**レーダーが原理的に測れない「回転」をIMUで埋める**ための一式。

レーダーの自己速度推定は `v_d = -v·cosθ` を解いているが、この式に回転の項は無い。
その場で回転しても静止物体のドップラーは0のままなので、**ヨー角は原理的に観測できない**。
曲がり角で軌跡が崩れるのはこれが理由で、実装の不備ではない。

| 測るもの | 担当 | 理由 |
|---|---|---|
| 回転（ヨー角） | IMU のジャイロZ | レーダーでは原理的に観測不能 |
| 並進（前進量） | レーダーのドップラー | 加速度の2回積分は数秒でメートル級に発散する |

加速度から位置は出さない。地磁気も使わない（屋内は鉄筋・EV・配電盤で方位が出ない）。

---

## ブラウザから書き込む

**<https://sat0esus.github.io/imu1/>**

Chrome / Edge で開いて、CoreS3 をUSB-Cで繋いでボタンを押すだけ。
Arduino IDE も PlatformIO も要らない。

> CoreS3 には **BMI270(6軸) + BMM150** が最初から載っているので、**配線は不要**。
> 本体を台車に固定するだけで使える。

---

## 構成

```
firmware/        CoreS3 ファームウェア (PlatformIO)
  src/main.cpp     内蔵BMI270を読んでUSBシリアルへ1行ずつ流すだけ
pc/              PC側 Python
  logger.py        レーダーとIMUを1プロセスで記録（TI Visualizer の置き換え）
  imu.py           バイアス推定・ヨー積分・レーダーフレームへの整列
  analyze_d1.py    D-1実験（ジャイロの素性測定）の解析
  run_moving_imu.py  IMU対応版のMCL
web/             GitHub Pages（書き込みページ）
docs/            手順書
```

既存の `preprocess.py` / `mcl.py` / `egomotion.py` は**変更していない**。
`logger.py` が出す `.dat` は TI Visualizer と同一形式なので、既存の解析はそのまま動く。

---

## 使い方

### 1. PC側の準備

```bash
git clone https://github.com/sat0esuS/imu1.git
cd imu1/pc
pip install -r requirements.txt
python logger.py --list      # シリアルポートを調べる
```

| 表示 | 用途 |
|---|---|
| XDS110 Class Application/User UART | `--radar-cfg` |
| XDS110 Class Auxiliary Data Port | `--radar-data` |
| USB Serial / M5Stack / ESP32-S3 | `--imu` |

### 2. D-1実験（MCLに入れる前に必ず）

A系列でレーダーの癖を測ったのと同じことを、IMUに対しても行う。
レーダーを繋がなくても、IMUだけで記録できる。

| | 内容 | 測るもの |
|---|---|---|
| **D-1a** | 静止60秒 × 3回 | バイアス値と再現性 |
| **D-1b** | 静止5分 | ドリフト率 [deg/分] |
| **D-1c** | 正確な90°旋回 × 10回 | **スケール補正係数**と1回あたりの誤差 |
| **D-1d** | ルート1周（合計360°） | 実走行での累積誤差 |

```bash
python logger.py --no-radar --imu COM7 --out d1a_1 --duration 60
python logger.py --no-radar --imu COM7 --out d1c            # 旋回の合間に必ず数秒止める
python analyze_d1.py --bias d1a_*_imu.csv --scale d1c_*_imu.csv --expect 90 --turns 10
```

`d1_report.txt` と図が出る。レポートの末尾に次の手順で使うコマンドがそのまま印字される。

**D-1c が本命。** ここで出る「1回の旋回あたり±何度」が、そのままMCLの動作モデルのパラメータになる。

### 3. 本番の記録

```bash
python logger.py --cfg xwr68xx_AOP_profile.cfg \
                 --radar-cfg COM5 --radar-data COM6 --imu COM7 \
                 --out run2 --outdir data
```

| 出力 | 中身 |
|---|---|
| `run2_<日時>.dat` | **Visualizer と同一形式** |
| `run2_<日時>_frames.csv` | 各レーダーフレームの受信時刻 |
| `run2_<日時>_imu.csv` | IMUサンプル（デバイス時刻とホスト時刻の両方） |
| `run2_<日時>_meta.json` | 開始時刻、ポート、送った cfg |

### 4. MCL

```bash
python run_moving_imu.py data/run2_*.dat \
    --map map_corridor.npz --init 0.9 1.0 90 \
    --imu data/run2_*_imu.csv --frames data/run2_*_frames.csv \
    --gyro-scale 0.982 --gyro-sign +1

# 比較用：同じデータで回転だけ無効にする
python run_moving_imu.py ... --no-imu
```

回転が観測できるぶん、動作モデルのノイズを絞れる。

| | IMUなし | IMUあり |
|---|---|---|
| `s_th`（回転ノイズ） | 0.04 rad (2.3°) | **0.012 rad (0.7°)** |
| `s_cross`（回転時の位置の不確かさ） | 0.06 | **0.02** |

パラメータを**緩める方向ではなく締める方向**に変えられるのが、IMUを足す実質的な効果。

---

## 判定の基準は 13.2°

このセンサは角度分解能 26.4° なので、距離 r の点は元々 `r·tan(13.2°)` だけ横にぼやけている。
向きの誤差がこれを超えると、IMU由来の誤差がセンサ本来のぼやけより大きくなる。
ドリフトもスケール誤差も周回誤差も、すべてこの一本の線で合否を判断できる。

---

## 時刻の扱い

レーダーの `.dat` には**絶対時刻が一切入っていない**（フレーム番号だけ）。
別々に記録して後から合わせると 0.3〜0.5秒ずれるが、

| 曲がり方 | 角速度 | 許容Δt (13.2°以内) |
|---|---|---|
| ゆっくり(90°を3秒) | 30°/s | 0.44 s |
| 普通(90°を2秒) | 45°/s | 0.29 s |
| 速い(90°を1秒) | 90°/s | 0.15 s |

となり、ギリギリかアウトになる。
`logger.py` が両方を1つのプロセスで受けて同じ時計で打刻することで、**Δtは約10ms**に収まる。

CoreS3 の `micros()` は**間隔**が正確だがPCと時計が違う。
ホスト受信時刻は**時計が共通**だがUSBのバッファで数ms揺らぐ。
`imu.py` は `t_host ≈ a·t_dev + b` を最小二乗で当てはめて両方の良いところを取る。

---

## ハードウェアの注意

手持ちの **Adafruit LSM9DS1** を使う場合は Port A（G2=SDA, G1=SCL）に挿すことになるが、
**Grove の 5V を LSM9DS1 の VIN に入れてはいけない。**
あのボードは I2C のプルアップが VIN 電圧に引っ張られるため、
SDA/SCL が 5V になって ESP32-S3 の 3.3V GPIO を壊す。使うなら 3.3V 給電にすること。

取り付けは剛に固定し、水準器で水平を出す。角度 φ 傾くとヨーレートが cos φ 倍に縮む
（10°傾きで1.5%、90°旋回あたり1.4°の誤差）。

---

## ライセンス

MIT
