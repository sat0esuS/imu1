# IMU(M5Stack CoreS3)導入手順

レーダーは原理的に回転を測れない(`v_d = -v·cosθ` に回転の項がない)。
そこを IMU で埋めて、曲がり角で姿勢が失われるのを防ぐための手順書。

**分業**

| 測るもの | 担当 | 理由 |
|---|---|---|
| 回転（ヨー角） | IMU のジャイロZ | レーダーでは原理的に観測不能 |
| 並進（前進量） | レーダーのドップラー | IMU の加速度を2回積分すると数秒でメートル級に発散する |

加速度から位置を出そうとしないこと。地磁気も使わない（屋内は鉄筋・EV・配電盤で方位が出ない）。

---

## 0. 結論：配線はゼロ

**CoreS3 には BMI270(6軸)+ BMM150 が最初から載っている。**
内蔵IMUを使えば配線もレベル変換も不要で、本体を台車に固定するだけで済む。

手持ちの Adafruit LSM9DS1 を使う場合は Port A(G2=SDA, G1=SCL)に挿すことになるが、
**Grove の 5V を LSM9DS1 の VIN に入れてはいけない**。
あのボードは I2C のプルアップが VIN 電圧に引っ張られるため、
SDA/SCL が 5V になって ESP32-S3 の 3.3V GPIO を壊す。使うなら 3.3V 給電にすること。

まずは内蔵BMI270で進めて、性能に不満が出たら LSM9DS1 と D-1 で比較すればよい。

---

## 1. ファームウェアを書き込む

### Arduino IDE の設定

| 項目 | 値 |
|---|---|
| ボードマネージャURL | `https://static-cdn.m5stack.com/resource/arduino/package_m5stack_index.json` |
| ボード | **M5CoreS3** |
| **USB CDC On Boot** | **Enabled** ← これを忘れると Serial に何も出ない |
| ライブラリ | **M5Unified**（ライブラリマネージャから） |

`cores3_imu/cores3_imu.ino` を書き込む。

### 書き込み後の確認

画面に `IMU LOGGER (BMI270)` と出て、`gz` の値が動けば成功。
本体を机に置いて静止させ、`bias measuring` が `bias +0.xxxx dps` に変われば準備完了。

画面下部のタッチ操作:

| 位置 | 動作 |
|---|---|
| 左 | バイアス測定をやり直す（静止させてから押す） |
| 中央 | 表示のヨー角をゼロに戻す |
| 右 | 出力の ON / OFF |

**このファームは積分も姿勢推定もしない。** 生の値を1行ずつ流すだけで、
解析はすべて PC 側の Python で行う。画面の値は動作確認用。

---

## 2. 取り付け

- **剛に固定する。** 両面テープで浮かせると振動がジャイロに乗る
- **水平を出す。** 水準器で合わせる。角度 φ 傾くとヨーレートが cos φ 倍に縮む（10°傾きで1.5%、90°旋回あたり1.4°の誤差）
- **レーダーと前方向を揃える。** 揃えておくと符号で悩まない
- 台車の**回転中心に近い位置**が望ましい（ジャイロ自体は位置に依らないが、後で加速度を見るときに楽）

---

## 3. 記録する（Visualizer を置き換える）

`logger.py` が Visualizer の代わりになる。やっていることは同じで、

- CFG ポート(115200)に `.cfg` の各行を送る
- DATA ポート(921600)から来たバイト列をそのまま `.dat` に落とす

これに加えて、**レーダーのフレームと IMU のサンプルに同じ時計で時刻を打つ**。

### ポートを調べる

```bash
pip install pyserial
python logger.py --list
```

| 表示 | 用途 |
|---|---|
| XDS110 Class Application/User UART | `--radar-cfg` |
| XDS110 Class Auxiliary Data Port | `--radar-data` |
| USB Serial / M5Stack / ESP32-S3 | `--imu` |

### 記録する

```bash
python logger.py --cfg xwr68xx_AOP_profile.cfg \
                 --radar-cfg COM5 --radar-data COM6 --imu COM7 \
                 --out run2 --outdir data
```

`Ctrl-C` で停止。出力は4つ:

| ファイル | 中身 |
|---|---|
| `run2_<日時>.dat` | **Visualizer と同一形式。`parse_dat()` がそのまま使える** |
| `run2_<日時>_frames.csv` | `t_host, frame, n_bytes` — 各レーダーフレームの受信時刻 |
| `run2_<日時>_imu.csv` | `t_host, t_dev_us, gx..gz, ax..az` |
| `run2_<日時>_meta.json` | 開始時刻、ポート、送った cfg |

IMU だけ記録したいとき（D-1実験）は `--no-radar` を付ける。

### 時刻の扱い

CoreS3 の `micros()` は**間隔**が正確だが PC と時計が違う。
ホスト受信時刻は**時計が共通**だが USB のバッファで数 ms 揺らぐ。
`imu.py` は `t_host ≈ a·t_dev + b` を最小二乗で当てはめて両方の良いところを取る。
`imu.summary()` が出す `時計の係数 a` が 1 から大きくずれていたら、デバイス側の時計を疑う。

---

## 4. D-1 実験：MCL に入れる前に必ずやる

A系列でレーダーの癖を測ったのと同じことを IMU に対して行う。
**買ってすぐ MCL に入れないこと。** ジャイロには必ずバイアスがある。

| | 内容 | 測るもの |
|---|---|---|
| **D-1a** | 完全に静止させて60秒 × 3回 | バイアス値と、その再現性 |
| **D-1b** | 静止したまま5分間 | ドリフト率 [deg/分] |
| **D-1c** | 床に正確な90°をマーキングし、旋回 × 10回 | **スケール補正係数**と1回あたりの誤差 |
| **D-1d** | ルートを1周（合計360°回る） | 実走行での累積誤差 |

**D-1c が本命。** ここで出る「1回の旋回あたり±何度」が、そのまま MCL の動作モデルの
パラメータになる。10回のうち数回でも構わないが、多いほど良い。

### 実施

```bash
# D-1a: 静止60秒を3本
python logger.py --no-radar --imu COM7 --out d1a_1 --duration 60
python logger.py --no-radar --imu COM7 --out d1a_2 --duration 60
python logger.py --no-radar --imu COM7 --out d1a_3 --duration 60

# D-1b: 静止5分
python logger.py --no-radar --imu COM7 --out d1b --duration 300

# D-1c: 旋回10回（静止3秒 → 90°旋回 → 静止3秒 … を繰り返す）
python logger.py --no-radar --imu COM7 --out d1c

# D-1d: ルート1周
python logger.py --no-radar --imu COM7 --out d1d
```

D-1c は**旋回の合間に必ず数秒止める**。止まっている区間からバイアスを測り直すため。

### 解析

```bash
python analyze_d1.py --bias  d1a_1_*_imu.csv d1a_2_*_imu.csv d1a_3_*_imu.csv \
                     --drift d1b_*_imu.csv \
                     --scale d1c_*_imu.csv --expect 90 --turns 10 \
                     --loop  d1d_*_imu.csv --loop-expect 360
```

`d1_report.txt` と図4枚が出る。レポートの末尾に、次の手順で使うコマンドがそのまま印字される。

### 読み方

| 指標 | 目安 | 外れていたら |
|---|---|---|
| 3回のバイアスのばらつき | < 0.05 deg/s | 大きければ毎回測り直す（`--bias-track`） |
| 5分のドリフト | < 13.2°（ビーム半値幅） | 大きければ `--bias-track` 必須 |
| スケール補正係数 | 0.95 〜 1.05 | 大きく外れるなら取り付けの傾きを疑う |
| 補正後の旋回ばらつき | ±2° 程度 | — |
| D-1d の周回誤差 | < 13.2° | スケールかバイアス追従を見直す |

**13.2° を基準にしている理由**：このセンサは角度分解能 26.4° なので、
距離 r の点は元々 `r·tan(13.2°)` だけ横にぼやけている。
向きの誤差がこれを超えると、IMU 由来の誤差がセンサ本来のぼやけより大きくなる。

---

## 5. MCL に組み込む

`run_moving_imu.py` が `run_moving.py` の IMU 対応版。
変更点は動作モデルの `dtheta` を IMU から与えることだけで、`mcl.py` 本体は元から
`dtheta` を受け取れる作りになっている。

```bash
python run_moving_imu.py data/run2_20261012_143000.dat \
    --map map_corridor.npz --init 0.9 1.0 90 \
    --imu    data/run2_20261012_143000_imu.csv \
    --frames data/run2_20261012_143000_frames.csv \
    --gyro-scale 0.982 --gyro-sign +1 \
    --out mcl_imu.png
```

`--gyro-scale` と `--gyro-sign` は `analyze_d1.py` が出す値をそのまま入れる。

### IMU あり / なしの比較

```bash
python run_moving_imu.py ... --no-imu --out mcl_noimu.png
```

同じデータで回転だけ無効にする。**曲がり角でどれだけ改善したかの証拠**になるので、
発表にはこの2枚を並べる。

### 自動で変わるパラメータ

回転が観測できるようになったぶん、動作モデルのノイズを絞れる。

| | IMU なし | IMU あり |
|---|---|---|
| `s_th`（回転ノイズ） | 0.04 rad (2.3°) | **0.012 rad (0.7°)** |
| `s_cross`（回転時の位置の不確かさ） | 0.06 | **0.02** |

D-1c の結果に応じて `--s-th` で上書きできる。
**パラメータを緩める方向ではなく締める方向に変えられる**のが、IMU を足す実質的な効果。

### 主なオプション

| オプション | 既定 | 意味 |
|---|---|---|
| `--gyro-scale` | 1.0 | D-1c のスケール補正係数 |
| `--gyro-sign` | +1 | ジャイロZの符号。左回りが正になるように |
| `--gyro-bias` | 自動 | 省略すると静止区間から自動推定 |
| `--bias-track` | off | 静止区間ごとにバイアスを測り直す（長時間走行向け） |
| `--s-th` | 自動 | 回転ノイズを手で指定 |
| `--no-imu` | off | 比較用。IMUを読んでも回転に使わない |
| `--motion` | doppler | 並進の出し方（doppler / const / none） |

---

## 6. 符号の確認

MCL の θ は「前方=0、**左が正**（反時計回り）」。
CoreS3 の取り付け向き次第でジャイロZの符号が逆になる。

```python
import imu as I
d = I.load_imu("d1c_..._imu.csv")
print(I.check_sign(d, t0=3.0, t1=5.0))   # 左に90°回した区間の時刻を渡す
# -> {'integrated_deg': 92.7, 'sign_z': 1.0, 'message': 'そのまま sign_z=+1 でよい'}
```

`sign_z: -1.0` と出たら `--gyro-sign -1` を付ける。

---

## 7. ファイル一覧

| ファイル | 役割 |
|---|---|
| `cores3_imu/cores3_imu.ino` | CoreS3 ファームウェア。内蔵BMI270を読んでUSBシリアルへ流す |
| `logger.py` | レーダーとIMUを1プロセスで記録（Visualizerの置き換え） |
| `imu.py` | IMUログの読み込み、バイアス推定、ヨー積分、フレームへの整列 |
| `analyze_d1.py` | D-1実験の解析とレポート生成 |
| `run_moving_imu.py` | IMU対応版のMCL |

`preprocess.py` / `mcl.py` / `egomotion.py` は**変更していない**。
`.dat` の形式が Visualizer と同一なので、既存の解析はすべてそのまま動く。

---

## 8. つまずきやすいところ

| 症状 | 原因 |
|---|---|
| Serial に何も出ない | Arduino IDE の **USB CDC On Boot** が Disabled |
| `IMU NOT FOUND` | `M5.config()` の `internal_imu` が無効。M5Unified が古い可能性 |
| レーダーのフレームが0件 | DATAポートの指定違い、または cfg が通っていない |
| `frame間隔` が 66.7ms から大きくずれる | USB の取りこぼし。他のアプリ（Visualizer）が同じポートを掴んでいないか確認 |
| 周回誤差が大きい | スケール未適用、またはバイアスが温度で動いている（`--bias-track`） |
| 向きが逆に回る | `--gyro-sign -1` |
