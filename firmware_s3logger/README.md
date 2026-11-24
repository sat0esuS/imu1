# firmware_s3logger — CoreS3 単体でレーダーと IMU を記録する

PC を使わずに、M5Stack CoreS3 だけで IWR6843AOPEVM（レーダー）と内蔵 BMI270（IMU）を
**同じ時計で打刻して** microSD に記録するファームウェア（ESP-IDF）。

```
IWR6843AOPEVM ──USB(CP2105)──> CoreS3 ──> microSD
                               ↑ 内蔵 BMI270（FIFO）
```

- CoreS3 の USB-C を **USB ホスト**にして EVM の micro-B とつなぐ（Type-C→micro-B ケーブル）。
  CP2105 のドライバは Espressif 公式の `usb_host_cp210x_vcp` を使う
- cfg（`cfg/corridor.cfg` と同じ内容）は `main.cpp` に埋め込んであり、起動時に CoreS3 から送る
- 出力は `pc/logger.py` と**同じ形式**。`pc/` の解析スクリプトがそのまま使える

## 出力

| ファイル | 中身 |
|---|---|
| `s3_YYYYMMDD_hhmmss.dat` | レーダーの生バイト列（フレーム先頭から始まる） |
| `…_frames.csv` | `t_host,frame,n_bytes`。t_host は記録開始からの秒（esp_timer） |
| `…_imu.csv` | `t_host,t_dev_us,gx,gy,gz,ax,ay,az`（deg/s、G） |
| `…_meta.json` | 記録時間、フレーム数、取りこぼし数など |

IMU の時刻は 2 本ある。

- `t_dev_us` … BMI270 のサンプル番号 × 5000 µs。間隔は正確だが、センサの時計は
  esp_timer より約 1.3% 速い（実測 202.6 Hz）
- `t_host` … FIFO を読んだ時刻から逆算した値（esp_timer）。読み出しのばらつきで ±数 ms 揺れる

**IMU の時刻は必ず `imu.load_imu()` を通して使うこと。**
`t_host ≈ a·t_dev + b` の当てはめで両方の良いところを取る（PC 記録と同じ考え方）。

## ビルドと書き込み

ESP-IDF 5.5 の「ESP-IDF 5.5 CMD」で、1 行ずつ実行する。

```
cd firmware_s3logger
idf.py set-target esp32s3        （最初の 1 回だけ）
idf.py -p COM5 flash
```

- 初回は依存部品（usb_host_cp210x_vcp、M5Unified）が自動でダウンロードされる
- 書き込めないときは、CoreS3 のリセットボタンを約 3 秒長押ししてダウンロードモードにする
- 動作中は USB-C がホストになっているので、PC につないでも書き込めない（上と同じ操作で書き込む）
- ファイル名用の RTC は、**そのビルドの初回起動時にビルド時刻に合わせる**（数十秒〜数分遅れる）

## 使い方

1. FAT32 の microSD を CoreS3 に挿す
2. EVM のスイッチを**すべて OFF**（PC で使うときと同じ）にする。
   S1.1・S1.2 が ON だと CFG が 60 ピン側に回り、cfg が通らない
3. CoreS3 と EVM をつないで起動。画面に次が出れば準備完了
   - `SD ok` / `USB 5V out: on` / `cfg ok (30 lines)` / `radar 15.0 fps` / `imu 200 Hz 前後`
4. 画面下の枠をタッチで記録開始、もう一度で停止。**microSD は停止してから抜く**

## 試験結果（2026-11-24、静止、70.5 秒）

- レーダー：1058 フレーム、欠落 0。打刻の一定周期からのずれ 0.24 ms（標準偏差）
- IMU：14290 サンプル、抜け 0。静止時の加速度 0.995 G、ジャイロ Z のバイアス −0.03 deg/s

## 注意

- EVM（5 V・約 500 mA）に CoreS3 の電池から給電するので、電池の減りが早い。
  電池が空になると、PC につないでも画面が点かないことがある。その場合は USB を抜いて
  電源ボタンを 6 秒以上長押しして完全に切り、しばらく充電してから起動する
- USB 経由（CP2105 内部）の遅れは、PC 記録との同時比較でまだ評価していない
