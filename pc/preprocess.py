# -*- coding: utf-8 -*-
"""
TI mmWave .dat → MCL用の2Dスキャンへの前処理

A系列で実測した補正をここで一括適用する:
  - 距離バイアス -0.16 m
  - 仰角チルト   -3.58 deg
  - 高さフィルタ (センサ高さ基準の水平帯だけ使う)
"""
import struct
import numpy as np
import pandas as pd

MAGIC = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

RANGE_BIAS = 0.16     # [m] A-1実測
TILT_DEG = 3.58       # [deg] A-1実測
FRAME_PERIOD = 0.0667  # [s] 15fps時


def parse_dat(path):
    """点群TLV(1)とSide Info(7)を抽出して DataFrame で返す。"""
    data = open(path, "rb").read()
    pos = data.find(MAGIC)
    rows = []
    while pos >= 0 and pos + 40 <= len(data):
        v, tl, pf, frame, cyc, nobj, ntlv, sub = struct.unpack("<8I", data[pos + 8:pos + 40])
        if pos + tl > len(data):
            break
        tp, P, S, ok = pos + 40, None, None, True
        for _ in range(ntlv):
            if tp + 8 > pos + tl:
                ok = False
                break
            t, l = struct.unpack("<2I", data[tp:tp + 8])
            pl = data[tp + 8:tp + 8 + l]
            if t == 1 and nobj:
                P = np.frombuffer(pl[:nobj * 16], dtype=np.float32).reshape(-1, 4)
            elif t == 7 and nobj:
                S = np.frombuffer(pl[:nobj * 4], dtype=np.int16).reshape(-1, 2)
            tp += 8 + l
        if ok and P is not None:
            for i in range(P.shape[0]):
                snr = S[i, 0] * 0.1 if S is not None and i < len(S) else np.nan
                rows.append((frame, *P[i], snr))
        pos = data.find(MAGIC, pos + tl)
    df = pd.DataFrame(rows, columns=["frame", "x", "y", "z", "doppler", "snr_db"])
    return df


def to_scan(df, z_band=(-0.5, 0.5), snr_min=None, doppler_filter=False,
            apply_bias=True, apply_tilt=True):
    """
    点群DataFrame → 2Dスキャン (r, theta, snr) のフレーム辞書。
    theta: センサ前方(+y)を0とし、反時計回り正 [rad]
    """
    d = df.copy()
    r3 = np.sqrt(d.x ** 2 + d.y ** 2 + d.z ** 2)
    if apply_bias:
        scale = np.where(r3 > 1e-6, (r3 - RANGE_BIAS) / np.maximum(r3, 1e-6), 0.0)
        d[["x", "y", "z"]] = d[["x", "y", "z"]].values * scale[:, None]
    if apply_tilt:
        t = np.radians(-TILT_DEG)   # 上向きチルトを打ち消す
        y, z = d.y.values.copy(), d.z.values.copy()
        d["y"] = y * np.cos(t) - z * np.sin(t)
        d["z"] = y * np.sin(t) + z * np.cos(t)
    if snr_min is not None:
        d = d[d.snr_db >= snr_min]
    if doppler_filter:
        d = d[d.doppler.abs() < 1e-6]      # 静止点のみ(C-1より補助的に)
    d = d[(d.z >= z_band[0]) & (d.z <= z_band[1])]

    scans = {}
    for f, g in d.groupby("frame"):
        r = np.hypot(g.x.values, g.y.values)
        th = np.arctan2(-g.x.values, g.y.values)   # センサ前方=0, 左が正
        keep = r > 0.2
        scans[int(f)] = dict(r=r[keep], th=th[keep], snr=g.snr_db.values[keep],
                             doppler=g.doppler.values[keep])
    return scans


def merge_scans(scans, frames=None):
    """複数フレームを1スキャンに統合(静止観測の点密度を稼ぐ)。"""
    ks = sorted(scans) if frames is None else [f for f in frames if f in scans]
    def cat(key):
        return np.concatenate([scans[k][key] for k in ks]) if ks else np.array([])
    return dict(r=cat("r"), th=cat("th"), snr=cat("snr"), doppler=cat("doppler"))
