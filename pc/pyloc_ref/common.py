# -*- coding: utf-8 -*-
"""pyloc との比較で共通に使う、データ読み込みと1走行ぶんの再生。"""
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import parse_dat, to_scan, merge_scans, FRAME_PERIOD  # noqa: E402
from egomotion import estimate_velocity                               # noqa: E402
import imu as I                                                       # noqa: E402

STEP = 3            # 何フレームずつまとめて1更新にするか


def expand(pat):
    """PowerShell は * を展開しないので自前で展開する。"""
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(pat)
    return hits[0]


def load_run(dat, imu_csv, frames_csv):
    """.dat / _imu.csv / _frames.csv を読んで、1更新ぶんずつ取り出せる形にする。"""
    scans = to_scan(parse_dat(expand(dat)))
    fr = I.load_frames(expand(frames_csv))
    d = I.load_imu(expand(imu_csv))
    bias = I.estimate_bias(d)["bias"]
    tb = I.dtheta_per_frame(d, fr, bias)
    dth = dict(zip(tb.frame.values, tb.dtheta.values))
    return scans, dth, sorted(scans)


def steps(scans, dth, frames):
    """(前進量[m], 回転量[rad], 距離[m]の配列, 方位[rad]の配列, SNR) を順に返す。"""
    for i in range(0, len(frames) - STEP, STEP):
        fs = frames[i:i + STEP]
        sc = merge_scans(scans, fs)
        if len(sc["r"]) == 0:
            continue
        e = estimate_velocity(sc["r"], sc["th"], sc["doppler"])
        v = e["vx"] if e["ok"] else 0.0
        rot = float(sum(dth.get(f, 0.0) for f in fs))
        yield v * STEP * FRAME_PERIOD, rot, sc["r"], sc["th"], sc["snr"]


def landmarks_from_map(map_npz):
    """占有格子の占有セル中心を、そのまま点ランドマークにする。"""
    z = np.load(map_npz)
    occ, res, org = z["occ"], float(z["resolution"]), z["origin"]
    rr, cc = np.where(occ)
    return np.stack([org[0] + (cc + 0.5) * res,
                     org[1] + (rr + 0.5) * res], axis=1)
