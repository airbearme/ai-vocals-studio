#!/usr/bin/env python3
"""Analyze real F0 distribution across all Pacaveli (2Pac) acapellas."""
import glob
import numpy as np
import librosa
import pyworld as pw

SR = 22050
SR2 = "/home/coden607/Projects/ai-vocals-studio"
import sys
sys.path.insert(0, SR2)

files = sorted(glob.glob(SR2 + "/dataset/Pacaveli_processed/*.wav"))
print(f"Files: {len(files)}")
all_f0 = []
for p in files:
    y, sr = librosa.load(p, sr=SR, mono=True)
    y = y[:60 * SR]  # cap at 60s for quick read
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    f0, _ = pw.dio(y.astype(np.float64), SR, frame_period=5.0)
    v = f0[f0 > 0]
    print(f"{p.split('/')[-1]}: n={len(v)} median={np.median(v) if len(v) else 0:.1f}Hz "
          f"mean={np.mean(v) if len(v) else 0:.1f}Hz p5={np.percentile(v,5) if len(v) else 0:.1f}Hz "
          f"p95={np.percentile(v,95) if len(v) else 0:.1f}Hz")
    all_f0.extend(v.tolist())

a = np.asarray(all_f0)
print(f"\nALL: n={len(a)} median={np.median(a):.1f} mean={np.mean(a):.1f} "
      f"p2={np.percentile(a,2):.1f} p98={np.percentile(a,98):.1f}")
# voiced fraction in the male band 70..220
mask = (a > 70) & (a < 220)
print(f"frac in 70-220 Hz band: {np.mean(mask)*100:.1f}%  median in-band: {np.median(a[mask]):.1f}")