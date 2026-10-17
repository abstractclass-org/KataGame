#!/usr/bin/python3
"""Enclosure fork: per shuffled training window, which targets its rows carry and how spread its scores are.

    python enclosure_window_stats.py SHUFFLEDDATA_DIR [SHUFFLEDDATA_DIR ...] [--files 6]

For each window (a dir from the loop's shuffle, with train/*.npz), over a sample of its files: the mean target
weight of the Enclosure spatial targets (future positions: C33; occupancy: C34; cut edges: the ownership weight
C27 times the C68 flag; territory income: C27 times C69), which is what the reported losses are scaled by
(they divide by every row), and the final score margin's spread (scoreDistrN, the mover's view, in points).
"""

import argparse
import glob
import os

import numpy as np

from katago.train.games.enclosure import SCORE_SCALE
from katago.train.model_pytorch import EXTRA_SCORE_DISTR_RADIUS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--files", type=int, default=6)
    args = ap.parse_args()
    print("window rows w_futurepos w_occupancy w_cutedge w_income margin_mean |margin|_mean margin_std")
    for d in args.dirs:
        files = sorted(glob.glob(os.path.join(d, "train", "*.npz")))
        files = files[:: max(1, len(files) // args.files)][: args.files]
        g, margins = [], []
        for f in files:
            z = np.load(f)
            g.append(z["globalTargetsNC"])
            sd = z["scoreDistrN"].astype(np.float64)
            mid = sd.shape[1] // 2
            centres = SCORE_SCALE * (np.arange(sd.shape[1]) - mid + 0.5)
            margins.append((sd * centres).sum(axis=1) / np.maximum(sd.sum(axis=1), 1e-9))
        g = np.concatenate(g)
        m = np.concatenate(margins)
        has_score = g[:, 27] > 0
        print(os.path.basename(d.rstrip("/")), len(g),
              f"{g[:, 33].mean():.3f}", f"{g[:, 34].mean():.3f}",
              f"{(g[:, 27] * g[:, 68]).mean():.3f}", f"{(g[:, 27] * g[:, 69]).mean():.3f}",
              f"{m[has_score].mean():.1f}", f"{np.abs(m[has_score]).mean():.1f}", f"{m[has_score].std():.1f}")


if __name__ == "__main__":
    main()
