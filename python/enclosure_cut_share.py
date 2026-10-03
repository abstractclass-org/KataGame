#!/usr/bin/python3
"""Enclosure fork: how often self-play cuts, per self-play model (one cycle's games each).

    python enclosure_cut_share.py SELFPLAY_DIR [--last 12] [--files 8]

For each model folder under SELFPLAY_DIR (selfplay/MODEL/tdata/*.npz), over a sample of its files and the rows
that carry the cut-edge target (C68): the share of on-edge points whose edge ends up cut (valueTargetsNCHW C1:
+1 the mover's cut, -1 the opponent's, 0 not cut, 2 no edge; trainingwrite.h), and the mover's share of them.
The measure behind katago-fork.md, "Why the score and the new targets' losses rose" (self-play turned to cutting).
"""

import argparse
import glob
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selfplay")
    ap.add_argument("--last", type=int, default=12)
    ap.add_argument("--files", type=int, default=8)
    args = ap.parse_args()
    dirs = [d for d in glob.glob(os.path.join(args.selfplay, "*")) if glob.glob(os.path.join(d, "tdata", "*.npz"))]
    dirs.sort(key=lambda d: min(os.path.getmtime(f) for f in glob.glob(os.path.join(d, "tdata", "*.npz"))))
    print("model first_file rows cut_share movers_cut_share")
    for d in dirs[-args.last:]:
        files = sorted(glob.glob(os.path.join(d, "tdata", "*.npz")), key=os.path.getmtime)
        sample = files[:: max(1, len(files) // args.files)][: args.files]
        on_edge = cut = own = rows = 0
        for f in sample:
            z = np.load(f)
            g = z["globalTargetsNC"]
            flagged = g[:, 68] > 0
            v = z["valueTargetsNCHW"][flagged, 1]
            on_edge += int(np.isin(v, (-1, 0, 1)).sum())
            cut += int((np.abs(v) == 1).sum())
            own += int((v == 1).sum())
            rows += int(flagged.sum())
        if not on_edge:
            continue
        stamp = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(files[0])))
        print(f"{os.path.basename(d)} {stamp} {rows} {cut / on_edge:.3f} {own / on_edge:.3f}")


if __name__ == "__main__":
    main()
