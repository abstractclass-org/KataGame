#!/usr/bin/python3
"""Enclosure fork: what RGSC's self-play left behind: its buffer and the regret targets in the rows.

    python enclosure_rgsc_check.py BUFFER_FILE [TDATA_NPZ ...]

The buffer (program/rgsc.cpp's file): the finished-game count, the states, their regrets, how often games restarted
from them, their depth in moves. The rows: the share of rows carrying a regret target (C71) and its spread (C70),
which lies in [0, 4] (a squared difference of values in [-1, 1]).
"""

import sys

import numpy as np


def main():
    games, regrets, samples, depths = 0, [], [], []
    try:
        with open(sys.argv[1]) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:  # not saved yet: self-play saves it every 50 games counted (none without the heads)
        lines = []
        print(f"buffer: no file yet at {sys.argv[1]}")
    for line in lines:
        w = line.split()
        if w[0] == "gamesFinished":
            games = int(w[1])
        elif w[0] == "entry":
            regrets.append(float(w[1]))
            samples.append(int(w[2]))
            depths.append(int(w[3]))
    if not regrets:  # as expected while the self-play model has no regret heads (nothing is offered)
        print(f"buffer: {games} games finished, empty")
    else:
        print(f"buffer: {games} games finished, {len(regrets)} states; regret min {min(regrets):.3f} "
              f"median {np.median(regrets):.3f} max {max(regrets):.3f}; restarts drawn {sum(samples)} "
              f"(states drawn at least once: {sum(1 for s in samples if s > 0)}); depth in moves "
              f"min {min(depths)} median {int(np.median(depths))} max {max(depths)}")
    ok = not regrets or (0.0 <= min(regrets) and max(regrets) <= 4.0)
    if len(sys.argv) > 2:
        g = np.concatenate([np.load(f)["globalTargetsNC"] for f in sys.argv[2:]])
        flagged = g[:, 71] > 0
        r = g[flagged, 70]
        print(f"rows: {len(g)}, with a regret target {flagged.mean():.3f}; regret min {r.min():.3f} "
              f"mean {r.mean():.3f} max {r.max():.3f}; rows without one have C70 = 0: {bool((g[~flagged, 70] == 0).all())}")
        ok = ok and r.min() >= 0.0 and r.max() <= 4.0 and bool((g[~flagged, 70] == 0).all())
    print("checks passed" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
