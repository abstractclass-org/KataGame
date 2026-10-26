#!/usr/bin/python3
"""Enclosure fork: what RGSC's self-play left behind: its buffers and the regret targets in the rows.

    python enclosure_rgsc_check.py BUFFER_FILE [BUFFER_FILE ...] [TDATA_NPZ ...]

Each buffer (program/rgsc.cpp's file; the vanilla buffer, and the swing buffer since 2026-10-11): the finished-game
count, the states, their regrets, how often games restarted from them, their depth in moves (a game that keeps states,
2026-10-27: the turn number, in actions), and their origins (a search tree's state or a played position; unknown in
files saved before 2026-10-11). The rows: the share of rows
carrying a regret target (C71) and its spread (C70), which lies in [0, 4] (a squared difference of values in
[-1, 1]); the same for the swing target (C72, flag C73: scaled and capped at 4), which RGSC writes on the same rows;
and the games' restart sources (C74: 0 none, 1 the vanilla buffer, 2 the swing buffer).
"""

import sys

import numpy as np

ORIGINS = {0: "unknown", 1: "search tree", 2: "played"}


def check_buffer(path):
    games, regrets, samples, depths, origins = 0, [], [], [], []
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:  # not saved yet: self-play saves it every 50 games counted (none without the heads)
        lines = []
        print(f"buffer {path}: no file yet")
    for line in lines:
        w = line.split()
        if w[0] == "gamesFinished":
            games = int(w[1])
        elif w[0] == "entry":
            regrets.append(float(w[1]))
            samples.append(int(w[2]))
            depths.append(int(w[3]))
            rest = w[4 + 2 * int(w[3]):]  # the origin, an optional last token
            origins.append(int(rest[0]) if rest else 0)
        elif w[0] == "state":
            # A state of a game that keeps states (2026-10-27, Blueprints): "state VERSION GAME KEY REGRET SAMPLES
            # ORIGIN TURN KOMI ADVANTAGEPLA ADVANTAGE" and the state's text. Its depth is its turn number.
            regrets.append(float(w[4]))
            samples.append(int(w[5]))
            origins.append(int(w[6]))
            depths.append(int(w[7]))
    if not regrets:  # as expected while the self-play model has no heads for this buffer (nothing is offered)
        print(f"buffer {path}: {games} games finished, empty")
        return True
    by_origin = ", ".join(f"{ORIGINS.get(o, o)} {origins.count(o)}" for o in sorted(set(origins)))
    print(f"buffer {path}: {games} games finished, {len(regrets)} states; regret min {min(regrets):.3f} "
          f"median {np.median(regrets):.3f} max {max(regrets):.3f}; restarts drawn {sum(samples)} "
          f"(states drawn at least once: {sum(1 for s in samples if s > 0)}); depth in moves "
          f"min {min(depths)} median {int(np.median(depths))} max {max(depths)}; origins: {by_origin}")
    return 0.0 <= min(regrets) and max(regrets) <= 4.0


def check_rows(files):
    g = np.concatenate([np.load(f)["globalTargetsNC"] for f in files])
    ok = True
    for name, col, flag in (("regret", 70, 71), ("swing", 72, 73)):
        flagged = g[:, flag] > 0
        zero_elsewhere = bool((g[~flagged, col] == 0).all())
        if flagged.any():
            r = g[flagged, col]
            print(f"rows: {len(g)}, with a {name} target {flagged.mean():.3f}; {name} min {r.min():.3f} "
                  f"mean {r.mean():.3f} max {r.max():.3f}; rows without one have C{col} = 0: {zero_elsewhere}")
            ok = ok and r.min() >= 0.0 and r.max() <= 4.0
        else:
            print(f"rows: {len(g)}, none with a {name} target; C{col} = 0 everywhere: {zero_elsewhere}")
        ok = ok and zero_elsewhere
    # A run without swing targets (rgscSwingScale 0, 2026-10-27: Blueprints) has none on any row.
    if not (g[:, 73] > 0).any():
        same_rows = True
        print("no swing targets (rgscSwingScale 0)")
    else:
        same_rows = bool(((g[:, 71] > 0) == (g[:, 73] > 0)).all())
        print(f"the swing target is on exactly the rows with a regret target: {same_rows}")
    # Games by restart source: a game's rows share its hash (C41-46) and its source (C74).
    hashes, first = np.unique(g[:, 41:47], axis=0, return_index=True)
    sources = g[first, 74].astype(int)
    print("games by restart source: " + ", ".join(
        f"{('none', 'vanilla', 'swing')[s] if s in (0, 1, 2) else s} {int((sources == s).sum())}"
        for s in sorted(set(sources.tolist()))) + f" (of {len(hashes)})")
    consistent = all(len(set(g[(g[:, 41:47] == h).all(axis=1), 74].tolist())) == 1 for h in hashes[:200])
    print(f"every row of a game has its game's source: {consistent}")
    restarted = g[:, 55] == 4
    source_set = g[:, 74] > 0
    print(f"rows with a source are rows of games from a start position (C55 = 4): {bool((~source_set | restarted).all())}")
    return ok and same_rows and consistent and bool((~source_set | restarted).all())


def main():
    buffers = [a for a in sys.argv[1:] if not a.endswith(".npz")]
    rows = [a for a in sys.argv[1:] if a.endswith(".npz")]
    ok = all([check_buffer(b) for b in buffers])
    if rows:
        ok = check_rows(rows) and ok
    print("checks passed" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
