#!/usr/bin/python3
"""Enclosure fork: RGSC's regret, and alternatives to it, on our own self-play games, before any change to the run.

    python enclosure_regret_variants.py RUN_DIR [--files 40] [--gamma 0.9]

Self-play SGFs keep, at every move after the game's start, the root search value (White's win, loss and no-result,
2 decimals; full and cheap searches alike) and the result. With v_i = win - loss (White's view) and z the outcome:

- vanilla: RGSC's Eq. 2, the mean of (v_i - z)^2 over the rest of the game (the root value stands in for the selected
  move's value, which is the next position's root value up to the search);
- discounted: the step-discounted regret of Lin (2026), the same errors weighted by gamma^(i - t);
- swing: how much the search value moves over the next moves, sum gamma^k (v_{t+k+1} - v_{t+k})^2 / sum gamma^k, a
  local-mistake signal that does not use z (our idea, not published);
- vanilla by depth: the vanilla regret's percentile among positions at the same move number (removes depth).

For each it prints the mean by depth, its rank correlation with depth, and where a per-game pick (the highest
position of the game, as RGSC's candidate from played positions) lands.
"""

import argparse
import glob
import os
import re

import numpy as np

VALUE = re.compile(r"C\[(-?\d\.\d\d) (-?\d\.\d\d) (-?\d\.\d\d) (-?[\d.]+) v=(\d+)")


def games(run_dir, files):
    paths = sorted(glob.glob(os.path.join(run_dir, "selfplay", "*", "sgfs", "*.sgfs")), key=os.path.getmtime)[-files:]
    for path in paths:
        for line in open(path, "rb"):
            text = line.decode("latin-1")
            root = re.search(r"C\[startTurnIdx=(\d+)[^]]*gtype=(\w+)", text)
            result = re.search(r"RE\[([WB0D])", text)
            if not root or not result:
                continue
            values = [float(m.group(1)) - float(m.group(2)) for m in VALUE.finditer(text)]
            z = {"W": 1.0, "B": -1.0}.get(result.group(1), 0.0)
            yield int(root.group(1)), root.group(2), np.array(values), z


def suffix_mean(err, weights=None):
    """For each t, the (weighted by gamma^(i-t)) mean of err[i] for i >= t."""
    n = len(err)
    if weights is None:
        return np.cumsum(err[::-1])[::-1] / np.arange(n, 0, -1)
    g = weights
    num, den, out = 0.0, 0.0, np.empty(n)
    for i in range(n - 1, -1, -1):
        num = err[i] + g * num
        den = 1.0 + g * den
        out[i] = num / den
    return out


def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--files", type=int, default=40)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--json", help="also write the report's figure data here")
    args = ap.parse_args()
    rows = {"depth": [], "game": [], "vanilla": [], "discounted": [], "swing": [], "value": [], "z": []}
    gtypes = {}
    n_games = 0
    for start, gtype, v, z in games(args.run_dir, args.files):
        gtypes[gtype] = gtypes.get(gtype, 0) + 1
        if gtype != "normal" or start != 0 or len(v) < 20:
            continue
        err = (v - z) ** 2
        swing_err = np.append((v[1:] - v[:-1]) ** 2, 0.0)
        rows["depth"].append(np.arange(len(v)))
        rows["game"].append(np.full(len(v), n_games))
        rows["vanilla"].append(suffix_mean(err))
        rows["discounted"].append(suffix_mean(err, args.gamma))
        rows["swing"].append(suffix_mean(swing_err, args.gamma))
        rows["value"].append(v)
        rows["z"].append(np.full(len(v), z))
        n_games += 1
    r = {k: np.concatenate(v) for k, v in rows.items()}
    print(f"games by type: {gtypes}; normal games from the start used: {n_games:,}, positions {len(r['depth']):,}")
    # The vanilla regret's percentile within its depth.
    pct = np.empty(len(r["depth"]))
    for d in np.unique(r["depth"]):
        sel = r["depth"] == d
        pct[sel] = (np.argsort(np.argsort(r["vanilla"][sel])) + 0.5) / sel.sum()
    r["vanilla by depth"] = pct
    variants = ("vanilla", "discounted", "swing", "vanilla by depth")
    print(f"\nmean by depth (gamma {args.gamma}):")
    print("  turns     " + "  ".join(f"{v:>16}" for v in variants))
    for lo in range(0, 120, 12):
        sel = (r["depth"] >= lo) & (r["depth"] < lo + 12)
        if sel.any():
            print(f"  {lo:3d}-{lo + 11:3d}   " + "  ".join(f"{r[v][sel].mean():>16.3f}" for v in variants))
    print("\nrank correlation with depth (negative: earlier is higher):")
    print("  " + "  ".join(f"{v} {rank_corr(r[v], r['depth']):+.3f}" for v in variants))
    print("\nwhere each variant's per-game pick lands (move number), and how often two variants pick the same move:")
    picks = {}
    for v in variants:
        p = []
        for g in range(n_games):
            idx = np.flatnonzero(r["game"] == g)
            p.append(idx[np.argmax(r[v][idx])])
        picks[v] = np.array(p)
        d = r["depth"][picks[v]]
        hist = np.histogram(d, bins=[0, 12, 24, 36, 60, 90, 121])[0] / len(d)
        print(f"  {v:>16}: median {int(np.median(d)):3d}; turns 0-11 {hist[0]:.0%}, 12-23 {hist[1]:.0%}, "
              f"24-35 {hist[2]:.0%}, 36-59 {hist[3]:.0%}, 60-89 {hist[4]:.0%}, 90-120 {hist[5]:.0%}")
    for i, a in enumerate(variants):
        for b in variants[i + 1:]:
            print(f"  same pick, {a} / {b}: {(picks[a] == picks[b]).mean():.0%}")
    if args.json:
        write_json(args, r, picks, variants, n_games)


def write_json(args, r, picks, variants, n_games):
    """The data of the report's figures (katago/python/enclosure_rgsc_report.py draws them)."""
    import json

    moves = np.arange(int(r["depth"].max()) + 1)
    by_move = {v: [float(r[v][r["depth"] == m].mean()) for m in moves] for v in variants}
    z = r["z"]
    value = r["value"]
    decisive = z != 0
    accuracy = {
        "sign_right": [float(np.mean(np.sign(value[(r["depth"] == m) & decisive]) == z[(r["depth"] == m) & decisive]))
                       for m in moves],
        "mean_abs_v": [float(np.abs(value[r["depth"] == m]).mean()) for m in moves],
        "mean_err": [float(((value - z) ** 2)[r["depth"] == m].mean()) for m in moves],
    }
    pick_depths = {v: r["depth"][picks[v]].astype(int).tolist() for v in variants}
    # Example games: decisive ones whose swing pick lies in the middle game, the value turning against the start.
    examples = []
    for g in range(n_games):
        idx = np.flatnonzero(r["game"] == g)
        sp, vp = int(r["depth"][picks["swing"][g]]), int(r["depth"][picks["vanilla"][g]])
        zz = float(z[idx[0]])
        if zz != 0 and 20 <= sp <= 80 and np.abs(value[idx]).max() > 0.5 and len(examples) < 6:
            examples.append({"value": value[idx].round(3).tolist(), "vanilla": r["vanilla"][idx].round(4).tolist(),
                             "swing": r["swing"][idx].round(5).tolist(), "z": zz, "swing_pick": sp,
                             "vanilla_pick": vp})
    out = {"games": n_games, "positions": int(len(r["depth"])), "gamma": args.gamma, "by_move": by_move,
           "accuracy": accuracy, "pick_depths": pick_depths, "examples": examples}
    with open(args.json, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
