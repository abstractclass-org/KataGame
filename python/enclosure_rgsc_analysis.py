#!/usr/bin/python3
"""Enclosure fork: is RGSC (regret-guided search control) doing its job? Read from self-play's own rows.

    python enclosure_rgsc_analysis.py RUN_DIR [--folders N] [--files-per-folder K] [--metrics TRAIN_DIR ...]

From the rows (globalTargetsNC: C41-46 the game hash, C51 the turn, C53 the moves before the game's start, C55 the
game's mode, C70 the regret of Eq. 2, C71 its flag) of the newest N self-play folders:

1. Restarts: the share of games and rows that start from a buffer state (mode 4, a start position: v2a uses no other).
2. The premise: the realized regret at a restarted game's start state, against the regret of normal games' positions
   at the same depth. RGSC helps only if the buffer's states are harder than the positions self-play reaches anyway.
3. Where regret lies: the mean regret by depth in normal games, and the depths of the restart states.
4. The heads, on a scale: the regret value loss against the variance of the targets (an R-squared), and the ranking
   loss between its value with no skill (equal scores) and with a perfect ranking, for groups of 32 dealt by stride
   from a batch of 256 sorted rows (metrics_pytorch.loss_regret_ranking). Needs --metrics (train/NAME folders).
"""

import argparse
import glob
import json
import os

import numpy as np

MODE_SGFPOS = 4  # FinishedGameData::MODE_SGFPOS: a game from a start position (RGSC's restarts in v2a)
COLS = {"hash": [41, 42, 43, 44, 45, 46], "turn": 51, "start": 53, "mode": 55, "regret": 70, "flag": 71}


def load_rows(run_dir, folders, files_per_folder, rng):
    dirs = sorted(d for d in glob.glob(os.path.join(run_dir, "selfplay", "*", "tdata")) if os.path.isdir(d))
    dirs = sorted(dirs, key=os.path.getmtime)[-folders:]
    parts = []
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        if len(files) > files_per_folder:
            files = list(rng.choice(files, files_per_folder, replace=False))
        for f in files:
            g = np.load(f)["globalTargetsNC"]
            parts.append(g[:, sorted({c for v in COLS.values() for c in (v if isinstance(v, list) else [v])})])
    if not parts:
        raise SystemExit(f"no rows under {run_dir}/selfplay/*/tdata")
    cols = sorted({c for v in COLS.values() for c in (v if isinstance(v, list) else [v])})
    g = np.concatenate(parts)
    at = {c: i for i, c in enumerate(cols)}
    rows = {k: (g[:, [at[c] for c in v]] if isinstance(v, list) else g[:, at[v]]) for k, v in COLS.items()}
    rows["game"] = np.unique(rows["hash"], axis=0, return_inverse=True)[1].reshape(-1)
    return rows, len(dirs)


def restart_premise(rows):
    has = rows["flag"] > 0
    restart = rows["mode"] == MODE_SGFPOS
    games = np.unique(rows["game"])
    restarted_games = np.unique(rows["game"][restart])
    print(f"rows {len(has):,} ({has.mean():.1%} with a regret target); games {len(games):,}, "
          f"restarted {len(restarted_games):,} ({len(restarted_games) / max(1, len(games)):.1%}); "
          f"rows from restarts {restart.mean():.1%}")
    # A restarted game's start state: its row at turn == start (the first row written after the start).
    first = has & restart & (rows["turn"] == rows["start"])
    start_depth = rows["start"][first].astype(int)
    start_regret = rows["regret"][first]
    normal = has & (rows["mode"] != MODE_SGFPOS) & (rows["start"] == 0)
    depth = rows["turn"][normal].astype(int)
    regret = rows["regret"][normal]
    by_depth = {d: regret[depth == d] for d in np.unique(depth)}
    print(f"\nrestart start states with a row: {first.sum():,}; depth min {start_depth.min() if len(start_depth) else '-'} "
          f"median {int(np.median(start_depth)) if len(start_depth) else '-'} max {start_depth.max() if len(start_depth) else '-'}")
    if len(start_regret):
        matched = np.array([by_depth[d].mean() for d in start_depth if d in by_depth and len(by_depth[d]) >= 20])
        own = np.array([r for r, d in zip(start_regret, start_depth) if d in by_depth and len(by_depth[d]) >= 20])
        pct = np.array([(by_depth[d] < r).mean() for r, d in zip(start_regret, start_depth)
                        if d in by_depth and len(by_depth[d]) >= 20])
        print(f"realized regret at the start state: mean {own.mean():.3f} (median {np.median(own):.3f}); normal games at "
              f"the same depths: mean {matched.mean():.3f}; ratio {own.mean() / matched.mean():.2f}")
        print(f"percentile of each start state's regret among normal positions at its depth: mean {pct.mean():.2f} "
              f"(0.50 = no harder than usual), share above the median {(pct > 0.5).mean():.2f}")
    print("\nregret by depth, normal games (mean, positions):")
    for lo in range(0, 120, 12):
        sel = (depth >= lo) & (depth < lo + 12)
        n_start = ((start_depth >= lo) & (start_depth < lo + 12)).sum()
        if sel.any():
            print(f"  turns {lo:3d}-{lo + 11:3d}: regret {regret[sel].mean():.3f} (n {sel.sum():,}); restart states here {n_start}")
    return rows["regret"][has]


def ranking_scale(regrets, rng, batch=256, group=32, trials=400):
    """The ranking loss with no skill (equal scores) and with a perfect ranking, per group as in training."""
    none, best = [], []
    for _ in range(trials):
        r = np.sort(rng.choice(regrets, batch, replace=False))[::-1]
        k = max(1, batch // group)
        for i in range(k):
            g = r[i::k]
            b = ((g - g.min()) / (g.max() - g.min() + 1e-8)) ** 2 * g
            none.append(-(np.logaddexp.reduce(b) - np.log(len(g))))
            best.append(-b.max())
    return float(np.mean(none)), float(np.mean(best))


def heads_on_a_scale(regrets, metrics_dirs, rng):
    var = float(np.var(regrets))
    no_skill, perfect = ranking_scale(regrets, rng)
    print(f"\nregret targets: mean {regrets.mean():.3f}, variance {var:.4f}; ranking loss with no skill {no_skill:.4f}, "
          f"perfect {perfect:.4f}")
    for d in metrics_dirs:
        f = os.path.join(d, "metrics_train.json")
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        rows = [m for m in rows if m.get("rgrows", 0) > 0.01 and "rgloss" in m]
        if not rows:
            continue
        print(f"{os.path.basename(d.rstrip('/'))}: by training window (per row with a target)")
        step = max(1, len(rows) // 8)
        for m in rows[::step] + ([rows[-1]] if (len(rows) - 1) % step else []):
            rg, rk = m["rgloss"] / m["rgrows"], m["rkloss"] / m["rgrows"]
            skill = (no_skill - rk) / (no_skill - perfect)
            print(f"  row {int(m.get('window_end_batch', 0)):>9,}: regret value R2 {1 - rg / var:+.3f} (loss {rg:.4f}); "
                  f"ranking skill {skill:+.3f} (loss {rk:+.4f})")


def depth_curriculum(run_dir, files_per_folder, rng, buckets=(0, 12, 24, 36, 48, 60, 120)):
    """The RGSC paper's Figures 16-18 from the rows: by self-play model (data row), the share of restart states by
    depth, and the mean realized regret at them. Every folder since RGSC began, a few files each."""
    dirs = sorted((d for d in glob.glob(os.path.join(run_dir, "selfplay", "*", "tdata")) if os.path.isdir(d)),
                  key=os.path.getmtime)
    table = []
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        if not files:
            continue
        if len(files) > files_per_folder:
            files = list(rng.choice(files, files_per_folder, replace=False))
        g = np.concatenate([np.load(f)["globalTargetsNC"][:, [51, 53, 55, 70, 71]] for f in files])
        start = (g[:, 2] == MODE_SGFPOS) & (g[:, 0] == g[:, 1]) & (g[:, 4] > 0)
        if start.sum() < 5:
            continue
        depth, regret = g[start, 1], g[start, 3]
        shares = [((depth >= lo) & (depth < hi)).mean() for lo, hi in zip(buckets[:-1], buckets[1:])]
        row = int(os.path.basename(os.path.dirname(d)).split("-d")[-1]) if "-d" in d else 0
        table.append((row, len(depth), shares, float(regret.mean())))
    if not table:
        print("\nno restart rows found")
        return
    print("\nrestart states by depth, by self-play model (the paper's Figures 16-18):")
    print("  data row     starts  " + "  ".join(f"{lo:>3}-{hi - 1:<3}" for lo, hi in zip(buckets[:-1], buckets[1:]))
          + "  regret")
    step = max(1, len(table) // 20)
    for row, n, shares, reg in table[::step] + ([table[-1]] if (len(table) - 1) % step else []):
        print(f"  {row:>9,}  {n:>6}  " + "  ".join(f"{s:>7.0%}" for s in shares) + f"  {reg:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--folders", type=int, default=12, help="the newest self-play folders (one per network)")
    ap.add_argument("--files-per-folder", type=int, default=40)
    ap.add_argument("--metrics", nargs="*", default=[], help="trainer folders (train/NAME) for the heads' losses")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows, n = load_rows(args.run_dir, args.folders, args.files_per_folder, rng)
    print(f"{n} self-play folders read")
    regrets = restart_premise(rows)
    heads_on_a_scale(regrets, args.metrics, rng)
    depth_curriculum(args.run_dir, max(2, args.files_per_folder // 8), rng)


if __name__ == "__main__":
    main()
