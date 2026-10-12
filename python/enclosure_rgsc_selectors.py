#!/usr/bin/python3
"""Enclosure fork: do RGSC's heads pick high-regret states? The RGSC paper's Figure 7, on our own rows.

    python enclosure_rgsc_selectors.py --checkpoint CKPT --npz ROWS.npz [...] [--source vanilla|swing] [--use-swa]
                                       [--device cuda]

One source's ranking and value heads run on self-play positions that carry the source's target, realized in the
game: vanilla (C70, the regret of Eq. 2; flag C71) or swing (C72, the value swing; flag C73). Only games from the
normal start count, as only they offer candidates to the buffers: restarted games (from either buffer) are left out.
In each game, one position is picked six ways:

- ranking: the ranking head's highest (what RGSC offers its buffer from a game's played positions);
- regret value: the value head's highest;
- random: any position (the uniform choice of Go-Exploit, the paper's baseline);
- shallowest: the game's first position (vanilla regret falls with depth by its definition);
- by depth: the position whose move number has the highest mean target, fitted on the other half of the games
  (depth alone as a selector, for a target that does not simply fall with depth);
- best: the highest realized target (the ceiling).

It prints the mean realized target of each selector's picks, and each head's rank correlation with the realized
target, over all positions and within depth buckets (a head that only learned depth scores 0 within a bucket).
--use-swa runs the checkpoint's SWA weights, which the run exports and self-play uses.
"""

import argparse

import numpy as np
import torch

from katago.train.data_processing_pytorch import read_npz_training_data
from katago.train.load_model import load_model
from enclosure_rgsc_analysis import MODE_SGFPOS, SOURCES  # each source's target and flag columns, and its heads

SIZE = 19


def load(path, device, use_swa):
    """The checkpoint's network, or its SWA weights (what the run exports), as KataGo's training loads them."""
    model, swa_model, _ = load_model(path, use_swa=use_swa, device=device)
    return (swa_model if use_swa else model).eval()


def rank_corr(a, b):
    """Spearman's rank correlation."""
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-rows", type=int, default=60000)
    ap.add_argument("--source", choices=sorted(SOURCES), default="vanilla", help="RGSC's source: its target and heads")
    ap.add_argument("--use-swa", action="store_true", help="the checkpoint's SWA weights (the run's exports)")
    ap.add_argument("--json", help="also write the report's figure data here")
    args = ap.parse_args()
    src = SOURCES[args.source]
    model = load(args.checkpoint, args.device, args.use_swa)
    cols = {"regret": [], "flag": [], "turn": [], "start": [], "mode": [], "game": [], "pred": [], "rank": []}
    n = 0
    with torch.no_grad():
        for batch in read_npz_training_data(args.npz, batch_size=256, world_size=1, rank=0, pos_len=SIZE,
                                            device=args.device, randomize_symmetries=False, include_meta=False,
                                            model_config=model.config):
            g = batch["globalTargetsNC"].float().cpu().numpy()
            if g.shape[1] <= src["flag"]:  # rows from before the source's columns: no target
                continue
            post = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))[0]
            cols["pred"].append(post[src["heads"][0]].float().cpu().numpy())
            cols["rank"].append(post[src["heads"][1]].float().cpu().numpy())
            cols["regret"].append(g[:, src["regret"]]); cols["flag"].append(g[:, src["flag"]])
            cols["turn"].append(g[:, 51]); cols["start"].append(g[:, 53]); cols["mode"].append(g[:, 55])
            h = g[:, 41:47].astype(np.int64)
            cols["game"].append(h[:, 0] * 4194304 * 4194304 + h[:, 1] * 4194304 + h[:, 3] + 31 * h[:, 4])
            n += len(g)
            if n >= args.max_rows:
                break
    c = {k: np.concatenate(v) for k, v in cols.items()}
    keep = (c["flag"] > 0) & (c["mode"] != MODE_SGFPOS) & (c["start"] == 0)
    c = {k: v[keep] for k, v in c.items()}
    print(f"{args.source}{' (SWA weights)' if args.use_swa else ''}: rows read {n:,}; normal-game rows with a target "
          f"{keep.sum():,}; games {len(np.unique(c['game'])):,}")
    print(f"value head: mean {c['pred'].mean():.3f} (realized {c['regret'].mean():.3f}); ranking scores "
          f"mean {c['rank'].mean():.3f}, spread {c['rank'].std():.3f}")
    var = np.var(c["regret"])
    print(f"value head R2 on these rows: {1 - np.mean((c['pred'] - c['regret']) ** 2) / var:+.3f}")
    # The same against a predictor from the move number alone: the mean target of each move, fitted on one half of the
    # games and applied to the other (both ways, so every game gets a prediction from games it is not in).
    half = (c["game"] % 2) == 0
    pred_move = np.empty(len(c["turn"]))
    for fit in (half, ~half):
        mean_by_move = {m: c["regret"][fit & (c["turn"] == m)].mean() for m in np.unique(c["turn"][fit])}
        pred_move[~fit] = [mean_by_move.get(m, c["regret"][fit].mean()) for m in c["turn"][~fit]]
    te = ~half
    print(f"on one half of the games: head R2 {1 - np.mean((c['pred'][te] - c['regret'][te]) ** 2) / np.var(c['regret'][te]):+.3f}, "
          f"move-number predictor R2 {1 - np.mean((pred_move[te] - c['regret'][te]) ** 2) / np.var(c['regret'][te]):+.3f}")

    rng = np.random.default_rng(0)
    picks = {"ranking": [], "regret value": [], "random": [], "shallowest": [], "by depth": [], "best": []}
    depths = {k: [] for k in picks}
    order = np.argsort(c["game"], kind="stable")
    games = np.split(order, np.flatnonzero(np.diff(c["game"][order])) + 1)
    used = 0
    for idx in games:
        if len(idx) < 8:  # a game with few sampled rows says little
            continue
        used += 1
        choice = {"ranking": idx[np.argmax(c["rank"][idx])], "regret value": idx[np.argmax(c["pred"][idx])],
                  "random": rng.choice(idx), "shallowest": idx[np.argmin(c["turn"][idx])],
                  "by depth": idx[np.argmax(pred_move[idx])], "best": idx[np.argmax(c["regret"][idx])]}
        for k, i in choice.items():
            picks[k].append(c["regret"][i])
            depths[k].append(c["turn"][i])
    print(f"\nper game (games with at least 8 sampled rows: {used:,}), the realized regret of the picked position:")
    for k in picks:
        p = np.array(picks[k])
        print(f"  {k:13} mean {p.mean():.3f}  (median depth {int(np.median(depths[k]))})")

    if args.json:
        import json
        corr = []
        for lo in range(0, 120, 24):
            sel = (c["turn"] >= lo) & (c["turn"] < lo + 24)
            corr.append({"turns": f"{lo}-{lo + 23}", "ranking": rank_corr(c["rank"][sel], c["regret"][sel]),
                         "regret value": rank_corr(c["pred"][sel], c["regret"][sel])})
        with open(args.json, "w") as f:
            json.dump({"games": used, "positions": int(keep.sum()),
                       "picks": {k: float(np.mean(v)) for k, v in picks.items()},
                       "pick_sd": {k: float(np.std(v) / np.sqrt(len(v))) for k, v in picks.items()},
                       "depths": {k: float(np.median(v)) for k, v in depths.items()}, "corr_by_depth": corr}, f)
        print(f"wrote {args.json}")
    print("\nrank correlation with the realized regret (Spearman):")
    print(f"  all positions:  ranking {rank_corr(c['rank'], c['regret']):+.3f}   regret value "
          f"{rank_corr(c['pred'], c['regret']):+.3f}   depth {rank_corr(-c['turn'], c['regret']):+.3f}")
    for lo in range(0, 120, 24):
        sel = (c["turn"] >= lo) & (c["turn"] < lo + 24)
        print(f"  turns {lo:3d}-{lo + 23:3d} (n {sel.sum():>6,}): ranking {rank_corr(c['rank'][sel], c['regret'][sel]):+.3f}   "
              f"regret value {rank_corr(c['pred'][sel], c['regret'][sel]):+.3f}")


if __name__ == "__main__":
    main()
