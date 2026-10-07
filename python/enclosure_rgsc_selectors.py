#!/usr/bin/python3
"""Enclosure fork: do RGSC's heads pick high-regret states? The RGSC paper's Figure 7, on our own rows.

    python enclosure_rgsc_selectors.py --checkpoint CKPT --npz ROWS.npz [...] [--device cuda]

The network's regret ranking and regret value heads run on self-play positions that carry a regret target (C70, the
regret of Eq. 2 realized in the game). In each game from the normal start, one position is picked five ways:

- ranking: the ranking head's highest (what RGSC offers its buffer from a game's played positions);
- regret value: the regret value head's highest;
- random: any position (the uniform choice of Go-Exploit, the paper's baseline);
- shallowest: the game's first position (regret falls with depth by its definition, so depth alone is a baseline);
- best: the highest realized regret (the ceiling).

It prints the mean realized regret of each selector's picks, and each head's rank correlation with the realized
regret, over all positions and within depth buckets (a head that only learned depth scores 0 within a bucket).
"""

import argparse

import numpy as np
import torch

from katago.train.data_processing_pytorch import read_npz_training_data
from katago.train.model_pytorch import Model

SIZE = 19
MODE_SGFPOS = 4


def load(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Model(ckpt["config"], SIZE)
    sd = ckpt["model"]
    prefix = [k for k in sd if k.endswith("value_head.conv1.weight")][0][:-len("value_head.conv1.weight")]
    model.load_state_dict({k[len(prefix):]: v for k, v in sd.items()}, strict=True)
    return model.to(device).eval()


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
    args = ap.parse_args()
    model = load(args.checkpoint, args.device)
    cols = {"regret": [], "flag": [], "turn": [], "start": [], "mode": [], "game": [], "pred": [], "rank": []}
    n = 0
    with torch.no_grad():
        for batch in read_npz_training_data(args.npz, batch_size=256, world_size=1, rank=0, pos_len=SIZE,
                                            device=args.device, randomize_symmetries=False, include_meta=False,
                                            model_config=model.config):
            post = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))[0]
            g = batch["globalTargetsNC"].float().cpu().numpy()
            cols["pred"].append(post[-2].float().cpu().numpy())
            cols["rank"].append(post[-1].float().cpu().numpy())
            cols["regret"].append(g[:, 70]); cols["flag"].append(g[:, 71]); cols["turn"].append(g[:, 51])
            cols["start"].append(g[:, 53]); cols["mode"].append(g[:, 55])
            h = g[:, 41:47].astype(np.int64)
            cols["game"].append(h[:, 0] * 4194304 * 4194304 + h[:, 1] * 4194304 + h[:, 3] + 31 * h[:, 4])
            n += len(g)
            if n >= args.max_rows:
                break
    c = {k: np.concatenate(v) for k, v in cols.items()}
    keep = (c["flag"] > 0) & (c["mode"] != MODE_SGFPOS) & (c["start"] == 0)
    c = {k: v[keep] for k, v in c.items()}
    print(f"rows read {n:,}; normal-game rows with a regret target {keep.sum():,}; games {len(np.unique(c['game'])):,}")
    print(f"regret value head: mean {c['pred'].mean():.3f} (realized {c['regret'].mean():.3f}); ranking scores "
          f"mean {c['rank'].mean():.3f}, spread {c['rank'].std():.3f}")

    rng = np.random.default_rng(0)
    picks = {"ranking": [], "regret value": [], "random": [], "shallowest": [], "best": []}
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
                  "best": idx[np.argmax(c["regret"][idx])]}
        for k, i in choice.items():
            picks[k].append(c["regret"][i])
            depths[k].append(c["turn"][i])
    print(f"\nper game (games with at least 8 sampled rows: {used:,}), the realized regret of the picked position:")
    for k in picks:
        p = np.array(picks[k])
        print(f"  {k:13} mean {p.mean():.3f}  (median depth {int(np.median(depths[k]))})")

    print("\nrank correlation with the realized regret (Spearman):")
    print(f"  all positions:  ranking {rank_corr(c['rank'], c['regret']):+.3f}   regret value "
          f"{rank_corr(c['pred'], c['regret']):+.3f}   depth {rank_corr(-c['turn'], c['regret']):+.3f}")
    for lo in range(0, 120, 24):
        sel = (c["turn"] >= lo) & (c["turn"] < lo + 24)
        print(f"  turns {lo:3d}-{lo + 23:3d} (n {sel.sum():>6,}): ranking {rank_corr(c['rank'][sel], c['regret'][sel]):+.3f}   "
              f"regret value {rank_corr(c['pred'][sel], c['regret'][sel]):+.3f}")


if __name__ == "__main__":
    main()
