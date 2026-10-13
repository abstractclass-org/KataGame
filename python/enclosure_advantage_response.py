#!/usr/bin/python3
"""Enclosure fork: does a V9 network read the playout doubling advantage? The judgement's second test (running-v2a.md
"The playout-gap inputs (V9) live").

    python enclosure_advantage_response.py RUN_DIR NAME [NAME ...] [--positions 512] [--files 64] [--threads 2]

V9's two global inputs (C++ NNInputs::fillRowV9): input 5 is 1 when one side searches with more visits, input 6 is half
the number of doublings, from the player to move's side (+1.5: the mover has 8 times the visits, maxAsymmetricRatio).
The migration gave both zero weights, so a network reads them only as far as training on uneven games (1% of
self-play) taught it.

- The response: positions from even games (input 5 is 0) in the newest self-play files, through each checkpoint NAME
  (models/ or models_extra/, SWA weights, as exported) as they are and with the inputs set to each advantage. For each:
  the mean change of the mover's win chance (over all positions, and over the undecided ones, win chance 0.1 to 0.9
  as they are), the share of undecided positions whose win chance moves the advantage's way, and the mean change of
  the lead in points. A network that reads the advantage moves the win chance up for +, down for -, more for more.
- The calibration: the uneven games' own rows in those files (input 5 is 1), the actual result against the network's
  win chance with the inputs zeroed, by the mover's side of the advantage: how much the advantage was worth in these
  games. Part of them were evened out by komi (minAsymmetricCompensateKomiProb), which the network sees in the scores.
"""

import argparse
import pathlib

import numpy as np
import torch

from enclosure_symmetry_spread import SIZE, checkpoint_dirs, load
from katago.train import modelconfigs
from katago.train.data_processing_pytorch import fit_global_input_nc

ADVANTAGES = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)  # half the doublings: 1.5 is 8 times the visits


def read_rows(run, files, limit):
    """The newest self-play files' rows: (binary inputs, 7 global inputs, results from the mover's side)."""
    paths = sorted((run / "selfplay").glob("*/tdata/*.npz"), key=lambda p: p.stat().st_mtime)[-files:]
    binary, glob, result = [], [], []
    for p in paths:
        with np.load(p) as z:
            b = np.unpackbits(z["binaryInputNCHWPacked"], axis=2)[:, :, :SIZE * SIZE]
            binary.append(b.reshape(b.shape[0], b.shape[1], SIZE, SIZE))
            glob.append(fit_global_input_nc(z["globalInputNC"], 7))
            gt = z["globalTargetsNC"]
            result.append(gt[:, 0] + 0.5 * (1.0 - gt[:, 0] - gt[:, 1]))  # win 1, loss 0, no result counts half
    binary, glob, result = np.concatenate(binary), np.concatenate(glob), np.concatenate(result)
    even = np.flatnonzero(glob[:, 5] == 0)
    uneven = np.flatnonzero(glob[:, 5] == 1)
    pick = even[np.linspace(0, len(even) - 1, min(limit, len(even))).astype(int)]
    print(f"{len(paths)} files, {len(glob)} rows: {len(pick)} positions from even games for the response, "
          f"{len(uneven)} rows of uneven games for the calibration")
    return (torch.from_numpy(binary[pick].astype(np.float32)), torch.from_numpy(glob[pick]).float(),
            torch.from_numpy(binary[uneven].astype(np.float32)), torch.from_numpy(glob[uneven]).float(), result[uneven])


@torch.no_grad()
def evaluate(model, binary, glob):
    out = model(binary, glob)[0]
    return torch.softmax(out[1].float(), dim=1)[:, 0], out[2][:, 2].float() * model.lead_multiplier


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--positions", type=int, default=512)
    ap.add_argument("--files", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    run = pathlib.Path(args.run)
    dirs = checkpoint_dirs(run)
    binary, glob, u_binary, u_glob, u_result = read_rows(run, args.files, args.positions)
    for name in args.names:
        model, weights = load(dirs[name] / "model.ckpt")
        if modelconfigs.get_num_global_input_features(model.config) < 7:
            print(f"{name}: V8 inputs, no advantage to read")
            continue
        base_win, base_lead = evaluate(model, binary, glob)
        undecided = (base_win >= 0.1) & (base_win <= 0.9)
        print(f"{name} ({weights}): advantage (half doublings), mean change of the win chance (all | undecided, "
              f"n {int(undecided.sum())}), undecided moving its way, mean change of the lead")
        for h in ADVANTAGES:
            g = glob.clone()
            g[:, 5], g[:, 6] = 1.0, h
            win, lead = evaluate(model, binary, g)
            d = win - base_win
            way = ((d > 0) if h > 0 else (d < 0))[undecided].float().mean()
            print(f"  {h:+.1f} ({2 ** (2 * abs(h)):.0f}x {'the mover' if h > 0 else 'the opponent'}): "
                  f"{d.mean():+.4f} | {d[undecided].mean():+.4f}, {way:.2f}, lead {(lead - base_lead).mean():+.1f}")
        if len(u_result):
            zeroed = u_glob.clone()
            zeroed[:, 5:7] = 0
            win0, _ = evaluate(model, u_binary, zeroed)
            win1, _ = evaluate(model, u_binary, u_glob)
            for label, m in (("the mover has the advantage", u_glob[:, 6] > 0), ("the opponent has it", u_glob[:, 6] < 0)):
                if m.any():
                    r = torch.from_numpy(u_result).float()[m]
                    print(f"  calibration, {label} ({int(m.sum())} rows): result {r.mean():.3f}, network with the inputs "
                          f"zeroed {win0[m].mean():.3f}, with them {win1[m].mean():.3f}")


if __name__ == "__main__":
    main()
