#!/usr/bin/python3
"""Enclosure fork: how much each policy head puts on never-legal listings, per symmetry, on the symmetry monitor's
positions.

    python enclosure_next_move_probe.py RUN_DIR NAME [NAME ...]

C++ lists a move between two of its player's own nodes only from the lower end, so a listing from the upper end is
never a legal move and never a correct target. Until 2026-10-14 the trainer mapped policy channel 1 (the next move)
with the mover's nodes, so in some symmetries it put the next mover's closing moves on such listings, and the
networks learned that (games/enclosure/docs/decisions/build.md, "The training's symmetry handling"). For each checkpoint NAME
(models/ or models_extra/, SWA weights) and symmetry: the main policy's mass on the mover's never-legal listings, and
the next-move head's on the next mover's, split by who moves next (the opponent after a turn's last edge, the same
player after its first). After the fix the next-move head's share should fall towards the main policy's.
The positions are RUN_DIR/symmetry_positions.npz (enclosure_symmetry_spread.py).
"""

import pathlib
import sys

import numpy as np
import torch

from enclosure_symmetry_spread import SIZE, checkpoint_dirs, load, load_positions
from katago.train import modelconfigs
from katago.train.data_processing_pytorch import fit_global_input_nc
from katago.train.enclosure_symmetry import NUM_ACTIONS, NUM_OFFSETS, NUM_SYMMETRIES, OFFSETS, apply_symmetry_inputs, player_nodes


def never_legal_tables():
    """Per action: its source point, its destination point, and whether it is listed from its upper end."""
    src, dst, upper = np.zeros(NUM_ACTIONS, np.int64), np.zeros(NUM_ACTIONS, np.int64), np.zeros(NUM_ACTIONS, bool)
    for y in range(SIZE):
        for x in range(SIZE):
            for k, (dx, dy) in enumerate(OFFSETS):
                bx, by = x + dx, y + dy
                if 0 <= bx < SIZE and 0 <= by < SIZE:
                    a = (y * SIZE + x) * NUM_OFFSETS + k
                    src[a], dst[a], upper[a] = y * SIZE + x, by * SIZE + bx, (bx, by) < (x, y)
    return torch.from_numpy(src), torch.from_numpy(dst), torch.from_numpy(upper)


def main():
    run = pathlib.Path(sys.argv[1])
    dirs = checkpoint_dirs(run)
    binary, global_raw = load_positions(run / "symmetry_positions.npz")
    src, dst, upper = never_legal_tables()
    player = torch.from_numpy(global_raw[:, 0]).round().long()
    last_edge = torch.from_numpy(global_raw[:, 1]) < 0.75
    next_player = torch.where(last_edge, 1 - player, player)
    print(f"{binary.shape[0]} positions; the opponent moves next in {int(last_edge.sum())}")

    def share(p, x, who):
        nodes = player_nodes(x, who)
        return (p[:, :NUM_ACTIONS] * (nodes[:, src] & nodes[:, dst] & upper)).sum(dim=1)

    for name in sys.argv[2:]:
        model, weights = load(dirs[name] / "model.ckpt")
        g = torch.from_numpy(fit_global_input_nc(global_raw, modelconfigs.get_num_global_input_features(model.config))).float()
        print(f"{name} ({weights}): symmetry, main policy | next move, opponent next | next move, same player next")
        with torch.no_grad():
            for s in range(NUM_SYMMETRIES):
                x = apply_symmetry_inputs(binary, s)
                out = model(x, g)[0][0].float()
                p0, p1 = torch.softmax(out[:, 0], dim=1), torch.softmax(out[:, 1], dim=1)
                nxt = share(p1, x, next_player)
                print(f"  {s}: {share(p0, x, player).mean():.4f} | {nxt[last_edge].mean():.4f} | {nxt[~last_edge].mean():.4f}")


if __name__ == "__main__":
    main()
