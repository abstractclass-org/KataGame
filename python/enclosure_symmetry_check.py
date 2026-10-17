#!/usr/bin/python3
"""Enclosure fork: the training augmentation's symmetries against the C++ SymmetryHelpers.

    testsymmetry -dump ../cpp/tests/data/enclosure_inputs.txt SYM_FILE
    python enclosure_symmetry_check.py SYM_FILE

For each position and symmetry in SYM_FILE: the augmented V8 inputs must equal the C++ inputs of the
image position (fillRowV8 of getSymBoard), and every legal move's policy entry must land on the C++
image move (getSymMoveLoc, the game's SymActionMap), in both policy channels:

- channel 0 (the mover's move), mapped with the mover's nodes from the row's inputs;
- channel 1 (the next move, which C++ lists for the next mover; 2026-10-14): the position stands for the next
  position, the row's own inputs keep its planes but name the other player as the mover, and valueTargetsNCHW C7
  (flagged by globalTargetsNC C75) holds the position's mover's nodes, as trainingwrite.cpp writes them. A mapping
  that took channel 1's nodes from the row's mover fails here; the count of such failures is printed too.

Exit code 0 when all agree.
"""

import sys

import numpy as np
import torch

from katago.train.games.enclosure import (FEATURE_PLANES_START, FEATURES_PER_PLAYER, NEXT_NODES_FLAG, NEXT_NODES_PLANE,
                                          NUM_ACTIONS, SIZE, apply_symmetry_inputs, apply_symmetry_policy,
                                          apply_symmetry_policy_targets)

CHANNELS = 215
GLOBALS = 5


def blocks(path):
    lines = open(path).read().splitlines()
    for i in range(0, len(lines), 5):
        _, label, symmetry = lines[i].split()
        inp = [int(v) for v in lines[i + 1].split()[2:]]
        glob = [float(v) for v in lines[i + 2].split()[1:]]
        image = [int(v) for v in lines[i + 3].split()[2:]]
        moves = [int(v) for v in lines[i + 4].split()[2:]]
        yield label, int(symmetry), inp, glob, image, moves


def dense(indices):
    x = torch.zeros(CHANNELS * SIZE * SIZE)
    x[indices] = 1.0
    return x.view(1, CHANNELS, SIZE, SIZE)


def lands(mapped, images, total):
    return all(mapped[b].item() == i + 1 for i, b in enumerate(images)) and mapped.sum().item() == total


def main():
    checked, failures, old_failures = 0, 0, 0
    per_symmetry = {}
    for label, symmetry, inp, glob, image, moves in blocks(sys.argv[1]):
        x = dense(inp)
        g = torch.tensor([glob])
        got = apply_symmetry_inputs(x, symmetry).reshape(-1).nonzero().flatten().tolist()
        ok_inputs = got == sorted(image)
        actions, images = moves[0::2], moves[1::2]
        numbers = torch.arange(1, len(actions) + 1, dtype=torch.float32)
        total = numbers.sum().item()
        policy = torch.zeros(1, 1, NUM_ACTIONS + 1)
        policy[0, 0, actions] = numbers
        ok_policy = lands(apply_symmetry_policy(policy, symmetry, x, g)[0, 0], images, total)
        # Channel 1: this position is the next position; the row names the other player as its mover.
        mover = int(round(glob[0]))
        targets = torch.zeros(1, 2, NUM_ACTIONS + 1)
        targets[0, 1, actions] = numbers
        g_row = g.clone()
        g_row[0, 0] = 1 - mover
        value = torch.zeros(1, NEXT_NODES_PLANE + 1, SIZE, SIZE)
        value[0, NEXT_NODES_PLANE] = x[0, FEATURE_PLANES_START + FEATURES_PER_PLAYER * mover]
        global_targets = torch.zeros(1, NEXT_NODES_FLAG + 5)
        global_targets[0, NEXT_NODES_FLAG] = 1
        ok_next = lands(apply_symmetry_policy_targets(targets, symmetry, x, g_row, value, global_targets)[0, 1],
                        images, total)
        if not lands(apply_symmetry_policy(targets[:, 1:], symmetry, x, g_row)[0, 0], images, total):
            old_failures += 1  # the mapping before 2026-10-14: channel 1 with the row's mover's nodes
        checked += 1
        per_symmetry.setdefault(symmetry, [0, 0])[0] += 1
        if not (ok_inputs and ok_policy and ok_next):
            failures += 1
            per_symmetry[symmetry][1] += 1
            if failures <= 10:
                print(f"FAIL {label} symmetry {symmetry}: inputs {'ok' if ok_inputs else 'differ'}, "
                      f"policy {'ok' if ok_policy else 'differs'}, next-move policy {'ok' if ok_next else 'differs'}")
    for s in sorted(per_symmetry):
        print(f"symmetry {s}: {per_symmetry[s][0]} positions, {per_symmetry[s][1]} failures")
    print(f"{checked} checked, {failures} failures; channel 1 with the row's mover's nodes (the old mapping) would "
          f"fail {old_failures}")
    return 0 if failures == 0 and checked > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
