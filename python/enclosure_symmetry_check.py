#!/usr/bin/python3
"""Enclosure fork: the training augmentation's symmetries against the C++ SymmetryHelpers.

    testsymmetry -dump ../cpp/tests/data/enclosure_inputs.txt SYM_FILE
    python enclosure_symmetry_check.py SYM_FILE

For each position and symmetry in SYM_FILE: the augmented V8 inputs must equal the C++ inputs of the
image position (fillRowV8 of getSymBoard), and every legal move's policy entry must land on the C++
image move (getSymAction). Exit code 0 when all agree.
"""

import sys

import numpy as np
import torch

from katago.train.enclosure_symmetry import NUM_ACTIONS, SIZE, apply_symmetry_inputs, apply_symmetry_policy

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


def main():
    checked, failures = 0, 0
    per_symmetry = {}
    for label, symmetry, inp, glob, image, moves in blocks(sys.argv[1]):
        x = dense(inp)
        g = torch.tensor([glob])
        got = apply_symmetry_inputs(x, symmetry).reshape(-1).nonzero().flatten().tolist()
        ok_inputs = got == sorted(image)
        actions, images = moves[0::2], moves[1::2]
        policy = torch.zeros(1, 1, NUM_ACTIONS + 1)
        policy[0, 0, actions] = torch.arange(1, len(actions) + 1, dtype=torch.float32)
        mapped = apply_symmetry_policy(policy, symmetry, x, g)[0, 0]
        ok_policy = all(mapped[b].item() == i + 1 for i, b in enumerate(images)) and mapped.sum().item() == policy.sum().item()
        checked += 1
        per_symmetry.setdefault(symmetry, [0, 0])[0] += 1
        if not (ok_inputs and ok_policy):
            failures += 1
            per_symmetry[symmetry][1] += 1
            if failures <= 10:
                print(f"FAIL {label} symmetry {symmetry}: inputs {'ok' if ok_inputs else 'differ'}, "
                      f"policy {'ok' if ok_policy else 'differs'}")
    for s in sorted(per_symmetry):
        print(f"symmetry {s}: {per_symmetry[s][0]} positions, {per_symmetry[s][1]} failures")
    print(f"{checked} checked, {failures} failures")
    return 0 if failures == 0 and checked > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
