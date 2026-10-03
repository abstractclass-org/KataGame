#!/usr/bin/python3
"""Enclosure fork: zero the first layer's weights for some spatial input channels in a training checkpoint.

    python enclosure_zero_input_planes.py CHECKPOINT CHANNEL [CHANNEL ...]

For a slot given a new meaning (2026-10-03: each player's plane 7, held ground, empty under no_maxflow, becomes the
turn-safe plane under ENCLOSURE_TURN_SAFE; V8 channels 200 and 209), so the network starts that input from zero
instead of from weights trained for the old meaning. Zeroes the channels' slices of conv_spatial's weight in the
model and in the SWA model, and of its optimizer state (momentum), then saves in place, keeping a copy of the
original next to it (CHECKPOINT.before-zero). Run only while no trainer is using the checkpoint.
"""

import shutil
import sys

import torch


def main():
    path, channels = sys.argv[1], [int(c) for c in sys.argv[2:]]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    shutil.copy2(path, path + ".before-zero")
    touched = []
    for part in ("model", "swa_model"):
        for key, tensor in ckpt.get(part, {}).items():
            if key.endswith("conv_spatial.weight"):
                tensor[:, channels] = 0
                touched.append(f"{part}.{key} {tuple(tensor.shape)}")
    shape = next(t.shape for k, t in ckpt["model"].items() if k.endswith("conv_spatial.weight"))
    for state in ckpt["optimizer"]["state"].values():
        for name, tensor in state.items():
            if torch.is_tensor(tensor) and tensor.shape == shape:
                tensor[:, channels] = 0
                touched.append(f"optimizer {name} {tuple(tensor.shape)}")
    torch.save(ckpt, path)
    print(f"{path}: zeroed input channels {channels} in " + "; ".join(touched))


if __name__ == "__main__":
    main()
