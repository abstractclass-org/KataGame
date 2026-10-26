#!/usr/bin/python3
"""KataGame fork: give a training checkpoint the inputs version 10, its game's extended inputs.

    python add_extended_inputs.py CHECKPOINT OUT

The inputs version 10 (2026-10-26; C++ NNInputs V10, model version 19) is version 9's with the game's extended inputs:
its extended planes after its planes, and its extended global inputs after version 9's (katago/train/games:
NUM_EXTENDED_SPATIAL_INPUTS and NUM_EXTENDED_GLOBAL_INPUTS). Blueprints' run bpj3 starts from a network of bpj2 this
way (the engine repository's games/blueprints/docs/decisions/port.md). The trunk's first convolution (conv_spatial)
reads the new planes through more input channels, and its first global layer (linear_global) reads the new global
inputs through more weight columns.

This sets the config's inputs_version to 10 and adds those channels and columns as zeros, after the old ones, in the
model, the SWA model and the optimizer's state, so that the network computes exactly what it did until training moves
them, whatever the new inputs hold. The optimizer keeps every state by position (no parameter is added): the new
channels and columns start at 0 in each state tensor of the two weights (SGD's momentum, Adam's moments). A
checkpoint without an optimizer or an SWA model (a network alone) gets the rest. The checkpoint's inputs must be
version 9. Writes OUT, which must not exist: the checkpoint stays as it was. Run it on a checkpoint that no trainer
writes at the time (a copy).
"""

import copy
import os
import sys

import torch

from enclosure_add_regret_heads import group_names
from katago.train import games, modelconfigs
from katago.train.model_pytorch import Model


def widen(t, old_n, new_n):
    """t with new_n - old_n zeros after its old_n entries on dimension 1."""
    assert t.dim() >= 2 and t.shape[1] == old_n, (tuple(t.shape), old_n)
    shape = list(t.shape)
    shape[1] = new_n - old_n
    return torch.cat([t, torch.zeros(shape, dtype=t.dtype, device=t.device)], dim=1)


def convert(ckpt):
    """ckpt (a loaded checkpoint) with the inputs version 10, in place. Returns what it widened, one line each."""
    config = ckpt["config"]
    if modelconfigs.get_inputs_version(config) != 9:
        raise ValueError(f"the checkpoint's inputs are version {modelconfigs.get_inputs_version(config)}, not 9")
    game = modelconfigs.get_game(config)
    if games.extended_spatial_inputs(game) == 0 and games.extended_global_inputs(game) == 0:
        raise ValueError(f"the game {game.NAME} has no extended inputs")
    new_config = copy.deepcopy(config)
    new_config["inputs_version"] = 10
    # The weights that read the inputs, on dimension 1: conv_spatial (out, in, y, x) and linear_global (out, in).
    sizes = {
        "conv_spatial.weight": (modelconfigs.get_num_bin_input_features(config),
                                modelconfigs.get_num_bin_input_features(new_config)),
        "linear_global.weight": (modelconfigs.get_num_global_input_features(config),
                                 modelconfigs.get_num_global_input_features(new_config)),
    }

    touched = []
    for part in ("model", "swa_model"):
        if part not in ckpt:
            continue
        for key, (old_n, new_n) in sizes.items():
            # The weight, after a wrapper's prefix ("_orig_mod." of a compiled model, "module." of the SWA model).
            names = [k for k in ckpt[part] if k == key or k.endswith("." + key)]
            assert len(names) == 1, (part, key, names)
            ckpt[part][names[0]] = widen(ckpt[part][names[0]], old_n, new_n)
            touched.append(f"{part} {names[0]}: {old_n} -> {new_n}")

    ckpt["config"] = new_config
    if "optimizer" not in ckpt:  # a checkpoint of a network alone
        return touched
    # The optimizer's state of each weight: its position in train.get_param_groups' order.
    indices = {}
    for saved, (_, names) in zip(ckpt["optimizer"]["param_groups"], group_names(Model(config, game.MAX_BOARD_LEN))):
        assert len(saved["params"]) == len(names), "the saved optimizer does not match this code's parameter groups"
        for p, n in zip(saved["params"], names):
            if n in sizes:
                indices[n] = p
    assert set(indices) == set(sizes), f"the optimizer's groups lack {set(sizes) - set(indices)}"
    for key, index in indices.items():
        old_n, new_n = sizes[key]
        state = ckpt["optimizer"]["state"].get(index, {})
        widened = []
        for k, v in state.items():
            if torch.is_tensor(v) and v.dim() >= 2:
                state[k] = widen(v, old_n, new_n)
                widened.append(k)
        touched.append(f"optimizer {key} (state {index}: {', '.join(widened) or 'none'})")
    return touched


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    path, out = sys.argv[1], sys.argv[2]
    if os.path.exists(out):
        sys.exit(f"{out} exists: give a new path")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    touched = convert(ckpt)
    torch.save(ckpt, out)
    print(f"{out}: {path} with the inputs version 10\n  " + "\n  ".join(touched))


if __name__ == "__main__":
    main()
