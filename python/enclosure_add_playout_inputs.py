#!/usr/bin/python3
"""Enclosure fork: give a training checkpoint the V9 inputs, KataGo's playout doubling advantage.

    python enclosure_add_playout_inputs.py CHECKPOINT

The V9 inputs (2026-10-13; games/enclosure/docs/decisions/inputs-and-targets.md "The playout doubling advantage as inputs") are V8's
with two more global inputs: 1 when the advantage is nonzero, and half the advantage, from the player to move's side
(C++ NNInputs V9, model version 18). Self-play gives one side more visits in some games, and its rows record the
advantage, so the network can learn what the advantage does. The trunk's first global layer (linear_global) reads the two inputs through two more weight columns.

This sets the config's inputs_version to 9 and adds both columns as zeros, in the model and the SWA model, so the
network computes exactly what it did until training moves them. The optimizer keeps every state by position (no
parameter is added): the two columns start at 0 in each state tensor of linear_global's weight (SGD's momentum, Adam's
moments). Saves in place, keeping the original as CHECKPOINT.before-playout. Run only while no trainer uses the
checkpoint.
"""

import copy
import shutil
import sys

import torch

from enclosure_add_regret_heads import group_names
from katago.train import modelconfigs
from katago.train.model_pytorch import Model

KEY = "linear_global.weight"


def main():
    path = sys.argv[1]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    if modelconfigs.get_inputs_version(config) == 9:
        print(f"{path}: already has the V9 inputs")
        return
    new_config = copy.deepcopy(config)
    new_config["inputs_version"] = 9
    old_n = modelconfigs.get_num_global_input_features(config)
    new_n = modelconfigs.get_num_global_input_features(new_config)
    assert (old_n, new_n) == (5, 7), (old_n, new_n)

    def widen(t):
        assert t.dim() == 2 and t.shape[1] == old_n, t.shape
        return torch.cat([t, torch.zeros(t.shape[0], new_n - old_n, dtype=t.dtype, device=t.device)], dim=1)

    touched = []
    for part in ("model", "swa_model"):
        if part not in ckpt:
            continue
        keys = [k for k in ckpt[part] if k.endswith(KEY)]
        assert len(keys) == 1, keys
        ckpt[part][keys[0]] = widen(ckpt[part][keys[0]])
        touched.append(f"{part} ({keys[0]})")

    # The optimizer state of linear_global's weight: its position in train.get_param_groups' order.
    index = None
    for saved, (_, names) in zip(ckpt["optimizer"]["param_groups"], group_names(Model(config, 19))):
        assert len(saved["params"]) == len(names), "the saved optimizer does not match this code's parameter groups"
        for p, n in zip(saved["params"], names):
            if n == KEY:
                index = p
    assert index is not None, f"no {KEY} in the optimizer's groups"
    state = ckpt["optimizer"]["state"].get(index, {})
    widened = []
    for k, v in state.items():
        if torch.is_tensor(v) and v.dim() == 2:
            state[k] = widen(v)
            widened.append(k)
    touched.append(f"optimizer (state {index}: {', '.join(widened) or 'none'})")

    ckpt["config"] = new_config
    shutil.copy2(path, path + ".before-playout")
    torch.save(ckpt, path)
    print(f"{path}: V9 inputs added to " + "; ".join(touched))


if __name__ == "__main__":
    main()
