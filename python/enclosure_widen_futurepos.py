#!/usr/bin/python3
"""Enclosure fork: widen a training checkpoint's future-position head from 2 outputs to 4.

    python enclosure_widen_futurepos.py CHECKPOINT

The territory-income target (owner, 2026-10-03) uses the future-position head's channels 2-3 (model_pytorch,
metrics_pytorch). For a checkpoint trained before, this keeps channels 0-1 and initializes 2-3 as a new model
would (KataGo's init for that layer), in the model and the SWA model, and gives the new rows zero optimizer
state (momentum). Saves in place, keeping the original as CHECKPOINT.before-income. Run only while no trainer
uses the checkpoint.
"""

import shutil
import sys

import torch

from katago.train.model_pytorch import Model

KEY = "value_head.conv_futurepos.weight"


def main():
    path = sys.argv[1]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    fresh = Model(ckpt["config"], 19).state_dict()[KEY]  # (4, c_in, 1, 1), KataGo's initialization
    touched = []
    old_shape = None
    for part in ("model", "swa_model"):
        for key in list(ckpt.get(part, {})):
            if key.endswith(KEY):
                old = ckpt[part][key]
                if old.shape[0] == 4:
                    print(f"{path}: {part}.{key} already has 4 outputs")
                    return
                assert old.shape[0] == 2 and old.shape[1:] == fresh.shape[1:], (key, old.shape, fresh.shape)
                old_shape = old.shape
                new = fresh.clone().to(old.dtype)
                new[:2] = old
                ckpt[part][key] = new
                touched.append(f"{part}.{key}")
    assert old_shape is not None, "no future-position head found"
    states = [s for s in ckpt["optimizer"]["state"].values()
              if any(torch.is_tensor(t) and t.shape == old_shape for t in s.values())]
    assert len(states) == 1, f"{len(states)} optimizer states of shape {tuple(old_shape)}; expected exactly 1"
    for name, t in list(states[0].items()):
        if torch.is_tensor(t) and t.shape == old_shape:
            padded = torch.zeros((4, *old_shape[1:]), dtype=t.dtype)
            padded[:2] = t
            states[0][name] = padded
            touched.append(f"optimizer {name}")
    shutil.copy2(path, path + ".before-income")
    torch.save(ckpt, path)
    print(f"{path}: widened to 4 outputs in " + "; ".join(touched))


if __name__ == "__main__":
    main()
