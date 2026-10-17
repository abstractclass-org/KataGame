#!/usr/bin/python3
"""Enclosure fork: the loss scales of the auxiliary targets (abscls-kcu).

    python enclosure_aux_scales_check.py CHECKPOINT ROWS_NPZ [ROWS_NPZ ...]

On the first batches of the rows, with the checkpoint's model and KataGo's metrics:

- no scales and scales of 1 give the same loss, bit for bit: the default changes nothing;
- with the four scales and -seki-loss-scale at 0, the loss falls by exactly the ownership, scoring (x 0.25), future
  positions, income and seki losses.

Each call gets a fresh Metrics, because the seki loss keeps a moving average between calls. Nothing is trained.
Exit code 0 when both hold on every batch.
"""

import math
import sys

import torch

from katago.train.data_processing_pytorch import read_npz_training_data
from katago.train.load_model import load_model
from katago.train.metrics_pytorch import Metrics

SIZE = 19
BATCHES = 4
OFF = {"ownership": 0.0, "futurepos": 0.0, "income": 0.0, "scoring": 0.0}


def loss(model, outputs, batch, seki, aux):
    m = Metrics(1, model).metrics_dict_batchwise(
        model, model.postprocess_output(outputs), None, batch, is_training=True,
        soft_policy_weight_scale=8.0, disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
        value_loss_scale=0.6, td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=seki,
        variance_time_loss_scale=1.0, main_loss_scale=1.0, intermediate_loss_scale=None,
        include_model_norms=False, **({} if aux is None else {"aux_loss_scales": aux}))
    return {k: float(v) for k, v in m.items() if torch.is_tensor(v) and v.numel() == 1}


def main():
    checkpoint, npz_files = sys.argv[1], sys.argv[2:]
    torch.manual_seed(0)
    model, _, _ = load_model(checkpoint, use_swa=False, device="cpu")
    model.train()
    bad = 0
    batches = read_npz_training_data(npz_files, batch_size=16, world_size=1, rank=0, pos_len=SIZE, device="cpu",
                                     randomize_symmetries=False, include_meta=False, model_config=model.config)
    for i, batch in zip(range(BATCHES), batches):
        with torch.no_grad():
            outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
        default = loss(model, outputs, batch, 1.0, None)
        ones = loss(model, outputs, batch, 1.0, {k: 1.0 for k in OFF})
        off = loss(model, outputs, batch, 0.0, OFF)
        same = default["loss_sum"] == ones["loss_sum"]
        terms = (default["oloss_sum"] + 0.25 * default["sloss_sum"] + default["fploss_sum"] + default["iloss_sum"]
                 + default["skloss_sum"])
        drop = default["loss_sum"] - off["loss_sum"]
        close = math.isclose(drop, terms, rel_tol=1e-5, abs_tol=1e-3)
        print(f"batch {i + 1}: loss {default['loss_sum']:.4f}; scales of 1 give the same: {same}; with the five at 0 "
              f"it falls by {drop:.4f}, their losses sum to {terms:.4f}: {'equal' if close else 'DIFFERENT'}")
        bad += not (same and close)
    print(f"{BATCHES} batches; {bad} failed")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
