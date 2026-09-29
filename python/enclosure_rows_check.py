#!/usr/bin/python3
"""Enclosure fork: training rows through KataGo's trainer input and losses (plan step 6).

    python enclosure_rows_check.py CHECKPOINT ROWS_NPZ [ROWS_NPZ ...]

The rows come from `katago runenclosurerowtests` via the Enclosure repository's tools/check_katago_rows.py
--npz (which also checks their values against the Enclosure engine). They are read by read_npz_training_data
(random symmetries and history dropout on), run through the model, and scored by KataGo's metrics; then one
backward pass. Nothing is trained: no optimizer step. Exit code 0 when every batch loads with the expected shapes
and every loss and gradient is finite.
"""

import math
import sys

import torch

from katago.train.data_processing_pytorch import read_npz_training_data
from katago.train.load_model import load_model
from katago.train.metrics_pytorch import Metrics
from katago.train.model_pytorch import policy_len

SIZE = 19


def main():
    checkpoint, npz_files = sys.argv[1], sys.argv[2:]
    torch.manual_seed(0)
    model, _, _ = load_model(checkpoint, use_swa=False, device="cpu")
    model.train()
    metrics_obj = Metrics(1, model)
    batches, bad = 0, 0
    for batch in read_npz_training_data(npz_files, batch_size=16, world_size=1, rank=0, pos_len=SIZE, device="cpu",
                                        randomize_symmetries=True, include_meta=False, model_config=model.config):
        n = batch["binaryInputNCHW"].shape[0]
        assert batch["binaryInputNCHW"].shape == (n, 215, SIZE, SIZE)
        assert batch["globalInputNC"].shape == (n, 5)
        assert batch["policyTargetsNCMove"].shape == (n, 2, policy_len(SIZE))
        assert batch["scoreDistrN"].shape == (n, 842)
        assert batch["valueTargetsNCHW"].shape == (n, 5, SIZE, SIZE)
        model.zero_grad()
        outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
        metrics = metrics_obj.metrics_dict_batchwise(
            model, model.postprocess_output(outputs), None, batch, is_training=True,
            soft_policy_weight_scale=8.0, disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
            value_loss_scale=0.6, td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=1.0,
            variance_time_loss_scale=1.0, main_loss_scale=1.0, intermediate_loss_scale=None,
        )
        loss = metrics["loss_sum"]
        loss.backward()
        grads_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        values = {k: float(v) for k, v in metrics.items() if torch.is_tensor(v) and v.numel() == 1}
        finite = all(math.isfinite(v) for v in values.values())
        if batches == 0:
            print("losses of the first batch (sums over its rows):")
            for k in sorted(values):
                if k.endswith("_sum") or k.startswith("p0") or k.startswith("v"):
                    print(f"  {k} {values[k]:.4g}")
        batches += 1
        if not (finite and grads_finite):
            bad += 1
            print(f"batch {batches}: finite losses {finite}, finite gradients {grads_finite}")
    print(f"{batches} batches of 16 rows; {bad} with a non-finite loss or gradient")
    return 0 if batches > 0 and bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
