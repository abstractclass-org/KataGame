#!/usr/bin/python3
"""Enclosure fork: several checkpoints' losses on the same training rows, per row that carries each target.

    python enclosure_checkpoint_losses.py --npz ROWS.npz [...] --checkpoints CKPT [CKPT ...] [--batches 40]

The trainer reports each loss as its sum over a batch divided by the batch's rows (metrics_logging.log_metrics),
so a target that only some rows carry (the Enclosure spatial targets, flagged since 2026-10-03) reads lower the
fewer rows carry it, and rises as flagged rows fill the window even while the network improves. This divides
each loss by the summed weight of the rows that carry its target instead, on fixed rows, so checkpoints compare
directly: a later checkpoint should score lower. Checkpoints from before the territory-income widening get their
future-position head padded to 4 outputs (their income loss is then meaningless).
"""

import argparse

import torch

from katago.train.data_processing_pytorch import read_npz_training_data
from katago.train.metrics_pytorch import Metrics
from katago.train.model_pytorch import Model

SIZE = 19
FUTUREPOS = "value_head.conv_futurepos.weight"

# loss key -> the per-row weight of its target, from globalTargetsNC (metrics_pytorch's weights).
WEIGHTS = {
    "fploss_sum": lambda g: g[:, 33],                # future positions (C33)
    "sloss_sum": lambda g: g[:, 34],                 # occupancy (KataGo's scoring target, C34)
    "skloss_sum": lambda g: g[:, 27] * g[:, 68],     # cut edges (ownership weight x C68)
    "iloss_sum": lambda g: g[:, 27] * g[:, 69],      # territory income (ownership weight x C69)
    "oloss_sum": lambda g: g[:, 27],                 # ownership
    "smloss_sum": lambda g: g[:, 27],                # score mean
    "sbpdfloss_sum": lambda g: g[:, 27],             # score belief
    "sbcdfloss_sum": lambda g: g[:, 27],
    "vloss_sum": lambda g: torch.ones_like(g[:, 0]),
    "p0loss_sum": lambda g: g[:, 26],
}


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Model(ckpt["config"], SIZE)
    sd = ckpt["model"]
    prefix = [k for k in sd if k.endswith("value_head.conv1.weight")][0][:-len("value_head.conv1.weight")]
    sd = {k[len(prefix):]: v for k, v in sd.items()}
    padded = False
    if sd[FUTUREPOS].shape[0] == 2:
        w = torch.zeros((4, *sd[FUTUREPOS].shape[1:]), dtype=sd[FUTUREPOS].dtype)
        w[:2] = sd[FUTUREPOS]
        sd[FUTUREPOS] = w
        padded = True
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, padded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--batches", type=int, default=40)
    args = ap.parse_args()
    print("checkpoint " + " ".join(k[:-4] for k in WEIGHTS) + "  (per row carrying the target)")
    for path in args.checkpoints:
        model, padded = load(path)
        metrics_obj = Metrics(1, model)
        sums = {k: 0.0 for k in WEIGHTS}
        weights = {k: 0.0 for k in WEIGHTS}
        torch.manual_seed(0)
        batches = read_npz_training_data(args.npz, batch_size=128, world_size=1, rank=0, pos_len=SIZE,
                                         device="cpu", randomize_symmetries=False, include_meta=False,
                                         model_config=model.config)
        with torch.no_grad():
            for i, batch in enumerate(batches):
                if i >= args.batches:
                    break
                outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
                m = metrics_obj.metrics_dict_batchwise(
                    model, model.postprocess_output(outputs), None, batch, is_training=False,
                    soft_policy_weight_scale=8.0, disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
                    value_loss_scale=0.6, td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=1.0,
                    variance_time_loss_scale=1.0, main_loss_scale=1.0, intermediate_loss_scale=None,
                )
                g = batch["globalTargetsNC"]
                for k, wf in WEIGHTS.items():
                    sums[k] += float(m[k])
                    weights[k] += float(wf(g).sum())
        cells = []
        for k in WEIGHTS:
            if k == "iloss_sum" and padded:
                cells.append("-")
            else:
                cells.append(f"{sums[k] / max(weights[k], 1e-9):.4f}")
        print(path.rsplit("/", 1)[-1], " ".join(cells))
    print("row shares: " + " ".join(f"{k[:-4]}={weights[k] / (args.batches * 128):.3f}" for k in WEIGHTS))


if __name__ == "__main__":
    main()
