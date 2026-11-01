#!/usr/bin/python3
"""KataGame fork (abscls-ra93.2): a checkpoint's losses on fixed rows, a holdout, so that networks which trained on the
same rows compare on rows that none of them trained on.

    python holdout_losses.py -checkpoint CKPT -npz FILE [FILE ...] [-batch-size 128] [-device cpu] [-seed 0]
        [-max-batches N] [-json OUT]

For the raw weights and the SWA weights (an export plays the SWA weights), it gives one line each and, with -json, one
JSON object for both:

- Each of the trainer's losses (metrics_pytorch, at train.py's default loss scales) as its mean over the rows that hold
  its target, as enclosure_checkpoint_losses.py divides them: p0loss over the weight C26, p1loss over C28, vloss and
  the score's losses over the rows, oloss over C27, the spatial targets' losses over their own weights (fploss C33,
  sloss C34, skloss C27 x C68, iloss C27 x C69), and the intrinsic value head's intaloss and intbloss over C81 and
  C83. The trainer's logs divide each loss by all rows, so a loss of a target that few rows hold reads low there.
- pacc1: the share of the rows with a policy target (C26) whose policy's top move is the target's top move.
- illegal: the policy's mean mass on the moves that the row's inputs do not make legal (games.legal_moves), over the
  rows with a policy target. The search never plays such a move, so a network that puts mass there misreads the
  position's local facts. Only for a game that gives legal_moves.

The rows are read as the trainer reads them (data_processing_pytorch.read_npz_training_data), without symmetries, the
files in the given order, with the history dropout drawn from -seed. So two checkpoints evaluated with the same files,
batch size and seed see the same inputs. Each file gives its whole batches only (the reader's rule). The forward runs
in fp32, under no gradient. Run it from katagame/python.
"""
import argparse
import json
import math
import sys

import numpy as np
import torch

from katago.train import data_processing_pytorch, games, modelconfigs
from katago.train.load_model import load_model
from katago.train.metrics_pytorch import Metrics

# Each loss and the sum that holds its rows' weight, over which its mean is taken ("wsum": the rows' weight C25).
LOSSES = {
    "p0loss": "p0w", "p1loss": "p1w", "vloss": "wsum", "tdvloss1": "wsum", "tdvloss2": "wsum", "tdvloss3": "wsum",
    "oloss": "ow", "sloss": "sw_sum", "fploss": "fpw_sum", "skloss": "skw_sum", "iloss": "iw_sum", "smloss": "ow",
    "leadloss": "leadw", "intaloss": "intarows_sum", "intbloss": "intbrows_sum", "loss": "wsum",
}


def validation_metrics(metrics_obj, network, outputs, batch):
    """The trainer's validation metrics of a batch at train.py's default loss scales (export_gate_check.py's)."""
    return metrics_obj.metrics_dict_batchwise(
        network, network.postprocess_output(outputs), None, batch, is_training=False, soft_policy_weight_scale=8.0,
        disable_optimistic_policy=False, meta_kata_only_soft_policy=False, value_loss_scale=0.6,
        td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=1.0, variance_time_loss_scale=1.0,
        main_loss_scale=None, intermediate_loss_scale=None, include_model_norms=False)


def batch_weights(batch):
    """The rows' weights that the metrics do not sum: each policy target's (C26, C28, as the loss's weights, without
    the rows of a target of sum 0), the ownership's (C27) and the lead's (C29), times the row's weight (C25)."""
    targets = batch["globalTargetsNC"]
    policy = batch["policyTargetsNCMove"]
    weight = targets[:, 25]
    has = policy.sum(dim=2) > 0
    return {"p0w": (weight * targets[:, 26] * has[:, 0]).sum(), "p1w": (weight * targets[:, 28] * has[:, 1]).sum(),
            "ow": (weight * targets[:, 27]).sum(), "leadw": (weight * targets[:, 29]).sum()}


def illegal_mass(game, network_outputs, batch):
    """The sum over the batch's rows with a policy target of the policy's mass outside the legal moves, and those rows'
    weight. None for a game without legal_moves."""
    legal = games.legal_moves(game, batch["binaryInputNCHW"], batch["globalInputNC"])
    if legal is None:
        return None
    policy_logits = network_outputs[0][0][:, 0, :].float()
    probs = torch.softmax(policy_logits, dim=1)
    outside = (probs * ~legal).sum(dim=1)
    weights = batch_weights(batch)
    targets = batch["globalTargetsNC"]
    row = targets[:, 25] * targets[:, 26] * (batch["policyTargetsNCMove"][:, 0].sum(dim=1) > 0)
    return (outside * row).sum(), weights["p0w"]


def evaluate(networks, npz_files, batch_size, device, seed, max_batches=None):
    """{label: {measure: value}} for each network of networks ({label: model}), on the same batches."""
    model = next(iter(networks.values()))
    config = model.config
    game = modelconfigs.get_game(config)
    metrics_obj = Metrics(1, model)
    sums = {label: {} for label in networks}
    torch.manual_seed(seed)
    np.random.seed(seed)
    rows = 0
    for i, batch in enumerate(data_processing_pytorch.read_npz_training_data(
            list(npz_files), batch_size, world_size=1, rank=0, pos_len=model.pos_len, device=device,
            randomize_symmetries=False, include_meta=False, model_config=config)):
        if max_batches is not None and i >= max_batches:
            break
        rows += batch["binaryInputNCHW"].shape[0]
        extra = {k: float(v) for k, v in batch_weights(batch).items()}
        for label, network in networks.items():
            outputs = network(batch["binaryInputNCHW"], batch["globalInputNC"])
            metrics = validation_metrics(metrics_obj, network, outputs, batch)
            add = {k: float(v) for k, v in metrics.items() if torch.is_tensor(v) or isinstance(v, (int, float))}
            add.update(extra)
            mass = illegal_mass(game, outputs, batch)
            if mass is not None:
                add["illegal_sum"], add["illegal_w"] = float(mass[0]), float(mass[1])
            for k, v in add.items():
                sums[label][k] = sums[label].get(k, 0.0) + v
    return {label: measures(s, rows) for label, s in sums.items()}


def measures(s, rows):
    """The means of one network's sums (evaluate), over the rows of each measure."""
    def mean(total, weight):
        return s[total] / s[weight] if s.get(weight, 0.0) > 0 else math.nan
    out = {"rows": rows}
    for loss, weight in LOSSES.items():
        if loss + "_sum" in s:
            out[loss] = mean(loss + "_sum", weight)
    out["pacc1"] = mean("pacc1_sum", "p0w")
    if "illegal_sum" in s:
        out["illegal"] = mean("illegal_sum", "illegal_w")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-checkpoint", required=True, help="a trainer's checkpoint, with its SWA weights")
    ap.add_argument("-npz", nargs="+", required=True, help="the rows, read in this order")
    ap.add_argument("-batch-size", type=int, default=128)
    ap.add_argument("-device", default="cpu")
    ap.add_argument("-seed", type=int, default=0, help="the seed of the history dropout (default 0)")
    ap.add_argument("-max-batches", type=int, default=None)
    ap.add_argument("-json", help="a file for the measures, one JSON object")
    args = ap.parse_args()
    raw, swa, other = load_model(args.checkpoint, use_swa=True, device=args.device)
    networks = {"raw": raw.eval(), "swa": swa.eval()}
    result = evaluate(networks, args.npz, args.batch_size, args.device, args.seed, args.max_batches)
    step = other.get("train_state", {}).get("global_step_samples") if isinstance(other, dict) else None
    for label, m in result.items():
        print(f"{label}: " + " ".join(f"{k} {v:.6g}" for k, v in m.items()), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "global_step_samples": step, **result}, f, indent=1)
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
