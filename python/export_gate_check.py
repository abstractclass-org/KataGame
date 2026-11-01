#!/usr/bin/python3
"""KataGame fork (abscls-8mbo): the trainer's export gate (katago/train/export_gate.py) on two checkpoints.

    python export_gate_check.py -reference CKPT -candidate CKPT -npz ROWS.npz [...] -margin M
        [-max-batches N] [-chunk-rows R] [-device cpu]

It runs the gate's code on rows of the caller's choice: the reference's and the candidate's raw and SWA weights on the
same batches, with the same symmetries, each batch in parts of at most R rows. It prints each pair's p0loss and p1loss
and the verdict: "export", or "refused" with the exit code 1. A checkpoint is a trainer's (train/NAME/checkpoint*.ckpt)
or an export's (models/NAME/model.ckpt), each with its SWA weights. Run it from katagame/python. On the CPU with 2
threads, one 128-row batch of Blueprints' board (67 x 67) took about 11 s for each of the 4 networks of a b6c96, and
the process held at most 5.0 GB, mostly the rows of one file (2026-11-01, bpj5's checkpoints).
"""
import argparse
import sys

import torch

from katago.train import export_gate
from katago.train.data_processing_pytorch import read_npz_training_data
from katago.train.load_model import load_model
from katago.train.metrics_pytorch import Metrics


def validation_metrics(metrics_obj):
    """metrics(network, outputs, part) for GateValidation: the trainer's validation metrics at train.py's default loss
    scales."""
    def metrics(network, outputs, part):
        return metrics_obj.metrics_dict_batchwise(
            network, network.postprocess_output(outputs), None, part, is_training=False, soft_policy_weight_scale=8.0,
            disable_optimistic_policy=False, meta_kata_only_soft_policy=False, value_loss_scale=0.6,
            td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=1.0, variance_time_loss_scale=1.0,
            main_loss_scale=None, intermediate_loss_scale=None, include_model_norms=False)
    return metrics


def batches(npz_files, batch_size, pos_len, device, model_config):
    # One file a call: within a call the reader keeps one more file in memory.
    for path in npz_files:
        yield from read_npz_training_data([path], batch_size, world_size=1, rank=0, pos_len=pos_len, device=device,
                                          randomize_symmetries=True, include_meta=False, model_config=model_config)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-reference", required=True, help="the last exported network's checkpoint")
    ap.add_argument("-candidate", required=True, help="the checkpoint that would export")
    ap.add_argument("-npz", nargs="+", required=True, help="the rows: validation files of the run")
    ap.add_argument("-margin", type=float, required=True, help="the trainer's -export-gate-margin")
    ap.add_argument("-batch-size", type=int, default=128)
    ap.add_argument("-chunk-rows", type=int, default=32, help="rows per forward (default 32, the trainer's batch / 4)")
    ap.add_argument("-max-batches", type=int, default=None)
    ap.add_argument("-device", default="cpu")
    args = ap.parse_args()

    networks = {}
    for label, suffix, path in (("candidate", "", args.candidate), ("reference", "_reference", args.reference)):
        model, swa, other = load_model(path, use_swa=True, device=args.device)
        step = other.get("train_state", {}).get("global_step_samples")
        print(f"{label}: {path} ({step} samples)", flush=True)
        networks["raw" + suffix] = model
        networks["swa" + suffix] = swa
    model = networks["raw"]

    def forward(network, part):
        return network(part["binaryInputNCHW"], part["globalInputNC"])

    gate = export_gate.GateValidation(networks, forward, validation_metrics(Metrics(1, model)), args.chunk_rows)
    for i, batch in enumerate(batches(args.npz, args.batch_size, model.pos_len, args.device, model.config)):
        if args.max_batches is not None and i >= args.max_batches:
            break
        gate.evaluate(batch)
        print(f"batch {i + 1}: {gate.sums['raw']['rows']} rows", flush=True)
    refused, lines = export_gate.compare(gate.losses(), args.margin)
    for line in lines:
        print(line)
    print(f"refused (margin {args.margin:g})" if refused else f"export (margin {args.margin:g})")
    return 1 if refused else 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
