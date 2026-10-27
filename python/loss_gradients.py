#!/usr/bin/python3
"""KataGame fork: the gradient that each of the trainer's losses gives the shared trunk, on one batch of rows.

    python loss_gradients.py -checkpoint CKPT -npz ROWS.npz [-batch-size 64] [-threads 2]

The losses are the terms of metrics_pytorch's loss_sum, in groups, each with the scale that the trainer gives it (the
trainer's defaults of -soft-policy-weight-scale, -value-loss-scale, -td-value-loss-scales, -seki-loss-scale and
-variance-time-loss-scale). For each group the tool prints its loss per row and the norm of the gradient of that loss
(its mean over the batch's rows) with respect to the trunk's parameters (every parameter outside the policy and value
heads) and to the trunk's final activation, which every head reads. The ratio to the player's policy shows whether a
head dominates the trunk or gives it almost nothing. The groups add up to loss_sum, and the tool checks that. Only
the main heads' losses count: a model's intermediate heads, if it has them, are left out.

The batch is the file's first, without symmetries and with the history dropout of a fixed seed, so two runs of the
tool on two versions of the trainer's code read the same rows. The model runs on the CPU in training mode (its norms
read the batch as in training), and nothing is stepped or saved.
"""

import argparse
import math

import torch

from katago.train.data_processing_pytorch import read_npz_training_data
from katago.train.load_model import load_model
from katago.train.metrics_pytorch import Metrics

HEADS = ("policy_head.", "value_head.", "intermediate_policy_head.", "intermediate_value_head.")


def groups(version, scales):
    """The loss groups: (name, [(key of the metrics' results, its scale in loss_sum)]), as metrics_pytorch sums
    them with the trainer's scales `scales` (the metrics' keyword arguments)."""
    soft, td = scales["soft_policy_weight_scale"], scales["td_value_loss_scales"]
    # metrics_pytorch's policy_opt_loss_scale, long_policy_opt_loss_scale and short_policy_opt_loss_scale
    opt, long_opt, short_opt = (1.0, 0.0, 0.0) if version <= 11 else (0.93, 0.1, 0.2)
    return [
        ("policy", [("p0loss_sum", opt), ("p0softloss_sum", soft), ("p0lopt_sum", long_opt),
                    ("p0sopt_sum", short_opt)]),
        ("opponent's policy", [("p1loss_sum", 1.0), ("p1softloss_sum", soft)]),
        ("value", [("vloss_sum", scales["value_loss_scale"])]),
        ("short-term values", [("tdvloss1_sum", td[0]), ("tdvloss2_sum", td[1]), ("tdvloss3_sum", td[2])]),
        ("value error", [("evstloss_sum", 1.0)]),
        ("scores", [("tdsloss_sum", 1.0), ("smloss_sum", 1.0), ("sbcdfloss_sum", 1.0), ("sbpdfloss_sum", 1.0),
                    ("sdregloss_sum", 1.0), ("leadloss_sum", 1.0), ("esstloss_sum", 1.0), ("qscloss_sum", 1.0)]),
        ("variance time", [("vtimeloss_sum", scales["variance_time_loss_scale"])]),
        ("q-value win-loss", [("qwlloss_sum", 1.0)]),
        ("C0 ownership", [("oloss_sum", 1.0)]),
        ("C1 seki slot", [("skloss_sum", scales["seki_loss_scale"])]),
        ("C2-3 future positions", [("fploss_sum", 1.0)]),
        ("C4 scoring slot", [("sloss_sum", 0.25)]),
        ("C5-6 income slot", [("iloss_sum", 1.0)]),
        ("regret", [("rgloss_sum", 1.0), ("rkloss_sum", 1.0), ("swloss_sum", 1.0), ("swkloss_sum", 1.0)]),
    ]


def norm(tensors):
    return math.sqrt(sum(float(torch.sum(torch.square(t))) for t in tensors if t is not None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-checkpoint", required=True)
    ap.add_argument("-npz", required=True)
    ap.add_argument("-batch-size", type=int, default=64)
    ap.add_argument("-threads", type=int, default=2)
    ap.add_argument("-seed", type=int, default=0)
    ap.add_argument("-soft-policy-weight-scale", type=float, default=8.0)
    ap.add_argument("-value-loss-scale", type=float, default=0.6)
    ap.add_argument("-td-value-loss-scales", type=str, default="0.6,0.6,0.6")
    ap.add_argument("-seki-loss-scale", type=float, default=1.0)
    ap.add_argument("-variance-time-loss-scale", type=float, default=1.0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    scales = dict(
        soft_policy_weight_scale=args.soft_policy_weight_scale, value_loss_scale=args.value_loss_scale,
        td_value_loss_scales=[float(x) for x in args.td_value_loss_scales.split(",")],
        seki_loss_scale=args.seki_loss_scale, variance_time_loss_scale=args.variance_time_loss_scale,
    )

    model, _, _ = load_model(args.checkpoint, use_swa=False, device="cpu")
    model.train()
    metrics = Metrics(1, model)
    torch.manual_seed(args.seed)
    batch = next(read_npz_training_data(
        [args.npz], batch_size=args.batch_size, world_size=1, rank=0, pos_len=model.pos_len, device="cpu",
        randomize_symmetries=False, include_meta=False, model_config=model.config,
    ))
    rows = batch["globalTargetsNC"].shape[0]

    trunk = [(name, p) for name, p in model.named_parameters() if p.requires_grad and not name.startswith(HEADS)]
    captured = {}
    model.act_trunkfinal.register_forward_hook(lambda module, inputs, output: captured.__setitem__("trunk", output))
    outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
    # The main heads' losses (output 0). is_intermediate=True only leaves out the statistics that follow the losses.
    results = metrics.metrics_dict_batchwise_single_heads_output(
        model, model.postprocess_output(outputs)[0], batch, is_training=False, disable_optimistic_policy=False,
        meta_kata_only_soft_policy=False, is_intermediate=True, **scales,
    )
    terms = [(name, sum(results[k] * scale for k, scale in parts))
             for name, parts in groups(model.config["version"], scales)]
    total = sum(t for _, t in terms)
    assert torch.isclose(total, results["loss_sum"], rtol=1e-4, atol=1e-4), (float(total), float(results["loss_sum"]))

    print(f"{args.checkpoint}: {rows} rows, {sum(p.numel() for _, p in trunk)} trunk parameters "
          f"in {len(trunk)} tensors")
    print(f"{'loss':<24} {'per row':>10} {'trunk params':>13} {'ratio':>8} {'trunk output':>13} {'ratio':>8}")
    measured = []
    for name, term in terms + [("all", total)]:
        loss = float(term.detach()) / rows
        if not term.requires_grad:
            measured.append((name, loss, 0.0, 0.0))
            continue
        grads = torch.autograd.grad(term / rows, [captured["trunk"]] + [p for _, p in trunk], retain_graph=True,
                                    allow_unused=True)
        measured.append((name, loss, norm(grads[1:]), norm(grads[:1])))
    policy = measured[0]
    for name, loss, params, output in measured:
        print(f"{name:<24} {loss:>10.5f} {params:>13.5g} {params / policy[2]:>8.4f} {output:>13.5g} "
              f"{output / policy[3]:>8.4f}")


if __name__ == "__main__":
    main()
