#!/usr/bin/python3
"""KataGame fork: the gradient that each of the trainer's losses gives the shared trunk, on batches of rows.

    python loss_gradients.py -checkpoint CKPT -npz ROWS.npz [ROWS.npz ...] [-batches 1] [-batch-size 64] [-threads 2]
                             [-loss-coefficients NAME=VALUE,...] [-shared-gradient-scales NAME=VALUE,...]

The losses are the terms of metrics_pytorch's loss_sum, in groups, each with the scale that the trainer gives it: the
trainer's defaults of -soft-policy-weight-scale, -value-loss-scale, -td-value-loss-scales, -seki-loss-scale and
-variance-time-loss-scale, and the game's loss coefficients (katago/train/games: LOSS_COEFFICIENTS).
-loss-coefficients sets some of the coefficients for the run, for example scoring=0.25,futurepos=0.25 (KataGo's) to
compare a game's own coefficients with them. The value head's shared gradient scales are the game's
(games.SHARED_GRADIENT_SCALES), or those that -shared-gradient-scales sets (for example scoring=1,futurepos=1). The
gradients go through them, so each group's line is its pull under them. The groups add up to loss_sum, and the tool
checks that. Only the main heads' losses count: a model's intermediate heads, if it has them, are left out.

For each batch, the tool prints a line for each group:
- per row: its loss per row, as loss_sum adds it (with its coefficient);
- trunk params, ratio: the norm of the gradient of its loss (the mean over the batch's rows) with respect to the trunk's
  parameters (every parameter outside the policy and value heads), and its ratio to the player's policy's;
- trunk output, ratio: the same with respect to the trunk's final activation, which every head reads;
- cos policy, cos value: the cosine of its gradient on the trunk's parameters with the player's policy's and the
  value's;
- trunk step, head step: the trainer's step for this gradient, if it held from step to step, over the norm of the
  parameters that it moves: the trunk's, and the group's own head parameters' (the head weights that no other group's
  loss reaches, "-" for a group without them). The step is SGD's with momentum at its limit: the per-sample rate
  (-lr) times the parameter group's factor (train.py: -head-lr-factor for the heads, -noreg-lr-factor for biases and
  norms) times the batch's rows, over 1 - momentum. Lookahead's slow weights move at this rate: its fast weights take
  steps 1 / alpha as large, and the slow weights take alpha of their way.
With several batches, the tool then prints the mean of each column over the batches.

The batches are the files' first ones, without symmetries and with the history dropout of a fixed seed, so two runs
of the tool on two versions of the trainer's code read the same rows. The model runs on the CPU in training mode (its
norms read the batch as in training), and nothing is stepped or saved.
"""

import argparse
import itertools
import math

import torch

from katago.train import games
from katago.train.data_processing_pytorch import read_npz_training_data
from katago.train.load_model import load_model
from katago.train.metrics_pytorch import Metrics

HEADS = ("policy_head.", "value_head.", "intermediate_policy_head.", "intermediate_value_head.")
COLUMNS = ("per row", "trunk params", "ratio", "trunk output", "ratio", "cos policy", "cos value", "trunk step",
           "head step")


def groups(version, scales, coefficients):
    """The loss groups: (name, [(key of the metrics' results, its scale in loss_sum)]), as metrics_pytorch sums
    them with the trainer's scales `scales` (the metrics' keyword arguments) and the loss coefficients `coefficients`
    (Metrics.loss_coefficients)."""
    soft, td, value = scales["soft_policy_weight_scale"], scales["td_value_loss_scales"], scales["value_loss_scale"]
    # metrics_pytorch's policy_opt_loss_scale, long_policy_opt_loss_scale and short_policy_opt_loss_scale
    opt, long_opt, short_opt = (1.0, 0.0, 0.0) if version <= 11 else (0.93, 0.1, 0.2)
    return [
        ("policy", [("p0loss_sum", opt), ("p0softloss_sum", soft), ("p0lopt_sum", long_opt),
                    ("p0sopt_sum", short_opt)]),
        ("opponent's policy", [("p1loss_sum", 1.0), ("p1softloss_sum", soft)]),
        ("value", [("vloss_sum", value)]),
        ("short-term values", [("tdvloss1_sum", td[0]), ("tdvloss2_sum", td[1]), ("tdvloss3_sum", td[2])]),
        ("value error", [("evstloss_sum", 1.0)]),
        ("scores", [("tdsloss_sum", 1.0), ("smloss_sum", 1.0), ("sbcdfloss_sum", 1.0), ("sbpdfloss_sum", 1.0),
                    ("sdregloss_sum", 1.0), ("leadloss_sum", 1.0), ("esstloss_sum", 1.0), ("qscloss_sum", 1.0)]),
        ("variance time", [("vtimeloss_sum", scales["variance_time_loss_scale"])]),
        ("q-value win-loss", [("qwlloss_sum", 1.0)]),
        ("C0 ownership", [("oloss_sum", 1.0)]),
        ("C1 seki slot", [("skloss_sum", scales["seki_loss_scale"])]),
        # fploss holds its coefficient (loss_futurepos_samplewise), sloss does not (loss_sum multiplies it).
        ("C2-3 future positions", [("fploss_sum", 1.0)]),
        ("C4 scoring slot", [("sloss_sum", coefficients["scoring"])]),
        ("C5-6 income slot", [("iloss_sum", 1.0)]),
        ("regret", [("rgloss_sum", 1.0), ("rkloss_sum", 1.0), ("swloss_sum", 1.0), ("swkloss_sum", 1.0)]),
        ("AZdb intrinsic", [("intaloss_sum", value), ("intbloss_sum", value)]),
    ]


def parse_settings(text, known, what):
    """NAME=VALUE,...: a dict of the settings that it names, each of `known` (`what` names them in an error)."""
    out = {}
    for item in filter(None, text.split(",")):
        name, _, value = item.partition("=")
        if name not in known:
            raise ValueError(f"no {what} named {name} ({', '.join(known)})")
        out[name] = float(value)
    return out


def parse_coefficients(text):
    """-loss-coefficients' NAME=VALUE,...: a dict of the loss coefficients that it names (games.LOSS_COEFFICIENTS)."""
    return parse_settings(text, games.LOSS_COEFFICIENTS, "loss coefficient")


def parse_scales(text):
    """-shared-gradient-scales' NAME=VALUE,...: a dict of the heads' factors that it names
    (games.SHARED_GRADIENT_SCALES)."""
    return parse_settings(text, games.SHARED_GRADIENT_SCALES, "shared gradient scale")


def set_shared_gradient_scales(model, scales):
    """Sets the value heads' factors that `scales` names, in place of the game's (model_pytorch's ValueHead)."""
    heads = [model.value_head] + ([model.intermediate_value_head] if model.get_has_intermediate_head() else [])
    for head in heads:
        head.scoring_gradient_scale = float(scales.get("scoring", head.scoring_gradient_scale))
        head.futurepos_gradient_scale = float(scales.get("futurepos", head.futurepos_gradient_scale))


def shared_gradient_scales(model):
    return {"scoring": model.value_head.scoring_gradient_scale, "futurepos": model.value_head.futurepos_gradient_scale}


def group_factors(model, head_lr_factor=0.5, noreg_lr_factor=1.0):
    """Each parameter's factor of the per-sample rate, as train.py's SGD gives its parameter group (add_reg_dict)."""
    reg_dict = {}
    model.add_reg_dict(reg_dict)
    scale = {"output": head_lr_factor, "output_noreg": head_lr_factor * noreg_lr_factor, "noreg": noreg_lr_factor,
             "input_noreg": noreg_lr_factor}
    return {id(p): scale.get(group, 1.0) for group, params in reg_dict.items() for p in params}


def norm(tensors):
    return math.sqrt(sum(float(torch.sum(torch.square(t))) for t in tensors if t is not None))


def dot(a, b):
    return sum(float(torch.sum(x * y)) for x, y in zip(a, b) if x is not None and y is not None)


def measure(model, metrics, batch, scales, lr=0.00003, momentum=0.9, head_lr_factor=0.5, noreg_lr_factor=1.0):
    """The groups' measures on the batch: [(name, the columns of COLUMNS)], with None for a head step without own
    parameters."""
    rows = batch["globalTargetsNC"].shape[0]
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trunk = [p for name, p in named if not name.startswith(HEADS)]
    heads = [p for name, p in named if name.startswith(HEADS)]
    factors = group_factors(model, head_lr_factor, noreg_lr_factor)
    captured = {}
    hook = model.act_trunkfinal.register_forward_hook(lambda module, inputs, output: captured.__setitem__("x", output))
    try:
        outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
    finally:
        hook.remove()
    # The main heads' losses (output 0). is_intermediate=True only leaves out the statistics that follow the losses.
    results = metrics.metrics_dict_batchwise_single_heads_output(
        model, model.postprocess_output(outputs)[0], batch, is_training=False, disable_optimistic_policy=False,
        meta_kata_only_soft_policy=False, is_intermediate=True, **scales,
    )
    terms = [(name, sum(results[k] * scale for k, scale in parts))
             for name, parts in groups(model.config["version"], scales, metrics.loss_coefficients)]
    total = sum(t for _, t in terms)
    assert torch.isclose(total, results["loss_sum"], rtol=1e-4, atol=1e-4), (float(total), float(results["loss_sum"]))

    # Each group's loss per row, the norm of its gradient on the trunk's output, and its gradients on the trunk's
    # parameters and the heads' (zeros where it does not reach).
    measured = []
    for name, term in terms + [("all", total)]:
        loss = float(term.detach() if torch.is_tensor(term) else term) / rows
        if not torch.is_tensor(term) or not term.requires_grad:
            measured.append((name, loss, 0.0, [torch.zeros_like(p) for p in trunk], [torch.zeros_like(p) for p in heads]))
            continue
        g = torch.autograd.grad(term / rows, [captured["x"]] + trunk + heads, retain_graph=True, allow_unused=True)
        g = [torch.zeros_like(p) if gi is None else gi for p, gi in zip([captured["x"]] + trunk + heads, g)]
        measured.append((name, loss, norm(g[:1]), g[1:1 + len(trunk)], g[1 + len(trunk):]))

    def step(params, g, own):
        """The trainer's momentum step for the batch's mean gradient g, over the norm of the parameters that it moves
        (own: 0 or 1 for each parameter's elements)."""
        size = norm([p.detach() * o for p, o in zip(params, own)])
        if size == 0:
            return None
        return norm([lr * factors[id(p)] * rows * gi * o / (1.0 - momentum) for p, gi, o in zip(params, g, own)]) / size

    # A group's own head weights: those with a gradient from it and from no other group ("all" is not a group).
    reach = [[gi != 0 for gi in head] for _, _, _, _, head in measured[:-1]]
    count = [sum(r[i].int() for r in reach) for i in range(len(heads))]
    _, _, _, policy, _ = measured[0]
    _, _, _, value, _ = measured[2]
    out = []
    for k, (name, loss, output, params, head) in enumerate(measured):
        own = [(reach[k][i] & (count[i] == 1)).to(p.dtype) for i, p in enumerate(heads)] if k < len(reach) else None
        out.append((name, (loss, norm(params), norm(params) / norm(policy), output, output / measured[0][2],
                           dot(params, policy) / (norm(params) * norm(policy) or 1.0),
                           dot(params, value) / (norm(params) * norm(value) or 1.0),
                           step(trunk, params, [1.0] * len(trunk)), None if own is None else step(heads, head, own))))
    return out


def print_table(title, lines):
    print(title)
    print(f"{'loss':<24} {'per row':>10} {'trunk params':>13} {'ratio':>8} {'trunk output':>13} {'ratio':>8} "
          f"{'cos policy':>10} {'cos value':>10} {'trunk step':>11} {'head step':>11}")
    for name, (loss, params, ratio, output, oratio, cos_p, cos_v, tstep, hstep) in lines:
        hs = "-" if hstep is None else f"{hstep:.3g}"
        print(f"{name:<24} {loss:>10.5f} {params:>13.5g} {ratio:>8.4f} {output:>13.5g} {oratio:>8.4f} "
              f"{cos_p:>10.3f} {cos_v:>10.3f} {tstep:>11.3g} {hs:>11}")


def mean_lines(batches):
    """The mean of each column over the batches' lines (a head step's mean over the batches that have one)."""
    out = []
    for i, (name, _) in enumerate(batches[0]):
        columns = [b[i][1] for b in batches]
        means = [sum(c[j] for c in columns) / len(columns) for j in range(len(COLUMNS) - 1)]
        steps = [c[-1] for c in columns if c[-1] is not None]
        out.append((name, (*means, sum(steps) / len(steps) if steps else None)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-checkpoint", required=True)
    ap.add_argument("-npz", required=True, nargs="+")
    ap.add_argument("-batches", type=int, default=1)
    ap.add_argument("-batch-size", type=int, default=64)
    ap.add_argument("-threads", type=int, default=2)
    ap.add_argument("-seed", type=int, default=0)
    ap.add_argument("-loss-coefficients", type=parse_coefficients, default={})
    ap.add_argument("-shared-gradient-scales", type=parse_scales, default={})
    ap.add_argument("-soft-policy-weight-scale", type=float, default=8.0)
    ap.add_argument("-value-loss-scale", type=float, default=0.6)
    ap.add_argument("-td-value-loss-scales", type=str, default="0.6,0.6,0.6")
    ap.add_argument("-seki-loss-scale", type=float, default=1.0)
    ap.add_argument("-variance-time-loss-scale", type=float, default=1.0)
    ap.add_argument("-lr", type=float, default=0.00003, help="the trainer's per-sample rate (KataGo's base)")
    ap.add_argument("-momentum", type=float, default=0.9)
    ap.add_argument("-head-lr-factor", type=float, default=0.5)
    ap.add_argument("-noreg-lr-factor", type=float, default=1.0)
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
    metrics.loss_coefficients.update(args.loss_coefficients)
    set_shared_gradient_scales(model, args.shared_gradient_scales)
    torch.manual_seed(args.seed)
    batches = itertools.islice(read_npz_training_data(
        args.npz, batch_size=args.batch_size, world_size=1, rank=0, pos_len=model.pos_len, device="cpu",
        randomize_symmetries=False, include_meta=False, model_config=model.config,
    ), args.batches)

    trunk = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith(HEADS)]
    print(f"{args.checkpoint}: {sum(p.numel() for p in trunk)} trunk parameters in {len(trunk)} tensors; loss "
          f"coefficients {metrics.loss_coefficients}; shared gradient scales {shared_gradient_scales(model)}")
    measured = []
    for i, batch in enumerate(batches):
        measured.append(measure(model, metrics, batch, scales, args.lr, args.momentum, args.head_lr_factor,
                                args.noreg_lr_factor))
        print_table(f"batch {i}: {batch['globalTargetsNC'].shape[0]} rows", measured[-1])
    if len(measured) > 1:
        print_table(f"the mean over {len(measured)} batches", mean_lines(measured))


if __name__ == "__main__":
    main()
