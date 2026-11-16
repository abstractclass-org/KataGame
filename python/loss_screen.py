#!/usr/bin/python3
"""KataGame fork: a screen of a change of the trainer's loss coefficients, on the CPU, before a run trains with it.

    python loss_screen.py curvature -checkpoint CKPT -npz ROWS.npz [ROWS.npz ...] [-batches 1] [-batch-size 64]
                          [-threads 2] [-sets "scoring=0.25,futurepos=0.25;scoring=75,futurepos=0.75" ...]
    python loss_screen.py steps -checkpoint TRAINER_CKPT -npz ROWS.npz [-steps 48] [-batch-size 64] [-threads 4]
                          [-loss-coefficients scoring=75,futurepos=0.75]

A coefficient multiplies its loss's curvature as much as its gradient. A head whose own weights get too large a step
for their curvature oscillates and diverges, whatever its pull on the trunk. On a quadratic, SGD with momentum m at
the rate r converges only while r * lambda < 2 (1 + m), lambda the curvature's largest eigenvalue (3.8 for KataGo's
m = 0.9). The trainer's lookahead (train.py: k = 6, alpha = 0.5) takes its fast steps at r / alpha and pulls the slow
weights alpha of the way every k steps: boundary() gives the limit for this cycle, in units of the fast rate.

curvature: on each batch, the trunk's final activation is computed once and the heads run on it (heads_on), so the
heads' Hessians cost little. For C4's and C2-3's losses, the largest eigenvalue of the loss's Hessian (the batch's sum,
as the trainer differentiates it) by power iteration on Hessian-vector products, per unit of the loss's coefficient:
- on its own head weights (C4: conv_scoring, C2-3: conv_futurepos's 2 channels of C2-3), which no other loss reaches;
- on the value head (its shared conv1 and every head weight on it);
- on the trunk's final activation (no rate: compare it with the policy's and the value's lines).
Then, for each set of coefficients (-sets, ";" between sets; each set as loss_gradients.py's -loss-coefficients), the
fast rate times the eigenvalue on the own head weights, against boundary(), and the largest eigenvalue of the whole
loss_sum on the value head.

steps: real steps of the whole network from a trainer's checkpoint (train/NAME/checkpoint.ckpt: the weights and SGD's
momentum), as train.py takes them with the run's flags: its parameter groups and their rates and weight decays, SGD
with momentum, lookahead, the gradient-norm cap, the batches of the file with random symmetries (the file again from
its start when it ends). Each step prints the losses per row (C4's and C2-3's without their coefficients, so that
sets compare), the gradient's norm against the cap, and whether every loss is finite. The departures from the
trainer: fp32 on the CPU (the trainer: fp16 with a gradient scaler on the GPU), and lookahead's slow weights start at
the checkpoint's weights.
"""

import argparse
import itertools
import math

import torch

from katago.train.data_processing_pytorch import read_npz_training_data
from katago.train.load_model import load_model
from katago.train.metrics_pytorch import Metrics
from loss_gradients import dot, group_factors, groups, norm, parse_coefficients

SCALES = dict(soft_policy_weight_scale=8.0, value_loss_scale=0.6, td_value_loss_scales=[0.6, 0.6, 0.6],
              seki_loss_scale=1.0, variance_time_loss_scale=1.0)  # train.py's defaults
LR, MOMENTUM, LOOKAHEAD_K, LOOKAHEAD_ALPHA, HEAD_LR_FACTOR, NOREG_LR_FACTOR = 0.00003, 0.9, 6, 0.5, 0.5, 1.0
C4, C23 = "C4 scoring slot", "C2-3 future positions"


def boundary(momentum=MOMENTUM, k=LOOKAHEAD_K, alpha=LOOKAHEAD_ALPHA):
    """The largest x = rate * curvature (the fast weights' rate) at which SGD with momentum under lookahead converges on
    a quadratic. The state (weight, rate * momentum buffer) moves by [[1 - x, -m], [x, m]] a step (torch's SGD), and
    lookahead's cycle maps the slow weight w to (1 - alpha) w + alpha * (w after k steps), keeping the buffer."""
    def radius(x):
        step = torch.tensor([[1.0 - x, -momentum], [x, momentum]], dtype=torch.float64)
        cycle = torch.diag(torch.tensor([alpha, 1.0], dtype=torch.float64)) @ torch.linalg.matrix_power(step, k)
        cycle[0, 0] += 1.0 - alpha
        return float(torch.linalg.eigvals(cycle).abs().max())
    x = 0.01
    while radius(x) < 1.0:
        x += 0.01
    lo, hi = x - 0.01, x
    for _ in range(40):
        lo, hi = ((lo + hi) / 2, hi) if radius((lo + hi) / 2) < 1.0 else (lo, (lo + hi) / 2)
    return lo


def heads_on(model, x, binary, global_inputs):
    """The main heads' outputs on the trunk's final activation x: Model.forward after act_trunkfinal, for a model
    without intermediate heads."""
    mask = binary[:, 0:1, :, :].contiguous()
    mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)
    mask_sum = torch.sum(mask)
    policy = model.policy_head(x, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=None)
    return ((policy, *model.value_head(x, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum,
                                       input_global=global_inputs, extra_outputs=None)),)


def terms(model, metrics, batch, outputs):
    """loss_gradients' groups on the outputs, by name, each the batch's sum as loss_sum adds it, and "all"."""
    results = metrics.metrics_dict_batchwise_single_heads_output(
        model, model.postprocess_output(outputs)[0], batch, is_training=False, disable_optimistic_policy=False,
        meta_kata_only_soft_policy=False, is_intermediate=True, **SCALES)
    out = {name: sum(results[k] * scale for k, scale in parts)
           for name, parts in groups(model.config["version"], SCALES, metrics.loss_coefficients)}
    out["all"] = results["loss_sum"]
    return out


def top_eigenvalue(term, params, masks, iterations, seed=0):
    """The largest eigenvalue (in size) of the Hessian of term on the masked elements of params, by power iteration,
    and its relative change in the last iteration."""
    g = torch.autograd.grad(term, params, create_graph=True, allow_unused=True)
    reached = [i for i, gi in enumerate(g) if gi is not None and gi.requires_grad]
    if not reached:
        return 0.0, 0.0
    params, masks, g = [params[i] for i in reached], [masks[i] for i in reached], [g[i] for i in reached]
    gen = torch.Generator().manual_seed(seed)
    v = [torch.randn(p.shape, generator=gen) * m for p, m in zip(params, masks)]
    lam = last = 0.0
    for _ in range(iterations):
        size = norm(v)
        v = [vi / size for vi in v]
        hv = torch.autograd.grad(g, params, grad_outputs=v, retain_graph=True, allow_unused=True)
        hv = [torch.zeros_like(p) if h is None else h * m for p, h, m in zip(params, hv, masks)]
        last, lam = lam, dot(v, hv)
        v = hv
    return lam, abs(lam - last) / (abs(lam) or 1.0)


def curvature(args):
    model, _, _ = load_model(args.checkpoint, use_swa=False, device="cpu")
    model.train()
    metrics = Metrics(1, model)
    sets = [parse_coefficients(s) for s in args.sets.split(";")] if args.sets else [{}]
    coefficients = dict(metrics.loss_coefficients)
    factors = group_factors(model, HEAD_LR_FACTOR, NOREG_LR_FACTOR)
    value_head = [(name, p) for name, p in model.named_parameters() if name.startswith("value_head.")]
    rate = {LR * factors[id(p)] / LOOKAHEAD_ALPHA for _, p in value_head}
    assert len(rate) == 1, f"the value head's weights train at several rates: {rate}"
    rate = rate.pop()  # the fast rate of every value head weight, per sample
    limit = boundary()
    scoring, futurepos = model.value_head.conv_scoring.weight, model.value_head.conv_futurepos.weight
    own = {C4: ([scoring], [torch.ones_like(scoring)]),
           C23: ([futurepos], [torch.zeros_like(futurepos).index_fill_(0, torch.tensor([0, 1]), 1.0)])}
    unit = {C4: coefficients["scoring"], C23: coefficients["futurepos"]}
    print(f"{args.checkpoint}: the value head's fast rate {rate:.3g} a sample; the boundary of rate * curvature for "
          f"momentum {MOMENTUM} and lookahead k {LOOKAHEAD_K}, alpha {LOOKAHEAD_ALPHA}: {limit:.3f} (without "
          f"lookahead {2 * (1 + MOMENTUM):.3f}); per unit of coefficient at {unit}")
    torch.manual_seed(args.seed)
    batches = itertools.islice(read_npz_training_data(
        args.npz, batch_size=args.batch_size, world_size=1, rank=0, pos_len=model.pos_len, device="cpu",
        randomize_symmetries=False, include_meta=False, model_config=model.config), args.batches)
    found = []
    for b, batch in enumerate(batches):
        binary, global_inputs = batch["binaryInputNCHW"], batch["globalInputNC"]
        captured = {}
        hook = model.act_trunkfinal.register_forward_hook(lambda m, i, o: captured.__setitem__("x", o.detach()))
        try:
            with torch.no_grad():
                full = model(binary, global_inputs)
        finally:
            hook.remove()
        x = captured["x"].clone().requires_grad_(True)
        with torch.no_grad():
            check = heads_on(model, x, binary, global_inputs)
        assert all((a is None and b is None) or torch.allclose(a, b, rtol=1e-4, atol=1e-5)
                   for a, b in zip(full[0], check[0])), "heads_on gives other outputs than the model"
        line = {}
        blocks = {"own head": None,
                  "value head": ([p for _, p in value_head], [torch.ones_like(p) for _, p in value_head]),
                  "trunk output": ([x], [torch.ones_like(x)])}
        # The policy and the value for comparison: the policy head does not read the value head.
        names = {C4: blocks, C23: blocks, "policy": ["trunk output"], "value": ["value head", "trunk output"]}
        for name, chosen in names.items():
            for block in chosen:
                params, masks = own[name] if blocks[block] is None else blocks[block]
                metrics.loss_coefficients.update(coefficients)
                term = terms(model, metrics, batch, heads_on(model, x, binary, global_inputs))[name]
                lam, change = top_eigenvalue(term, params, masks, args.iterations)
                per = unit[name] if name in unit else 1.0
                line[(name, block)] = lam / per
                print(f"batch {b} {name:<22} {block:<13} eigenvalue {lam:12.5g} (per unit {lam / per:12.5g}, "
                      f"last change {change:.1e})", flush=True)
        for i, s in enumerate(sets):
            metrics.loss_coefficients.update(coefficients)
            metrics.loss_coefficients.update(s)
            term = terms(model, metrics, batch, heads_on(model, x, binary, global_inputs))["all"]
            params, masks = [p for _, p in value_head], [torch.ones_like(p) for _, p in value_head]
            lam, change = top_eigenvalue(term, params, masks, args.iterations)
            line[("all", i)] = lam
            print(f"batch {b} set {i} {s}: loss_sum on the value head eigenvalue {lam:.5g} (rate * eigenvalue "
                  f"{rate * lam:.3g}, last change {change:.1e})", flush=True)
        found.append(line)
    metrics.loss_coefficients.update(coefficients)
    print(f"rate * eigenvalue on the own head weights, the largest over {len(found)} batches, against the boundary "
          f"{limit:.3f}:")
    for i, s in enumerate(sets):
        c = dict(coefficients, **s)
        c4 = rate * c["scoring"] * max(line[(C4, "own head")] for line in found)
        c23 = rate * c["futurepos"] * max(line[(C23, "own head")] for line in found)
        whole = rate * max(line[("all", i)] for line in found)
        print(f"set {i} {s}: C4 {c4:.3g} ({c4 / limit:.2f} of it), C2-3 {c23:.3g} ({c23 / limit:.2f}), loss_sum on the "
              f"value head {whole:.3g} ({whole / limit:.2f})")


def trainer_weight_decay(group, norm_kind, batch_size):
    """train.py's weight decay of a parameter group for SGD, without -wd-floor-frac and the factors' flags."""
    assert norm_kind in ("fixup", "fixscale"), f"the screen knows only fixup's weight decays, not {norm_kind}'s"
    scaling = batch_size / 256.0
    if group in ("input", "normal", "normal_gamma", "output"):
        return 0.000001 * scaling
    if group in ("normal_attn", "normal_gab", "gab_mlp", "tab_module"):
        return 0.000001 * 0.5 * scaling
    return 0.00000001 * scaling


def steps(args):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _, _ = load_model(args.checkpoint, use_swa=False, device="cpu")
    model.train()
    metrics = Metrics(1, model)
    metrics.loss_coefficients.update(args.loss_coefficients)
    reg_dict = {}
    model.add_reg_dict(reg_dict)  # train.py's get_param_groups
    param_groups = [{"params": ps, "group_name": g} for g, ps in reg_dict.items() if len(ps) > 0]
    optimizer = torch.optim.SGD(param_groups, lr=1.0, momentum=MOMENTUM)
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            for p in group["params"]:
                buf = optimizer.state[p].get("momentum_buffer")
                assert buf is None or buf.shape == p.shape, "the checkpoint's momentum does not fit the model"
    factor = {"output": HEAD_LR_FACTOR, "output_noreg": HEAD_LR_FACTOR * NOREG_LR_FACTOR, "noreg": NOREG_LR_FACTOR,
              "input_noreg": NOREG_LR_FACTOR}
    for group in optimizer.param_groups:
        group["lr"] = LR * factor.get(group["group_name"], 1.0) / LOOKAHEAD_ALPHA
        group["weight_decay"] = trainer_weight_decay(group["group_name"], model.config["norm_kind"], args.batch_size)
    cap = {"fixup": 2500.0, "fixscale": 2500.0, "fixscaleonenorm": 2500.0}.get(model.config["norm_kind"], 5500.0)
    cap *= math.sqrt(args.batch_size / 256.0)
    slow = {p: p.detach().clone() for group in optimizer.param_groups for p in group["params"]}
    print(f"{args.checkpoint}: momentum {'from the checkpoint' if 'optimizer' in state else 'from 0'}; loss coefficients "
          f"{metrics.loss_coefficients}; gradient-norm cap {cap:g}; rates "
          f"{ {g['group_name']: g['lr'] for g in optimizer.param_groups} }", flush=True)

    def batches():
        while True:
            yield from read_npz_training_data(
                args.npz, batch_size=args.batch_size, world_size=1, rank=0, pos_len=model.pos_len, device="cpu",
                randomize_symmetries=True, include_meta=False, model_config=model.config)

    torch.manual_seed(args.seed)
    c23 = metrics.loss_coefficients["futurepos"]
    print(f"{'step':>4} {'loss_sum':>9} {'policy':>8} {'value':>7} {'C0':>7} {'C4 raw':>9} {'C2-3 raw':>9} "
          f"{'gnorm':>9} {'clipped':>7} {'finite':>6}")
    for step, batch in enumerate(itertools.islice(batches(), args.steps)):
        rows = batch["globalTargetsNC"].shape[0]
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
        results = metrics.metrics_dict_batchwise(
            model, model.postprocess_output(outputs), None, batch, is_training=True, disable_optimistic_policy=False,
            meta_kata_only_soft_policy=False, main_loss_scale=None, intermediate_loss_scale=None,
            include_model_norms=False, **SCALES)
        results["loss_sum"].backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cap))
        optimizer.step()
        if (step + 1) % LOOKAHEAD_K == 0:  # train.py's lookahead
            with torch.no_grad():
                for p, s in slow.items():
                    s.add_(p.detach() - s, alpha=LOOKAHEAD_ALPHA)
                    p.copy_(s)
        finite = all(math.isfinite(float(v.detach() if torch.is_tensor(v) else v))
                     for k, v in results.items() if k.endswith("_sum"))
        per = {k: float(results[k]) / rows for k in ("loss_sum", "p0loss_sum", "vloss_sum", "oloss_sum", "sloss_sum",
                                                    "fploss_sum")}
        print(f"{step:>4} {per['loss_sum']:>9.4f} {per['p0loss_sum']:>8.4f} {per['vloss_sum']:>7.4f} "
              f"{per['oloss_sum']:>7.4f} {per['sloss_sum']:>9.6f} {per['fploss_sum'] / c23:>9.5f} {gnorm:>9.3f} "
              f"{'yes' if gnorm > cap else 'no':>7} {'yes' if finite else 'NO':>6}", flush=True)
        if not finite:
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("curvature", "steps"):
        p = sub.add_parser(name)
        p.add_argument("-checkpoint", required=True)
        p.add_argument("-npz", required=True, nargs="+")
        p.add_argument("-batch-size", type=int, default=64)
        p.add_argument("-threads", type=int, default=2)
        p.add_argument("-seed", type=int, default=0)
    curv, step = sub.choices["curvature"], sub.choices["steps"]
    curv.add_argument("-batches", type=int, default=1)
    curv.add_argument("-iterations", type=int, default=20)
    curv.add_argument("-sets", default="", help="sets of loss coefficients to compare, ';' between sets")
    step.add_argument("-steps", type=int, default=48)
    step.add_argument("-loss-coefficients", type=parse_coefficients, default={})
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"curvature": curvature, "steps": steps}[args.command](args)


if __name__ == "__main__":
    main()
