#!/usr/bin/python3
"""KataGame fork: a screen of a change of the trainer's loss coefficients, on the CPU, before a run trains with it.

    python loss_screen.py curvature -checkpoint CKPT -npz ROWS.npz [ROWS.npz ...] [-batches 1] [-batch-size 64]
                          [-threads 2] [-sets "scoring=0.25,futurepos=0.25;scoring=75,futurepos=0.75" ...]
    python loss_screen.py steps -checkpoint TRAINER_CKPT -npz ROWS.npz [-steps 48] [-batch-size 64] [-threads 4]
                          [-loss-coefficients scoring=75,futurepos=0.75] [-shared-gradient-scales scoring=400,...]
    python loss_screen.py shared -checkpoint CKPT -npz ROWS.npz [-batches 1] [-iterations 12] [-chunk 8]
                          [-epsilon 1] [-loss-coefficients ...] [-shared-gradient-scales scoring=400,futurepos=2]

Every command takes the value head's shared gradient scales of the game (games.SHARED_GRADIENT_SCALES), or those that
-shared-gradient-scales sets (as loss_gradients.py).

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
loss_sum on the value head. With a shared gradient scale other than 1, only the own head weights: the other blocks'
Hessian-vector products would go back through the scale (shared gives them).

shared: the curvature of the trainer's update on the layers that C4's and C2-3's heads share with the others (the
trunk's parameters, and the value head's first layer, conv1 and its bias), with every loss and the shared gradient
scales. On these layers, with the other weights fixed, the update is the gradient of the equivalent weighted loss:
loss_sum with C4's and C2-3's coefficients times their scales, at the scales 1. So the tool takes that loss's Hessian
(weighted_hessian: a double backward, which goes through no scale), between the square roots of each parameter's fast
rate (train.py's group factors / lookahead's alpha), so that its eigenvalue compares with boundary() at once. The
power iteration prints each estimate. If the largest in size is negative, a second power iteration shifted by it
gives the largest positive one. Then the real field's central differences along the top vector (update_jacobian: two
real gradients, through the scales, at parameters moved by +-epsilon * the rate's root * the vector), at epsilon and
4 epsilon, check it. The differences are no power iteration of their own: the trunk's ReLUs make the field kinked, so
a difference quotient depends on its step, and a power iteration on it follows the kinks.

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
from loss_gradients import (HEADS, dot, group_factors, groups, norm, parse_coefficients, parse_scales,
                            set_shared_gradient_scales, shared_gradient_scales)

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
    set_shared_gradient_scales(model, args.shared_gradient_scales)
    scaled = any(v != 1.0 for v in shared_gradient_scales(model).values())
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
          f"lookahead {2 * (1 + MOMENTUM):.3f}); per unit of coefficient at {unit}; shared gradient scales "
          f"{shared_gradient_scales(model)}{' (the own head weights only)' if scaled else ''}")
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
        if scaled:
            names = {C4: ["own head"], C23: ["own head"]}
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
        for i, s in enumerate([] if scaled else sets):
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
        whole = ""
        if not scaled:
            value_head_top = rate * max(line[("all", i)] for line in found)
            whole = f", loss_sum on the value head {value_head_top:.3g} ({value_head_top / limit:.2f})"
        print(f"set {i} {s}: C4 {c4:.3g} ({c4 / limit:.2f} of it), C2-3 {c23:.3g} ({c23 / limit:.2f}){whole}")


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
    set_shared_gradient_scales(model, args.shared_gradient_scales)
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
    factors = group_factors(model, HEAD_LR_FACTOR, NOREG_LR_FACTOR)
    for group in optimizer.param_groups:
        group["lr"] = LR * factors[id(group["params"][0])] / LOOKAHEAD_ALPHA
        group["weight_decay"] = trainer_weight_decay(group["group_name"], model.config["norm_kind"], args.batch_size)
    cap = {"fixup": 2500.0, "fixscale": 2500.0, "fixscaleonenorm": 2500.0}.get(model.config["norm_kind"], 5500.0)
    cap *= math.sqrt(args.batch_size / 256.0)
    slow = {p: p.detach().clone() for group in optimizer.param_groups for p in group["params"]}
    print(f"{args.checkpoint}: momentum {'from the checkpoint' if 'optimizer' in state else 'from 0'}; loss coefficients "
          f"{metrics.loss_coefficients}; shared gradient scales {shared_gradient_scales(model)}; seed {args.seed}; "
          f"gradient-norm cap {cap:g}; rates "
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
        per = {k: float(results[k].detach()) / rows
               for k in ("loss_sum", "p0loss_sum", "vloss_sum", "oloss_sum", "sloss_sum", "fploss_sum")}
        print(f"{step:>4} {per['loss_sum']:>9.4f} {per['p0loss_sum']:>8.4f} {per['vloss_sum']:>7.4f} "
              f"{per['oloss_sum']:>7.4f} {per['sloss_sum']:>9.6f} {per['fploss_sum'] / c23:>9.5f} {gnorm:>9.3f} "
              f"{'yes' if gnorm > cap else 'no':>7} {'yes' if finite else 'NO':>6}", flush=True)
        if not finite:
            break


def shared_parameters(model):
    """The layers that C4's and C2-3's heads share with the others: the trunk's parameters, and the value head's first
    layer (conv1 and its bias), which C4's head reads after its activation."""
    trunk = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith(HEADS)]
    return trunk + [model.value_head.conv1.weight, model.value_head.bias1.beta]


def power_iteration(product, v, iterations, label, shift=0.0):
    """The largest eigenvalue in size of product + shift (a symmetric linear map of lists of tensors), minus the
    shift, from the start v, printing each estimate: (the eigenvalue, its last relative change, the unit vector)."""
    lam, last = 0.0, float("nan")
    for i in range(iterations):
        size = norm(v)
        v = [vi / size for vi in v]
        u = product(v)
        if shift:
            u = [ui + shift * vi for ui, vi in zip(u, v)]
        last, lam = lam, dot(v, u)
        print(f"{label} iteration {i}: {lam - shift:.5g}", flush=True)
        v = u
    size = norm(v)
    return lam - shift, abs(lam - last) / (abs(lam) or 1.0), [vi / size for vi in v]


def update_jacobian(model, metrics, batch, params, roots):
    """product(v, epsilon): the trainer's update field on params (loss_sum's gradient on them, through the shared
    gradient scales, the other weights fixed), its Jacobian between the roots (each parameter's rate's square root),
    times v. A central difference of two real gradients at the parameters moved by +-epsilon * root * v."""
    base = [p.detach().clone() for p in params]

    def field(direction, epsilon):
        with torch.no_grad():
            for p, x, r, d in zip(params, base, roots, direction):
                p.copy_(x + (epsilon * r) * d)
        outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
        results = metrics.metrics_dict_batchwise(
            model, model.postprocess_output(outputs), None, batch, is_training=True, disable_optimistic_policy=False,
            meta_kata_only_soft_policy=False, main_loss_scale=None, intermediate_loss_scale=None,
            include_model_norms=False, **SCALES)
        return torch.autograd.grad(results["loss_sum"], params)

    def product(v, epsilon):
        plus = field(v, epsilon)
        minus = field([-vi for vi in v], epsilon)
        with torch.no_grad():
            for p, x in zip(params, base):
                p.copy_(x)
        return [r * (a - c) / (2.0 * epsilon) for r, a, c in zip(roots, plus, minus)]

    return product


def weighted_hessian(model, metrics, batch, params, roots, chunk):
    """product(v): the Hessian of the equivalent weighted loss on params, between the roots, times v. That loss is
    loss_sum with C4's and C2-3's coefficients times the model's shared gradient scales, at the scales 1: its gradient
    on the shared layers is the trainer's update there, and no second derivative goes through a scale. A
    Hessian-vector product (a double backward) for each chunk of the batch's rows, summed (loss_sum sums its rows)."""
    scales = shared_gradient_scales(model)
    weighted = dict(metrics.loss_coefficients)
    weighted["scoring"] *= scales["scoring"]
    weighted["futurepos"] *= scales["futurepos"]
    rows = batch["globalTargetsNC"].shape[0]

    def product(v):
        saved = dict(metrics.loss_coefficients)
        set_shared_gradient_scales(model, {"scoring": 1.0, "futurepos": 1.0})
        metrics.loss_coefficients.update(weighted)
        try:
            out = [torch.zeros_like(p) for p in params]
            for start in range(0, rows, chunk):
                part = {k: t[start:start + chunk] for k, t in batch.items()}
                outputs = model(part["binaryInputNCHW"], part["globalInputNC"])
                loss = metrics.metrics_dict_batchwise(
                    model, model.postprocess_output(outputs), None, part, is_training=True,
                    disable_optimistic_policy=False, meta_kata_only_soft_policy=False, main_loss_scale=None,
                    intermediate_loss_scale=None, include_model_norms=False, **SCALES)["loss_sum"]
                gradient = torch.autograd.grad(loss, params, create_graph=True)
                hv = torch.autograd.grad(gradient, params, grad_outputs=[r * vi for r, vi in zip(roots, v)])
                out = [o + r * h for o, r, h in zip(out, roots, hv)]
            return out
        finally:
            set_shared_gradient_scales(model, scales)
            metrics.loss_coefficients.clear()
            metrics.loss_coefficients.update(saved)

    return product


def shared(args):
    model, _, _ = load_model(args.checkpoint, use_swa=False, device="cpu")
    model.train()
    metrics = Metrics(1, model)
    metrics.loss_coefficients.update(args.loss_coefficients)
    set_shared_gradient_scales(model, args.shared_gradient_scales)
    factors = group_factors(model, HEAD_LR_FACTOR, NOREG_LR_FACTOR)
    params = shared_parameters(model)
    roots = [math.sqrt(LR * factors[id(p)] / LOOKAHEAD_ALPHA) for p in params]  # each fast rate's square root
    limit = boundary()
    print(f"{args.checkpoint}: {sum(p.numel() for p in params)} shared parameters in {len(params)} tensors, fast rates "
          f"{sorted({round(r * r, 12) for r in roots})} a sample; loss coefficients {metrics.loss_coefficients}; shared "
          f"gradient scales {shared_gradient_scales(model)}; epsilon {args.epsilon}; the boundary {limit:.3f}",
          flush=True)
    torch.manual_seed(args.seed)
    batches = itertools.islice(read_npz_training_data(
        args.npz, batch_size=args.batch_size, world_size=1, rank=0, pos_len=model.pos_len, device="cpu",
        randomize_symmetries=False, include_meta=False, model_config=model.config), args.batches)
    found = []
    v = [torch.randn(p.shape, generator=torch.Generator().manual_seed(args.seed)) for p in params]
    for b, batch in enumerate(batches):
        hessian = weighted_hessian(model, metrics, batch, params, roots, args.chunk)
        # Each batch after the first starts from the top vector of the batch before.
        lam, change, v = power_iteration(hessian, v, args.iterations, f"batch {b}")
        top = lam
        if lam < 0:  # the largest in size is negative: the largest positive one, shifted by it
            top, change, v = power_iteration(hessian, v, args.iterations, f"batch {b} shifted", shift=-lam)
        # The real field's central differences along the top vector, through the scales.
        difference = update_jacobian(model, metrics, batch, params, roots)
        checks = ", ".join(f"epsilon {e:g}: {dot(v, difference(v, e)):.4g}" for e in (args.epsilon, 4 * args.epsilon))
        found.append(top)
        print(f"batch {b}: rate * curvature {top:.4g} ({top / limit:.3f} of the boundary; the largest in size {lam:.4g},"
              f" last change {change:.1e}; the real field's difference quotient along it: {checks})", flush=True)
    print(f"the largest over {len(found)} batches: {max(found):.4g}, {max(found) / limit:.3f} of the boundary {limit:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("curvature", "steps", "shared"):
        p = sub.add_parser(name)
        p.add_argument("-checkpoint", required=True)
        p.add_argument("-npz", required=True, nargs="+")
        p.add_argument("-batch-size", type=int, default=64)
        p.add_argument("-threads", type=int, default=2)
        p.add_argument("-seed", type=int, default=0)
        p.add_argument("-shared-gradient-scales", type=parse_scales, default={})
    curv, step, share = sub.choices["curvature"], sub.choices["steps"], sub.choices["shared"]
    curv.add_argument("-batches", type=int, default=1)
    curv.add_argument("-iterations", type=int, default=20)
    curv.add_argument("-sets", default="", help="sets of loss coefficients to compare, ';' between sets")
    step.add_argument("-steps", type=int, default=48)
    step.add_argument("-loss-coefficients", type=parse_coefficients, default={})
    share.add_argument("-batches", type=int, default=1)
    share.add_argument("-iterations", type=int, default=12)
    share.add_argument("-epsilon", type=float, default=1.0)
    share.add_argument("-chunk", type=int, default=8, help="rows of each double backward (about 0.7 GB a row)")
    share.add_argument("-loss-coefficients", type=parse_coefficients, default={})
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"curvature": curvature, "steps": steps, "shared": shared}[args.command](args)


if __name__ == "__main__":
    main()
