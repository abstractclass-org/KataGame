"""loss_screen.py: the boundary of the trainer's step on a quadratic, the heads on a cached trunk output, and the
power iteration on a masked Hessian."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import loss_screen  # noqa: E402
from katago.train import games  # noqa: E402
from katago.train.metrics_pytorch import Metrics  # noqa: E402
from loss_gradients import group_factors  # noqa: E402
from test_diversity_members import intrinsic_model  # noqa: E402
from test_game_module import b6c96, random_batch  # noqa: E402


def test_the_boundary_without_lookahead_is_2_times_1_plus_the_momentum():
    assert loss_screen.boundary(0.9, k=1, alpha=1.0) == pytest.approx(3.8, abs=1e-6)
    assert loss_screen.boundary(0.0, k=1, alpha=1.0) == pytest.approx(2.0, abs=1e-6)


def trainer_steps(x, steps=600, k=6, alpha=0.5):
    """|w| after the trainer's steps on 0.5 * 2 * w^2 (curvature 2) at the fast rate x / 2: torch's SGD with momentum
    0.9, and lookahead every k steps (train.py)."""
    w = torch.tensor([1.0], requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=x / 2.0, momentum=0.9)
    slow = w.detach().clone()
    for step in range(steps):
        optimizer.zero_grad()
        (w * w).sum().backward()
        optimizer.step()
        if (step + 1) % k == 0:
            with torch.no_grad():
                slow.add_(w - slow, alpha=alpha)
                w.copy_(slow)
    return float(w.detach().abs())


def test_the_trainers_steps_converge_below_the_boundary_and_diverge_above_it():
    limit = loss_screen.boundary()
    assert trainer_steps(0.8 * limit) < 1e-3
    assert trainer_steps(1.2 * limit) > 1e3


def test_the_heads_on_the_trunks_output_give_the_models_outputs():
    model = intrinsic_model(kind="b1c6nbt")
    batch = random_batch(model)
    captured = {}
    model.act_trunkfinal.register_forward_hook(lambda m, i, o: captured.__setitem__("x", o))
    with torch.no_grad():
        full = model(batch["binaryInputNCHW"], batch["globalInputNC"])
        heads = loss_screen.heads_on(model, captured["x"], batch["binaryInputNCHW"], batch["globalInputNC"])
    assert len(full[0]) == len(heads[0])
    for a, b in zip(full[0], heads[0]):
        assert (a is None and b is None) or torch.equal(a, b)


def blueprints_b6c96(monkeypatch, scales):
    monkeypatch.setattr(games.get("blueprints"), "SHARED_GRADIENT_SCALES", scales, raising=False)
    model = b6c96("blueprints", 1)
    model.initialize()
    return model


def jacobian_product(model, coefficients, v, epsilon):
    batch = random_batch(model)
    batch["globalTargetsNC"][:, [34, 68]] = 1.0
    metrics = Metrics(1, model)
    metrics.loss_coefficients.update(coefficients)
    params = loss_screen.shared_parameters(model)
    factors = group_factors(model)
    roots = [(loss_screen.LR * factors[id(p)] / loss_screen.LOOKAHEAD_ALPHA) ** 0.5 for p in params]
    return loss_screen.update_jacobian(model, metrics, batch, params, roots)(v, epsilon)


def test_the_update_on_the_shared_layers_with_the_scales_is_that_of_the_coefficients_times_the_scales(monkeypatch):
    # With the shared gradient scales, the update on the shared layers is the gradient of the loss whose C4 and C2-3
    # coefficients are times their scales. So the Jacobians of the two real fields agree there. (A double backward is
    # no reference: KataGo's losses read some predictions detached, so the field's Jacobian is not the loss's Hessian.)
    scaled = blueprints_b6c96(monkeypatch, {"scoring": 400.0, "futurepos": 4.0})
    plain = blueprints_b6c96(monkeypatch, {})
    gen = torch.Generator().manual_seed(5)
    v = [torch.randn(p.shape, generator=gen) for p in loss_screen.shared_parameters(scaled)]
    size_v = sum(float((vi * vi).sum()) for vi in v) ** 0.5
    v = [vi / size_v for vi in v]
    got = jacobian_product(scaled, {}, v, 5.0)
    expected = jacobian_product(plain, {"scoring": 0.25 * 400.0, "futurepos": 0.25 * 4.0}, v, 5.0)
    size = sum(float((e * e).sum()) for e in expected) ** 0.5
    error = sum(float(((a - e) ** 2).sum()) for a, e in zip(got, expected)) ** 0.5
    assert size > 0 and error < 0.01 * size, (error, size)
    # Without the scales, the product differs: the scales reach it.
    unscaled = jacobian_product(plain, {}, v, 5.0)
    assert sum(float(((a - e) ** 2).sum()) for a, e in zip(unscaled, expected)) ** 0.5 > 0.05 * size


def test_the_weighted_hessian_by_chunks_of_rows_is_that_of_the_whole_batch(monkeypatch):
    # The double backward runs on chunks of the batch's rows, which loss_sum sums. On a model with the scales, it is the
    # Hessian of the loss whose coefficients are times the scales, at the scales 1, which it leaves as they were.
    model = blueprints_b6c96(monkeypatch, {"scoring": 400.0, "futurepos": 4.0})
    batch = random_batch(model)
    batch["globalTargetsNC"][:, [34, 68]] = 1.0
    metrics = Metrics(1, model)
    params = loss_screen.shared_parameters(model)
    roots = [0.01] * len(params)
    gen = torch.Generator().manual_seed(6)
    v = [torch.randn(p.shape, generator=gen) for p in params]
    whole = loss_screen.weighted_hessian(model, metrics, batch, params, roots, 2)(v)
    chunks = loss_screen.weighted_hessian(model, metrics, batch, params, roots, 1)(v)
    size = sum(float((w * w).sum()) for w in whole) ** 0.5
    assert size > 0
    assert sum(float(((a - b) ** 2).sum()) for a, b in zip(chunks, whole)) ** 0.5 < 1e-4 * size
    assert model.value_head.shared_gradient_scales["scoring"] == 400.0 and metrics.loss_coefficients["scoring"] == 0.25
    # At the scales 1 and the coefficients 0.25 x 400 and 0.25 x 4 directly, the same product.
    plain = blueprints_b6c96(monkeypatch, {})
    plain_metrics = Metrics(1, plain)
    plain_metrics.loss_coefficients.update(scoring=100.0, futurepos=1.0)
    direct = loss_screen.weighted_hessian(plain, plain_metrics, batch, loss_screen.shared_parameters(plain), roots, 2)(v)
    assert sum(float(((a - b) ** 2).sum()) for a, b in zip(direct, whole)) ** 0.5 < 1e-4 * size


def test_the_power_iteration_finds_the_largest_eigenvalue_on_the_masked_elements():
    a = torch.tensor([1.0, 5.0, 2.0])
    w = torch.tensor([0.3, -0.2, 0.7], requires_grad=True)
    term = 0.5 * (a * w * w).sum() + 0.0 * w.sum()
    assert loss_screen.top_eigenvalue(term, [w], [torch.ones(3)], 30)[0] == pytest.approx(5.0, rel=1e-4)
    assert loss_screen.top_eigenvalue(term, [w], [torch.tensor([1.0, 0.0, 1.0])], 30)[0] == pytest.approx(2.0, rel=1e-4)
