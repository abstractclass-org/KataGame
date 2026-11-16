"""loss_screen.py: the boundary of the trainer's step on a quadratic, the heads on a cached trunk output, and the
power iteration on a masked Hessian."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import loss_screen  # noqa: E402
from test_diversity_members import intrinsic_model  # noqa: E402
from test_game_module import random_batch  # noqa: E402


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


def test_the_power_iteration_finds_the_largest_eigenvalue_on_the_masked_elements():
    a = torch.tensor([1.0, 5.0, 2.0])
    w = torch.tensor([0.3, -0.2, 0.7], requires_grad=True)
    term = 0.5 * (a * w * w).sum() + 0.0 * w.sum()
    assert loss_screen.top_eigenvalue(term, [w], [torch.ones(3)], 30)[0] == pytest.approx(5.0, rel=1e-4)
    assert loss_screen.top_eigenvalue(term, [w], [torch.tensor([1.0, 0.0, 1.0])], 30)[0] == pytest.approx(2.0, rel=1e-4)
