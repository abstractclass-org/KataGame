"""The shared gradient scales (games.SHARED_GRADIENT_SCALES, model_pytorch's ValueHead): the identity in the forward
pass, and N times C4's and C2-3's gradients into the layers that their heads share with the others in the backward
pass. The heads' own weights and C5-6's channels of conv_futurepos get the gradients of the factor 1."""
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from katago.train import games  # noqa: E402
from katago.train.games import enclosure  # noqa: E402
from katago.train.metrics_pytorch import Metrics  # noqa: E402
from test_game_module import b6c96, random_batch  # noqa: E402

SCALES = {"scoring": 400.0, "futurepos": 2.0}
HEADS = ("policy_head.", "value_head.")


def model_at(monkeypatch, scales):
    """losses()' random Blueprints b6c96 (the same weights at every scale), with the shared gradient scales."""
    monkeypatch.setattr(games.get("blueprints"), "SHARED_GRADIENT_SCALES", scales, raising=False)
    model = b6c96("blueprints", 1)
    model.initialize()
    return model


def shared(model):
    """The layers that C4's and C2-3's heads share with the others: the trunk, and the value head's first layer."""
    trunk = [p for name, p in model.named_parameters() if not name.startswith(HEADS)]
    return trunk + [model.value_head.conv1.weight, model.value_head.bias1.beta]


def run(model):
    """The model's outputs on the tests' random batch, and the gradient of each term on every parameter."""
    batch = random_batch(model)
    batch["globalTargetsNC"][:, [34, 68]] = 1.0  # C4's weight and C1's flag
    outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
    metrics = Metrics(1, model)
    # The main heads' losses with their graphs (metrics_dict_batchwise detaches all but loss_sum).
    results = metrics.metrics_dict_batchwise_single_heads_output(
        model, model.postprocess_output(outputs)[0], batch, is_training=False, disable_optimistic_policy=False,
        meta_kata_only_soft_policy=False, is_intermediate=True, soft_policy_weight_scale=8.0, value_loss_scale=0.6,
        td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=1.0, variance_time_loss_scale=1.0)
    terms = {"C4": results["sloss_sum"] * metrics.loss_coefficients["scoring"], "C2-3": results["fploss_sum"],
             "C5-6": results["iloss_sum"], "value": results["vloss_sum"], "all": results["loss_sum"]}
    params = list(model.parameters())
    grads = {name: dict(zip(map(id, params), torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)))
             for name, term in terms.items()}
    return [o.detach() for o in model.postprocess_output(outputs)[0] if torch.is_tensor(o)], grads


def grad_of(grads, term, params):
    return [grads[term][id(p)] if grads[term][id(p)] is not None else torch.zeros_like(p) for p in params]


def close(a, b, rel=1e-5):
    """Each tensor of a against b, within rel of the largest |b|."""
    return all(torch.allclose(x, y, rtol=0.0, atol=rel * max(float(y.abs().max()), 1e-30)) for x, y in zip(a, b))


def test_the_scales_leave_the_forward_pass_as_it_is(monkeypatch):
    plain, _ = run(model_at(monkeypatch, {}))
    scaled, _ = run(model_at(monkeypatch, SCALES))
    assert len(plain) == len(scaled)
    for a, b in zip(plain, scaled):
        assert torch.allclose(a, b, rtol=1e-6, atol=1e-6)


def test_the_heads_own_weights_get_the_gradients_of_the_factor_1(monkeypatch):
    plain_model = model_at(monkeypatch, {})
    _, plain = run(plain_model)
    scaled_model = model_at(monkeypatch, SCALES)
    _, scaled = run(scaled_model)
    for name in ("conv_scoring", "conv_futurepos", "conv_ownership"):  # every row of conv_futurepos: C2-3's and C5-6's
        a = getattr(plain_model.value_head, name).weight
        b = getattr(scaled_model.value_head, name).weight
        assert close(grad_of(scaled, "all", [b]), grad_of(plain, "all", [a]))


def test_the_shared_layers_get_n_times_each_heads_part(monkeypatch):
    plain_model = model_at(monkeypatch, {})
    _, plain = run(plain_model)
    scaled_model = model_at(monkeypatch, SCALES)
    _, scaled = run(scaled_model)
    a, b = shared(plain_model), shared(scaled_model)
    for term, factor in (("C4", 400.0), ("C2-3", 2.0), ("C5-6", 1.0), ("value", 1.0)):
        assert close(grad_of(scaled, term, b), [factor * g for g in grad_of(plain, term, a)]), term
    # The whole loss: each head's part N times, every other loss's part as it was.
    expected = [g + 399.0 * c4 + 1.0 * c23 for g, c4, c23 in
                zip(grad_of(plain, "all", a), grad_of(plain, "C4", a), grad_of(plain, "C2-3", a))]
    assert close(grad_of(scaled, "all", b), expected, rel=1e-4)
    # C4's part reaches the trunk through the value head's first layer, C2-3's directly.
    assert float(grad_of(plain, "C4", [plain_model.value_head.conv1.weight])[0].abs().sum()) > 0
    assert float(grad_of(plain, "C2-3", [plain_model.value_head.conv1.weight])[0].abs().sum()) == 0


def test_every_game_but_blueprints_has_the_factor_1_and_the_model_reads_its_games():
    # Blueprints' factors: each head pulls the trunk about as hard as C1 (blueprints.py).
    for name, scales in (("enclosure", {"futurepos": 1.0, "scoring": 1.0}), ("connect4", {"futurepos": 1.0, "scoring": 1.0}),
                         ("blueprints", {"futurepos": 2.0, "scoring": 400.0})):
        assert games.shared_gradient_scales(games.get(name)) == scales
        head = b6c96(name, 1).value_head
        assert (head.scoring_gradient_scale, head.futurepos_gradient_scale) == (scales["scoring"], scales["futurepos"])


def test_a_module_that_sets_the_scale_of_an_unknown_head_is_refused(monkeypatch):
    stub = types.ModuleType("katago.train.games.badscales")
    stub.__dict__.update({m: getattr(enclosure, m) for m in games.MEMBERS})
    stub.NAME = "badscales"
    stub.SHARED_GRADIENT_SCALES = {"scoring": 2.0, "score": 2.0}
    monkeypatch.setitem(sys.modules, "katago.train.games.badscales", stub)
    with pytest.raises(ValueError, match="sets the gradient scale of no head named score"):
        games.get("badscales")
    assert games.shared_gradient_scales(enclosure) == {"futurepos": 1.0, "scoring": 1.0}
