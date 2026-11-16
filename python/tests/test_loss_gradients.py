"""loss_gradients.py: its groups add up to the trainer's loss_sum on a model with AZdb's intrinsic value head, C4's
group follows the game's coefficient, and the steps of C4's and C2-3's own head weights follow their coefficients."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import loss_gradients  # noqa: E402
from katago.train import games  # noqa: E402
from katago.train.metrics_pytorch import Metrics  # noqa: E402
from test_diversity_members import intrinsic_model  # noqa: E402
from test_game_module import random_batch  # noqa: E402

SCALES = dict(soft_policy_weight_scale=8.0, value_loss_scale=0.6, td_value_loss_scales=[0.6, 0.6, 0.6],
              seki_loss_scale=1.0, variance_time_loss_scale=1.0)


def measured(**coefficients):
    """loss_gradients' lines on a random batch of 2 Blueprints rows whose AZdb targets have weights, on a b1c6nbt
    with the intrinsic value head, with the coefficients set."""
    model = intrinsic_model(kind="b1c6nbt")
    model.train()
    batch = random_batch(model)
    batch["globalTargetsNC"][:, 80:84] = torch.tensor([0.5, 1.0, -0.5, 1.0])
    metrics = Metrics(1, model)
    metrics.loss_coefficients.update(coefficients)
    return dict(loss_gradients.measure(model, metrics, batch, SCALES))


def test_the_groups_add_up_to_loss_sum_with_azdbs_intrinsic_losses():
    # measure() checks that the groups add up to loss_sum: without the intrinsic group, they fell short of it.
    lines = measured()
    loss, params = lines["AZdb intrinsic"][:2]
    assert loss > 0 and params > 0
    assert lines["all"][-1] is None  # "all" is not a group, so it has no own head weights


def test_c4s_and_c2_3s_lines_follow_their_coefficients():
    base, raised = measured(scoring=0.25, futurepos=0.25), measured(scoring=2.5, futurepos=1.0)
    for name, factor in (("C4 scoring slot", 10.0), ("C2-3 future positions", 4.0)):
        (loss, params, _, output, _, cos_p, cos_v, tstep, hstep) = raised[name]
        (loss0, params0, _, output0, _, cos_p0, cos_v0, tstep0, hstep0) = base[name]
        assert hstep is not None and hstep0 is not None  # each has head weights of its own (conv_scoring, C2-3's two
        # channels of conv_futurepos)
        for got, before in ((loss, loss0), (params, params0), (output, output0), (tstep, tstep0), (hstep, hstep0)):
            assert got == pytest.approx(factor * before, rel=1e-4)
        assert (cos_p, cos_v) == pytest.approx((cos_p0, cos_v0), abs=1e-4)
    assert raised["C0 ownership"] == pytest.approx(base["C0 ownership"], rel=1e-6)


def test_a_coefficient_of_no_loss_is_refused():
    assert loss_gradients.parse_coefficients("scoring=150,futurepos=0.75") == {"scoring": 150.0, "futurepos": 0.75}
    with pytest.raises(ValueError, match="no loss coefficient named score"):
        loss_gradients.parse_coefficients("score=1")
    assert set(loss_gradients.parse_coefficients(",".join(f"{k}=1" for k in games.LOSS_COEFFICIENTS))) == set(
        games.LOSS_COEFFICIENTS)
