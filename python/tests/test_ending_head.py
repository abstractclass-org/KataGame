"""The targets of one ending and the ending head (katago/train/games: ENDING_CLASSES). In a game whose targets differ by
ending, the rows' C27 weighs the score's targets and C0 alone, and the trainer weighs the rest of what KataGo weighs by
C27 apart from it. The ending head (train.py's -heads ending) learns the ending's class, G76, at its weight, G77. A game
or a model without them trains as before (test_game_module pins the games' losses and gradients)."""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import loss_gradients  # noqa: E402
from katago.train import games, modelconfigs  # noqa: E402
from katago.train.games import connect4  # noqa: E402
from katago.train.metrics_pytorch import Metrics  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402
from test_diversity_members import export_header  # noqa: E402
from test_game_module import loss_sums, random_batch, stub_game  # noqa: E402
from test_loss_gradients import SCALES  # noqa: E402

CLASSES = 6
# Columns of the global targets.
COLUMNS = {"win": 0, "loss": 1, "no_result": 2, "score": 3, "short_win": 12, "short_loss": 13, "short_score": 15,
           "variance_time": 22, "row_weight": 25, "c27": 27, "no_value": 35, "c68": 68, "c69": 69, "ending": 76,
           "ending_weight": 77}
# The losses of the score's targets and of C0: C27 weighs them in every game.
SCORE_LOSSES = ("tdsloss_sum", "smloss_sum", "sbcdfloss_sum", "sbpdfloss_sum", "esstloss_sum", "oloss_sum")
# The other losses that KataGo weighs by C27 (with C68 and C69 for C1 and C5-6). Their targets exist in every ending.
OUTCOME_LOSSES = ("p0lopt_sum", "p0sopt_sum", "evstloss_sum", "vtimeloss_sum")
SLOT_LOSSES = ("skloss_sum", "iloss_sum")


@pytest.fixture(autouse=True)
def stub_games(monkeypatch):
    """Two games with Connect-4's inputs and every target but C7: one with 6 classes of endings, one without."""
    targets = (games.TARGET_OWNERSHIP | games.TARGET_SEKI | games.TARGET_FUTURE_PIECES | games.TARGET_SCORING |
               games.TARGET_INCOME)
    stub_game(monkeypatch, "endingsgame", connect4, TARGETS=targets, ENDING_CLASSES=CLASSES)
    stub_game(monkeypatch, "plaingame", connect4, TARGETS=targets)


def model_of(game, heads="", kind="b1c6nbt"):
    """A model of the game with the heads, its parameters of one seed."""
    torch.manual_seed(1)
    config = modelconfigs.with_game(modelconfigs.config_of_name[kind], game)
    model = Model(modelconfigs.with_heads(config, modelconfigs.parse_heads(heads)), games.get(game).MAX_BOARD_LEN)
    model.initialize()
    return model


def rows(model, n=2, **targets):
    """n random rows (test_game_module.random_batch) of a game that its mover won, with every target's weight or flag
    at 1, and the columns that `targets` sets (a value, or a value for each row)."""
    batch = random_batch(model, n)
    for name, value in {"score": 2.0, "short_win": 1.0, "short_score": 1.0, "variance_time": 10.0, "ending": 0.0,
                        "ending_weight": 1.0, **targets}.items():
        batch["globalTargetsNC"][:, COLUMNS[name]] = torch.as_tensor(value, dtype=torch.float32)
    return batch


def sums(game, heads="", **targets):
    model = model_of(game, heads)
    return loss_sums(model, rows(model, **targets))


def test_a_game_has_the_classes_of_endings_that_it_gives():
    for name in ("enclosure", "connect4", "blueprints", "plaingame"):
        assert games.ending_classes(games.get(name)) == 0
    assert games.ending_classes(games.get("endingsgame")) == CLASSES


def test_a_model_has_the_head_only_when_its_config_names_it():
    assert modelconfigs.parse_heads("ending,regret") == ("regret", "ending")
    model = model_of("endingsgame", "regret,ending")
    assert modelconfigs.get_heads(model.config) == ("regret", "ending")
    assert model.state_dict()["value_head.linear_ending.weight"].shape[0] == CLASSES
    plain = model_of("endingsgame", "regret")
    assert set(model.state_dict()) - set(plain.state_dict()) == {"value_head.linear_ending.weight",
                                                                 "value_head.linear_ending.bias"}
    # The trainer's parameter groups hold every parameter.
    groups = {}
    model.add_reg_dict(groups)
    grouped = {id(p) for group in groups.values() for p in group}
    assert [n for n, p in model.named_parameters() if id(p) not in grouped] == []


@pytest.mark.parametrize("game", ["enclosure", "blueprints", "plaingame"])
def test_a_game_without_classes_of_endings_refuses_the_head(game):
    config = modelconfigs.with_game(modelconfigs.config_of_name["b1c6nbt"], game)
    with pytest.raises(ValueError, match=f"'{game}'.*ENDING_CLASSES"):
        modelconfigs.with_heads(config, ("ending",))
    with pytest.raises(ValueError, match=f"'{game}'.*ENDING_CLASSES"):
        Model(dict(config, ending_head=True), games.get(game).MAX_BOARD_LEN)


def with_class_probabilities(model, probabilities):
    """The model with an ending head that gives each class its probability, at every position."""
    with torch.no_grad():
        model.value_head.linear_ending.weight.zero_()
        model.value_head.linear_ending.bias.copy_(torch.log(torch.tensor(probabilities)))
    return model


def test_the_heads_loss_is_the_cross_entropy_of_the_ending_at_its_weight():
    model = with_class_probabilities(model_of("endingsgame", "ending"), [0.4, 0.1, 0.2, 0.15, 0.1, 0.05])
    # A game that ended by class 0, one that ended by class 4, and one at the move cap, which has no ending.
    results = loss_sums(model, rows(model, n=3, ending=[0, 4, 0], ending_weight=[1, 1, 0]))
    assert results["endloss_sum"] == pytest.approx(-math.log(0.4) - math.log(0.1), rel=1e-5)
    assert results["endw_sum"] == 2.0
    assert results["endacc1_sum"] == 1.0  # the head's top class is class 0
    # A row's weight (C25) weighs it, as every loss.
    weighted = loss_sums(model, rows(model, n=3, ending=[0, 4, 0], ending_weight=[1, 1, 0], row_weight=[3, 1, 1]))
    assert weighted["endloss_sum"] == pytest.approx(-3 * math.log(0.4) - math.log(0.1), rel=1e-5)


@pytest.mark.parametrize("game, coefficient", [("endingsgame", 0.25), ("heaviergame", 2.0)])
def test_the_loss_sum_holds_the_heads_loss_times_its_coefficient(monkeypatch, game, coefficient):
    stub_game(monkeypatch, "heaviergame", games.get("endingsgame"), ENDING_CLASSES=CLASSES,
              LOSS_COEFFICIENTS={"ending": 2.0})
    model = with_class_probabilities(model_of(game, "ending"), [0.4, 0.1, 0.2, 0.15, 0.1, 0.05])
    likely, unlikely = (loss_sums(model, rows(model, ending=c)) for c in (0, 5))
    assert unlikely["endloss_sum"] - likely["endloss_sum"] == pytest.approx(2 * math.log(0.4 / 0.05), rel=1e-5)
    assert unlikely["loss_sum"] - likely["loss_sum"] == pytest.approx(coefficient * 2 * math.log(0.4 / 0.05), rel=1e-3)


def test_the_head_trains_the_trunk():
    model = model_of("endingsgame", "ending")
    batch = rows(model)
    outputs = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))[0]
    assert outputs[-1].shape == (2, CLASSES)
    torch.nn.functional.cross_entropy(outputs[-1], torch.tensor([0, 5])).backward()
    assert model.conv_spatial.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("game", ["endingsgame", "plaingame"])
def test_a_model_without_the_head_gives_no_ending_and_logs_none(game):
    model = model_of(game)
    batch = rows(model)
    outputs = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))[0]
    assert outputs[-1].shape == (2, 0)
    assert not [k for k in loss_sums(model, batch) if k.startswith("end")]


def test_c27_weighs_the_score_and_c0_alone_in_a_game_with_classes_of_endings():
    # A game that did not end by the score, such as a road of Tak: C27 is 0, and the other targets stay.
    results = sums("endingsgame", c27=0)
    assert all(results[k] == 0 for k in SCORE_LOSSES), results
    assert all(results[k] > 0 for k in OUTCOME_LOSSES + SLOT_LOSSES), results
    # KataGo's rule, in a game without classes: C27 weighs them all.
    results = sums("plaingame", c27=0)
    assert all(results[k] == 0 for k in SCORE_LOSSES + OUTCOME_LOSSES + SLOT_LOSSES), results


def test_a_row_with_every_target_trains_as_in_a_game_without_classes():
    assert sums("endingsgame") == sums("plaingame")
    # A game without classes does not read G76 and G77: Blueprints holds other facts there.
    assert sums("plaingame") == sums("plaingame", ending=-1, ending_weight=3)


def test_a_row_without_the_games_outcome_trains_none_of_its_targets():
    # A side position, and a reanalyzed row that leaves the outcome out: KataGo's C27 leaves them out of these losses.
    results = sums("endingsgame", "ending", c27=0, c68=0, c69=0, ending_weight=0)
    assert all(results[k] == 0 for k in SCORE_LOSSES + OUTCOME_LOSSES + SLOT_LOSSES + ("endloss_sum",)), results
    # A game at the move cap has no ending and no capture before its end (C1), but it has the next turns' (C5-6).
    results = sums("endingsgame", "ending", c27=0, c68=0, ending_weight=0)
    assert results["iloss_sum"] > 0
    assert all(results[k] == 0 for k in SCORE_LOSSES + OUTCOME_LOSSES + ("skloss_sum", "endloss_sum")), results


def test_the_values_weight_weighs_the_slots_c1_and_c5_6():
    # C35 is 1 minus the value's weight.
    results = sums("endingsgame", c27=0, no_value=0.25)
    assert (results["skw_sum"], results["iw_sum"]) == (1.5, 1.5)
    results = sums("plaingame", no_value=0.25)
    assert (results["skw_sum"], results["iw_sum"]) == (2.0, 2.0)  # KataGo's: C27


def test_the_optimistic_policies_read_the_scores_surprise_on_the_rows_with_a_score_alone():
    # The mover lost, and the score is far above what the network expects.
    lost = dict(win=0.0, loss=1.0, short_win=0.0, short_loss=1.0)
    surprise = dict(lost, score=1000.0, short_score=1000.0)
    with_score = sums("endingsgame", **surprise)
    assert (with_score["p0loptw_sum"], with_score["p0soptw_sum"]) == (2.0, 2.0)
    without = sums("endingsgame", c27=0, **surprise)
    assert without["p0loptw_sum"] == 0 and without["p0soptw_sum"] < 0.1
    assert without["p0soptw_sum"] == sums("endingsgame", c27=0, **lost)["p0soptw_sum"]
    # A win weighs a row whatever its score, and a draw as a quarter, the square of its half win.
    assert sums("endingsgame", c27=0)["p0loptw_sum"] == 2.0
    draw = sums("endingsgame", c27=0, win=0.0, no_result=1.0)
    assert draw["p0loptw_sum"] == pytest.approx(2 * 0.25)


def test_the_exported_network_has_no_head(tmp_path):
    model = model_of("endingsgame", "ending", kind="b6c96")
    state = model.state_dict()
    with_head = export_header(tmp_path / "head", {"model": state, "config": model.config})
    plain = export_header(tmp_path / "plain", {
        "model": {k: v for k, v in state.items() if "linear_ending" not in k},
        "config": {k: v for k, v in model.config.items() if k != "ending_head"}})
    assert with_head == plain


def test_the_loss_groups_add_up_with_the_head():
    model = model_of("endingsgame", "ending")
    model.train()
    metrics = Metrics(1, model)
    lines = dict(loss_gradients.measure(model, metrics, rows(model), SCALES))  # it checks the sum
    per_row, trunk = lines["how the game ends"][:2]
    assert per_row > 0 and trunk > 0
    # A model without the head has no such group.
    plain = model_of("endingsgame")
    plain.train()
    assert "how the game ends" not in dict(loss_gradients.measure(plain, Metrics(1, plain), rows(plain), SCALES))
