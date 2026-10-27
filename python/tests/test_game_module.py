"""The trainer's games (katago/train/games): a model config names its game, the game's module gives the model and the
losses its facts, and the export writes the game into the model file (model option H, neuralnet/desc.cpp): 0 for
Enclosure, another game's name for another game."""
import os
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# export_model_pytorch parses its command line at import.
sys.argv = [sys.argv[0], "-export-dir", "/nonexistent", "-model-name", "x", "-filename-prefix", "x"]
import export_model_pytorch as export  # noqa: E402
import shuffle  # noqa: E402
from katago.train import games, modelconfigs  # noqa: E402
from katago.train.data_processing_pytorch import (  # noqa: E402
    GLOBAL_TARGETS_NC_CHANNELS, VALUE_TARGETS_NCHW_CHANNELS, read_npz_training_data)
from katago.train.games import enclosure  # noqa: E402
from katago.train.metrics_pytorch import Metrics  # noqa: E402
from katago.train.model_pytorch import Model, policy_len  # noqa: E402

OPTION_H = 18  # model option H's line in the header (as in the engine's games/enclosure/tests/test_model_game.py)


@pytest.fixture
def game_without_ownership(monkeypatch):
    """A game with Enclosure's facts that writes every target but the ownership."""
    stub = types.ModuleType("katago.train.games.stubgame")
    stub.__dict__.update({m: getattr(enclosure, m) for m in games.MEMBERS})
    stub.NAME = "stubgame"
    stub.TARGETS = enclosure.TARGETS & ~games.TARGET_OWNERSHIP
    monkeypatch.setitem(sys.modules, "katago.train.games.stubgame", stub)
    return stub


def test_enclosure_gives_its_inputs_policy_and_scales():
    game = games.get("enclosure")
    config = modelconfigs.config_of_name["b6c96"]
    assert modelconfigs.get_num_bin_input_features(config) == 215
    assert modelconfigs.get_num_global_input_features(config) == 5
    assert modelconfigs.get_num_global_input_features(dict(config, inputs_version=9)) == 7
    assert (game.MAX_BOARD_LEN, game.POLICY_CHANNELS, game.SCORE_SCALE) == (19, 48, 12.0)
    assert 1 + game.HISTORY_PLANE + game.NUM_HISTORY_PLANES == 215
    assert games.symmetries(game) == list(range(8))


def test_connect4_gives_its_inputs_policy_and_scales():
    game = games.get("connect4")
    config = modelconfigs.with_game(modelconfigs.config_of_name["b6c96"], "connect4")
    assert modelconfigs.get_num_bin_input_features(config) == 4
    assert modelconfigs.get_num_global_input_features(config) == 1
    assert modelconfigs.get_num_global_input_features(dict(config, inputs_version=9)) == 3
    assert (game.MAX_BOARD_LEN, game.POLICY_CHANNELS, game.SCORE_SCALE) == (7, 1, 1.0)
    assert game.NUM_HISTORY_PLANES == 0
    assert games.symmetries(game) == [0, 2]
    assert game.TARGETS == games.TARGET_OWNERSHIP


@pytest.mark.parametrize("name, scale", [("enclosure", 12.0), ("connect4", 1.0), ("blueprints", 20.0)])
def test_the_score_multipliers_and_bins_follow_the_games_score_scale(name, scale):
    # KataGo's multipliers in points scale by s (the squared error's by s squared), and the score bins are s points
    # wide. Blueprints' s: the median |margin| of people's games, 409 points, over KataGo's score multiplier of 20.
    game = games.get(name)
    assert game.SCORE_SCALE == scale
    model = Model(modelconfigs.with_game(modelconfigs.config_of_name["b6c96"], name), game.MAX_BOARD_LEN)
    multipliers = (model.td_score_multiplier, model.scoremean_multiplier, model.scorestdev_multiplier,
                   model.lead_multiplier, model.shortterm_score_error_multiplier)
    assert multipliers == (20.0 * scale,) * 4 + (150.0 * scale ** 2,)
    centres = model.value_head.score_belief_offset_vector
    assert float(centres[1] - centres[0]) == scale


def test_connect4s_mirror_moves_each_point_to_its_column_mirror():
    # A row of the standard board, 7 x 6 on the network's 7 x 7: the policy at (x, y) moves to (6 - x, y), pass stays.
    game = games.get("connect4")
    policy = torch.arange(2 * 50, dtype=torch.float32).view(1, 2, 50)
    inputs = torch.rand(1, 4, 7, 7)
    image = game.apply_symmetry_policy_targets(policy, 2, inputs, None, None, None)
    for c in range(2):
        for y in range(7):
            for x in range(7):
                assert image[0, c, y * 7 + (6 - x)] == policy[0, c, y * 7 + x]
        assert image[0, c, 49] == policy[0, c, 49]
    assert torch.equal(game.apply_symmetry_inputs(inputs, 2), inputs.flip(-1))
    assert game.apply_symmetry_policy_targets(policy, 0, inputs, None, None, None) is policy


def test_a_row_has_the_bytes_of_the_row_writers_arrays():
    # A Go row as KataGo's shuffle.py measures it (19 x 19, 22 input planes, 19 global inputs, no playout inputs), with
    # the fork's 3 more value-target planes (8 against KataGo's 5).
    go = types.SimpleNamespace(MAX_BOARD_LEN=19, POLICY_CHANNELS=1, NUM_SPATIAL_INPUTS=21, NUM_GLOBAL_INPUTS=19)
    go_row = shuffle.UNCOMPRESSED_BYTES_PER_ROW_REQUIRED_19 + shuffle.UNCOMPRESSED_BYTES_PER_ROW_QVALUES_19
    assert go_row == games.GO_ROW_BYTES
    assert games.row_bytes(go, inputs_version=8) == go_row + (VALUE_TARGETS_NCHW_CHANNELS - 5) * 19 * 19
    assert GLOBAL_TARGETS_NC_CHANNELS == games.ROW_GLOBAL_TARGETS
    assert modelconfigs.PLAYOUT_GLOBAL_INPUTS == games.ROW_PLAYOUT_INPUTS
    # Connect-4's V9 rows of its first run's check: 1,470 bytes (games/connect4/docs/decisions/training.md).
    assert games.row_bytes(games.get("connect4")) == 1470
    # Enclosure's rows, 186,167 bytes when measured (2026-10-02: V8, 5 value-target planes), with the 3 planes and the
    # 2 playout inputs added since.
    assert games.row_bytes(games.get("enclosure")) == 186167 + 3 * 19 * 19 + 2 * 4


def test_a_game_without_a_module_is_refused():
    with pytest.raises(ValueError, match="no module for the game 'nogame'"):
        games.get("nogame")
    with pytest.raises(ValueError, match="nogame"):
        modelconfigs.with_game(modelconfigs.config_of_name["b6c96"], "nogame")


def test_a_module_that_lacks_a_member_is_refused(monkeypatch):
    stub = types.ModuleType("katago.train.games.partial")
    stub.NAME = "partial"
    monkeypatch.setitem(sys.modules, "katago.train.games.partial", stub)
    with pytest.raises(ValueError, match="lacks MAX_BOARD_LEN"):
        games.get("partial")


def test_a_config_names_its_game_and_one_without_is_enclosure():
    config = modelconfigs.config_of_name["b6c96"]
    assert "game" not in config and modelconfigs.get_game_name(config) == "enclosure"
    assert modelconfigs.with_game(config, None)["game"] == "enclosure"
    assert "game" not in config  # a copy


def test_the_network_board_is_the_games_largest_board():
    with pytest.raises(ValueError, match="pos_len 9, but the game enclosure's board is 19"):
        Model(modelconfigs.config_of_name["b6c96"], 9)


def export_random_model(tmp_path, game=None):
    args = dict(export.args, export_dir=str(tmp_path), model_name="t", filename_prefix="model",
                export_random_initialized_model="b6c96", game=game)
    export.main(args)
    return (tmp_path / "model.bin").read_bytes().split(b"\n", OPTION_H + 1)


def test_the_export_writes_0_for_enclosure_and_another_games_name_into_option_h(tmp_path, game_without_ownership):
    header = export_random_model(tmp_path / "enclosure")
    assert header[1] == b"17" and header[OPTION_H] == b"0"
    assert export_random_model(tmp_path / "stub", game="stubgame")[OPTION_H] == b"stubgame"
    assert export_random_model(tmp_path / "connect4", game="connect4")[OPTION_H] == b"connect4"


def random_batch(model, n=2):
    torch.manual_seed(0)
    size = model.pos_len
    binary = (torch.rand(n, modelconfigs.get_num_bin_input_features(model.config), size, size) < 0.2).float()
    binary[:, 0] = 1.0
    policy = torch.rand(n, 2, policy_len(size, model.policy_head.policy_channels))
    global_targets = torch.zeros(n, GLOBAL_TARGETS_NC_CHANNELS)
    global_targets[:, 0] = 1.0  # a win
    global_targets[:, [25, 26, 27, 28, 33, 34, 68, 69]] = 1.0  # the global weight and each target's weight or flag
    score = torch.zeros(n, model.value_head.scorebelief_len)
    score[:, model.value_head.scorebelief_mid] = 100.0
    value_targets = torch.randint(-1, 2, (n, VALUE_TARGETS_NCHW_CHANNELS, size, size)).float()
    global_inputs = torch.randn(n, modelconfigs.get_num_global_input_features(model.config))
    return {"binaryInputNCHW": binary, "globalInputNC": global_inputs, "policyTargetsNCMove": policy,
            "globalTargetsNC": global_targets, "scoreDistrN": score, "valueTargetsNCHW": value_targets}


def losses(game_name, flagged=1.0):
    """Every loss sum of a random b6c96 of the game on a random batch of 2 rows (random_batch), with the flag C68 of C1
    and the weight C34 of C4 at `flagged`."""
    torch.manual_seed(1)
    model = Model(modelconfigs.with_game(modelconfigs.config_of_name["b6c96"], game_name),
                  games.get(game_name).MAX_BOARD_LEN)
    model.initialize()
    batch = random_batch(model)
    batch["globalTargetsNC"][:, [34, 68]] = flagged
    with torch.no_grad():
        outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
        metrics = Metrics(1, model).metrics_dict_batchwise(
            model, model.postprocess_output(outputs), None, batch, is_training=False,
            soft_policy_weight_scale=8.0, disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
            value_loss_scale=0.6, td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=1.0,
            variance_time_loss_scale=1.0, main_loss_scale=None, intermediate_loss_scale=None)
    return {k: float(v) for k, v in metrics.items() if k.endswith("_sum")}


def spatial_losses(game_name):
    out = losses(game_name)
    return {k: out[k] for k in ("oloss_sum", "skloss_sum", "fploss_sum", "sloss_sum", "iloss_sum", "smloss_sum")}


def test_a_game_without_the_ownership_target_trains_no_ownership(game_without_ownership):
    # C27 weighs C0 and the score's targets, so the rows cannot leave out C0 by its weight: the trainer does.
    full = spatial_losses("enclosure")
    assert all(v > 0 for v in full.values()), full
    without = spatial_losses("stubgame")
    assert without["oloss_sum"] == 0
    assert all(without[k] == full[k] for k in full if k != "oloss_sum"), (without, full)


def test_a_game_without_mask_targets_keeps_the_weights():
    global_targets = torch.ones(2, GLOBAL_TARGETS_NC_CHANNELS)
    assert not hasattr(enclosure, "mask_targets")
    assert games.mask_targets(enclosure, None, None, None, global_targets) is global_targets


def write_blueprints_rows(path, version):
    """Four Blueprints rows as self-play writes them, empty but for what the second policy target's mask reads: a pick
    waits for its drop (global input 9) or not, and the next mover is the row's mover (+1) or not (-1). Rows of the
    targets' version 1 hold the next mover in G76 (and the version in G77), older rows (version 0) in the slot C1.
    Every other global target is 1. Returns the global targets."""
    game = games.get("blueprints")
    points = game.MAX_BOARD_LEN ** 2
    moves = policy_len(game.MAX_BOARD_LEN, game.POLICY_CHANNELS)
    picked, next_mover = [0, 0, 1, 1], [1, -1, 1, -1]
    global_inputs = np.zeros((4, game.NUM_GLOBAL_INPUTS), dtype=np.float32)
    global_inputs[:, game.PICKED_INPUT] = picked
    value_targets = np.zeros((4, VALUE_TARGETS_NCHW_CHANNELS, game.MAX_BOARD_LEN, game.MAX_BOARD_LEN), dtype=np.int8)
    global_targets = np.ones((4, GLOBAL_TARGETS_NC_CHANNELS), dtype=np.float32)
    global_targets[:, game.TARGETS_VERSION_TARGET] = version
    if version >= 1:
        global_targets[:, game.NEXT_MOVER_TARGET] = next_mover
    else:
        value_targets[:, game.NEXT_MOVER_SLOT] = np.array(next_mover, dtype=np.int8)[:, None, None]
    np.savez_compressed(
        path,
        binaryInputNCHWPacked=np.zeros((4, 1 + game.NUM_SPATIAL_INPUTS, (points + 7) // 8), dtype=np.uint8),
        globalInputNC=global_inputs,
        policyTargetsNCMove=np.zeros((4, 2, moves), dtype=np.int16),
        globalTargetsNC=global_targets,
        scoreDistrN=np.zeros((4, 2 * points + 2 * games.ROW_EXTRA_SCORE_RADIUS), dtype=np.int8),
        valueTargetsNCHW=value_targets,
        qValueTargetsNCMove=np.zeros((4, games.ROW_QVALUE_TARGETS, moves), dtype=np.int16),
    )
    return torch.from_numpy(global_targets)


@pytest.mark.parametrize("version", [0, 1])
@pytest.mark.parametrize("randomize_symmetries", [False, True])
def test_blueprints_rows_of_a_drop_with_the_same_mover_next_have_no_next_policy_weight(tmp_path, randomize_symmetries,
                                                                                       version):
    # The trainer cannot map their second policy target under a symmetry (katago/train/games/blueprints.py), so its
    # weight, C28, is 0 in every batch, with or without the symmetries, in the rows of each version of the targets.
    # Every other weight stays.
    path = tmp_path / "rows.npz"
    written = write_blueprints_rows(path, version)
    config = modelconfigs.with_game(modelconfigs.config_of_name["b6c96"], "blueprints")
    batches = list(read_npz_training_data([str(path)], batch_size=4, world_size=1, rank=0, pos_len=67, device="cpu",
                                          randomize_symmetries=randomize_symmetries, include_meta=False,
                                          model_config=config))
    assert len(batches) == 1
    weights = batches[0]["globalTargetsNC"]
    assert weights[:, 28].tolist() == [1.0, 1.0, 0.0, 1.0]
    others = [c for c in range(GLOBAL_TARGETS_NC_CHANNELS) if c != 28]
    assert torch.equal(weights[:, others], written[:, others])


def test_blueprints_next_mover_is_g76_in_new_rows_and_c1_in_old_rows():
    game = games.get("blueprints")
    value_targets = torch.zeros(3, VALUE_TARGETS_NCHW_CHANNELS, 67, 67)
    value_targets[:, game.NEXT_MOVER_SLOT] = torch.tensor([1.0, -1.0, 1.0])[:, None, None]
    global_targets = torch.zeros(3, GLOBAL_TARGETS_NC_CHANNELS)
    global_targets[:, game.NEXT_MOVER_TARGET] = torch.tensor([-1.0, 1.0, 0.0])
    global_targets[1:, game.TARGETS_VERSION_TARGET] = 1.0  # the first row is an old row
    assert game.next_movers(value_targets, global_targets).tolist() == [1.0, 1.0, 0.0]
    numpy = game.next_movers(value_targets.numpy(), global_targets.numpy())
    assert numpy.tolist() == [1.0, 1.0, 0.0]


def test_a_module_that_sets_an_unknown_loss_coefficient_is_refused(monkeypatch):
    stub = types.ModuleType("katago.train.games.badcoefficients")
    stub.__dict__.update({m: getattr(enclosure, m) for m in games.MEMBERS})
    stub.NAME = "badcoefficients"
    stub.LOSS_COEFFICIENTS = {"ownership": 0.5, "owner": 0.5}
    monkeypatch.setitem(sys.modules, "katago.train.games.badcoefficients", stub)
    with pytest.raises(ValueError, match="sets the coefficient of no loss named owner"):
        games.get("badcoefficients")
    assert games.loss_coefficients(enclosure) == {"ownership": 1.5, "seki": 1.0, "income": 1.5}


def test_every_game_has_katagos_and_enclosures_coefficients_and_blueprints_reads_361_points():
    # KataGo's ownership 1.5, Enclosure's 1 for the seki slot and 1.5 for the income slot (katago/train/games/__init__.py).
    for name in ("enclosure", "connect4", "blueprints"):
        assert games.loss_coefficients(games.get(name)) == {"ownership": 1.5, "seki": 1.0, "income": 1.5}
    assert games.loss_area(games.get("enclosure")) is None and games.loss_area(games.get("connect4")) is None
    assert games.loss_area(games.get("blueprints")) == 361.0


def metrics_of(game_name):
    """Metrics of a random b6c96 of the game."""
    torch.manual_seed(2)
    model = Model(modelconfigs.with_game(modelconfigs.config_of_name["b6c96"], game_name),
                  games.get(game_name).MAX_BOARD_LEN)
    return Metrics(1, model)


def spatial_rows(size, board):
    """Two rows on a network's board of size x size whose on-board points are the board x board corner, with random
    predictions and targets in [-1, 1]: (pred, target, weight, mask, mask_sum_hw, global_weight)."""
    torch.manual_seed(3)
    pred, target = torch.randn(2, 1, size, size), torch.randint(-1, 2, (2, size, size)).float()
    mask = torch.zeros(2, size, size)
    mask[:, :board[1], :board[0]] = 1.0
    return pred, target, torch.ones(2), mask, mask.sum(dim=(1, 2)), torch.tensor([1.0, 0.5])


# KataGo's area of each game's spatial losses: the on-board points of the row (Connect-4: 7 x 6 of its 7 x 7), or the
# game's LOSS_AREA (Blueprints: 361 of its view's 67 x 67).
LOSS_AREAS = [("connect4", (7, 6), 42.0), ("blueprints", (67, 67), 361.0)]


@pytest.mark.parametrize("game_name,board,area", LOSS_AREAS)
def test_the_ownership_loss_sums_the_points_over_the_games_loss_area(game_name, board, area):
    # KataGo's ownership loss: 1.5 times the cross-entropy of each on-board point, summed over the row, over the area.
    pred, target, weight, mask, mask_sum_hw, global_weight = spatial_rows(games.get(game_name).MAX_BOARD_LEN, board)
    loss = metrics_of(game_name).loss_ownership_samplewise(pred, target, weight, mask, mask_sum_hw, global_weight)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(2.0 * pred[:, 0], (1.0 + target) / 2.0, reduction="none")
    assert torch.allclose(loss, 1.5 * global_weight * (bce * mask).sum(dim=(1, 2)) / area)


@pytest.mark.parametrize("game_name,board,area", LOSS_AREAS)
def test_the_future_positions_loss_sums_the_points_over_the_square_root_of_the_loss_area(game_name, board, area):
    # KataGo's: 0.25 times the squared errors of each on-board point, the far horizon's times 0.25, summed over the row,
    # over the square root of the area ("let's compromise and scale by sqrt(boardarea)").
    pred, target, weight, mask, mask_sum_hw, global_weight = spatial_rows(games.get(game_name).MAX_BOARD_LEN, board)
    pred, target = torch.cat((pred, -pred), dim=1), torch.stack((target, -target), dim=1)
    loss = metrics_of(game_name).loss_futurepos_samplewise(pred, target, weight, mask, mask_sum_hw, global_weight)
    squares = torch.square(torch.tanh(pred) - target) * mask.unsqueeze(1)
    expected = 0.25 * global_weight * (squares[:, 0] + 0.25 * squares[:, 1]).sum(dim=(1, 2)) / area ** 0.5
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize("game_name,board,area", LOSS_AREAS)
def test_the_scoring_loss_is_katagos_transform_of_the_squared_errors_over_the_loss_area(game_name, board, area):
    # KataGo's: the squared errors of each on-board point, summed over the row, over the area: x. Then its "huberlike
    # transform", 4 (sqrt(x / 2 + 1) - 1). The coefficient 0.25 is in the total (metrics_pytorch, loss_sum).
    pred, target, weight, mask, mask_sum_hw, global_weight = spatial_rows(games.get(game_name).MAX_BOARD_LEN, board)
    loss = metrics_of(game_name).loss_scoring_samplewise(pred, target, weight, mask, mask_sum_hw, global_weight)
    x = (torch.square(pred[:, 0] - target) * mask).sum(dim=(1, 2)) / area
    assert torch.allclose(loss, global_weight * 4.0 * (torch.sqrt(x / 2.0 + 1.0) - 1.0))


@pytest.mark.parametrize("game_name,board,area", LOSS_AREAS)
def test_the_income_loss_without_a_mask_sums_both_channels_over_twice_the_loss_area(game_name, board, area):
    # Enclosure's: the cross-entropy of each on-board point of both channels, summed over the row, over twice the area,
    # at the coefficient 1.5. Blueprints passes its income_mask (test_blueprints_income_loss_averages_each_channel...).
    pred, target, weight, mask, mask_sum_hw, global_weight = spatial_rows(games.get(game_name).MAX_BOARD_LEN, board)
    logits, probs = torch.cat((pred, -pred), dim=1), torch.stack(((1.0 + target) / 2.0, (1.0 - target) / 2.0), dim=1)
    loss = metrics_of(game_name).loss_income_samplewise(logits, probs, weight, mask, mask_sum_hw, global_weight)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, probs, reduction="none") * mask.unsqueeze(1)
    assert torch.allclose(loss, 1.5 * global_weight * bce.sum(dim=(1, 2, 3)) / (2.0 * area))


def blueprints_losses(monkeypatch, coefficients, flagged):
    """Blueprints' losses(), with the loss coefficients `coefficients` and C68 and C34 at `flagged`."""
    monkeypatch.setattr(games.get("blueprints"), "LOSS_COEFFICIENTS", coefficients, raising=False)
    return losses("blueprints", flagged)


def test_blueprints_rows_of_the_targets_versions_0_and_1_train_no_c1_or_c4(monkeypatch):
    # The rows of the targets' versions 0 and 1 have C68 = C34 = 0, so no loss reads their C1 (the next mover in version
    # 0) or their C4, whatever C1's coefficient. The rows of version 2 have both, and C1's loss follows its coefficient.
    old_rows, new_rows = losses("blueprints", 0.0), losses("blueprints", 1.0)
    assert old_rows["skloss_sum"] == old_rows["sloss_sum"] == 0.0
    assert new_rows["skloss_sum"] > 0 and new_rows["sloss_sum"] > 0
    old_rows_half = blueprints_losses(monkeypatch, {"seki": 0.5}, 0.0)
    new_rows_half = blueprints_losses(monkeypatch, {"seki": 0.5}, 1.0)
    assert old_rows == old_rows_half
    assert new_rows_half["skloss_sum"] == pytest.approx(0.5 * new_rows["skloss_sum"], rel=1e-6)
    assert all(new_rows[k] == new_rows_half[k] for k in new_rows if k not in ("skloss_sum", "loss_sum")), new_rows


def test_blueprints_income_loss_averages_each_channel_over_its_cells():
    # Row 0: cells that can go in both channels. Row 1: in C6 only, so its loss is C6's mean alone. Row 2: none.
    # Enclosure's coefficient, 1.5.
    metrics = metrics_of("blueprints")
    torch.manual_seed(4)
    pred, target = torch.randn(3, 2, 67, 67), (torch.rand(3, 2, 67, 67) < 0.3).float()
    income_mask = torch.zeros(3, 2, 67, 67)
    income_mask[0, 0, 10:12, 10:15] = 1.0
    income_mask[0, 1, 20, 30:33] = 1.0
    income_mask[1, 1, 5:9, 5] = 1.0
    mask, weight, global_weight = torch.ones(3, 67, 67), torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 2.0, 1.0])
    loss = metrics.loss_income_samplewise(pred, target, weight, mask, mask.sum(dim=(1, 2)), global_weight, income_mask)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
    mean = lambda r, c: bce[r, c][income_mask[r, c] > 0].mean()  # noqa: E731
    expected = torch.stack((1.5 * (mean(0, 0) + mean(0, 1)) / 2, 1.5 * 2.0 * mean(1, 1), torch.tensor(0.0)))
    assert torch.allclose(loss, expected)
    # The mask of a batch's rows: the cells of the input planes, the opponent's (C5) and the mover's (C6), not walls.
    game = games.get("blueprints")
    inputs = torch.zeros(1, 1 + game.NUM_SPATIAL_INPUTS, 67, 67)
    for plane, (x, y) in zip(range(6), ((1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1))):
        inputs[0, 1 + plane, y, x] = 1.0  # the mover's plain, star and wall cells, then the opponent's
    got = game.income_mask(inputs)[0]
    assert got[0].nonzero().tolist() == [[1, 4], [1, 5]] and got[1].nonzero().tolist() == [[1, 1], [1, 2]]
