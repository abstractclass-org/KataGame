"""The trainer's games (katago/train/games): a model config names its game, the game's module gives the model and the
losses its facts, and the export writes the game into the model file (model option H, neuralnet/desc.cpp): 0 for
Enclosure, another game's name for another game."""
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# export_model_pytorch parses its command line at import.
sys.argv = [sys.argv[0], "-export-dir", "/nonexistent", "-model-name", "x", "-filename-prefix", "x"]
import export_model_pytorch as export  # noqa: E402
import shuffle  # noqa: E402
from katago.train import games, modelconfigs  # noqa: E402
from katago.train.data_processing_pytorch import GLOBAL_TARGETS_NC_CHANNELS, VALUE_TARGETS_NCHW_CHANNELS  # noqa: E402
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


def spatial_losses(game_name):
    torch.manual_seed(1)
    model = Model(modelconfigs.with_game(modelconfigs.config_of_name["b6c96"], game_name), 19)
    model.initialize()
    batch = random_batch(model)
    with torch.no_grad():
        outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
        metrics = Metrics(1, model).metrics_dict_batchwise(
            model, model.postprocess_output(outputs), None, batch, is_training=False,
            soft_policy_weight_scale=8.0, disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
            value_loss_scale=0.6, td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=1.0,
            variance_time_loss_scale=1.0, main_loss_scale=None, intermediate_loss_scale=None)
    return {k: float(metrics[k]) for k in ("oloss_sum", "skloss_sum", "fploss_sum", "sloss_sum", "iloss_sum",
                                           "smloss_sum")}


def test_a_game_without_the_ownership_target_trains_no_ownership(game_without_ownership):
    # C27 weighs C0 and the score's targets, so the rows cannot leave out C0 by its weight: the trainer does.
    full = spatial_losses("enclosure")
    assert all(v > 0 for v in full.values()), full
    without = spatial_losses("stubgame")
    assert without["oloss_sum"] == 0
    assert all(without[k] == full[k] for k in full if k != "oloss_sum"), (without, full)
