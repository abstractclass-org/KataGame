"""The inputs version 10: the game's extended inputs after the inputs version 9's (modelconfigs, katago/train/games), the
trainer's map of the extended global inputs, the export as model version 19, and add_extended_inputs.py, which gives a
checkpoint of version 9 the extended inputs as zero weights, so that its network computes what it did. The inputs
version 11: the game's turn inputs after version 10's global inputs, the trainer's maps and history dropout of its
rows, and the export as model version 20."""
import os
import sys

import pytest
import torch
from torch.optim.swa_utils import AveragedModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# export_model_pytorch parses its command line at import.
sys.argv = [sys.argv[0], "-export-dir", "/nonexistent", "-model-name", "x", "-filename-prefix", "x"]
import add_extended_inputs  # noqa: E402
import export_model_pytorch as export  # noqa: E402
from enclosure_add_regret_heads import group_names  # noqa: E402
from katago.train import games, modelconfigs  # noqa: E402
from katago.train.games import blueprints  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402

KIND = "b1c6nbt"  # the smallest model kind: the checks below read the first layers only


def config(game, inputs_version, kind="b6c96"):
    return dict(modelconfigs.with_game(modelconfigs.config_of_name[kind], game), inputs_version=inputs_version)


def test_version_10_adds_the_games_extended_inputs_after_version_9s():
    assert modelconfigs.get_num_bin_input_features(config("blueprints", 9)) == 90
    assert modelconfigs.get_num_global_input_features(config("blueprints", 9)) == 24
    assert modelconfigs.get_num_bin_input_features(config("blueprints", 10)) == 90 + 12
    assert modelconfigs.get_num_global_input_features(config("blueprints", 10)) == 24 + 20
    # A game without extended inputs: version 10 is version 9.
    assert games.extended_spatial_inputs(games.get("enclosure")) == 0
    assert modelconfigs.get_num_bin_input_features(config("enclosure", 10)) == 215
    assert modelconfigs.get_num_global_input_features(config("enclosure", 10)) == 7


def test_a_row_of_version_10_holds_the_extended_inputs():
    game = games.get("blueprints")
    points = game.MAX_BOARD_LEN ** 2
    assert games.row_bytes(game, 10) - games.row_bytes(game, 9) == 12 * ((points + 7) // 8) + 4 * 20
    assert games.row_bytes(games.get("enclosure"), 10) == games.row_bytes(games.get("enclosure"), 9)


def test_the_trainer_maps_only_the_extended_global_inputs_of_version_10_rows():
    rows = torch.randn(3, 22 + 2 + 20)
    out = games.apply_symmetry_global_inputs(blueprints, rows, 3)
    assert torch.equal(out[:, :24], rows[:, :24])
    assert torch.equal(out[:, 24:], blueprints.apply_symmetry_extended_global_inputs(rows[:, 24:], 3))
    assert not torch.equal(out, rows)
    assert games.apply_symmetry_global_inputs(blueprints, rows, 0) is rows
    v9 = rows[:, :24]
    assert games.apply_symmetry_global_inputs(blueprints, v9, 3) is v9
    enclosure_rows = torch.randn(3, 7)
    assert games.apply_symmetry_global_inputs(games.get("enclosure"), enclosure_rows, 3) is enclosure_rows


def test_the_window_boxes_map_as_boxes():
    # One box: x from 1 to 4, y from -2 to 3 (/ 67 left out). The turn by 180 degrees (3) negates and swaps the
    # corners, the transpose (4) swaps x and y, and the second reflection (7) does both.
    box = torch.tensor([[1.0, 1, 4, -2, 3] + [0.0] * 15])
    images = {s: blueprints.apply_symmetry_extended_global_inputs(box, s)[0, :5].tolist() for s in (3, 4, 7)}
    assert images == {3: [1, -4, -1, -3, 2], 4: [1, -2, 3, 1, 4], 7: [1, -3, 2, -4, -1]}
    for s in (3, 4, 7):  # each is its own inverse
        twice = blueprints.apply_symmetry_extended_global_inputs(blueprints.apply_symmetry_extended_global_inputs(box, s), s)
        assert torch.equal(twice, box)


def flat(outputs):
    if torch.is_tensor(outputs):
        return [outputs]
    if outputs is None:
        return []
    return [t for o in outputs for t in flat(o)]


def checkpoint(inputs_version=9):
    """A checkpoint of a random Blueprints model of the inputs version 9 (or another), as train.py saves one: the
    model, its SWA model and an SGD optimizer with momentum after one step, in train.get_param_groups' order."""
    torch.manual_seed(0)
    cfg = config("blueprints", inputs_version, KIND)
    model = Model(cfg, blueprints.MAX_BOARD_LEN)
    model.initialize()
    params = dict(model.named_parameters())
    optimizer = torch.optim.SGD([{"params": [params[n] for n in names]} for _, names in group_names(model)],
                                lr=0.01, momentum=0.9)
    binary, global_ = inputs(cfg, 2)
    sum(t.float().pow(2).mean() for t in flat(model(binary, global_))).backward()
    optimizer.step()
    swa = AveragedModel(model)
    return {"config": cfg, "model": model.state_dict(), "swa_model": swa.state_dict(),
            "optimizer": optimizer.state_dict()}


def inputs(cfg, n):
    size = blueprints.MAX_BOARD_LEN
    binary = (torch.rand(n, modelconfigs.get_num_bin_input_features(cfg), size, size) < 0.2).float()
    binary[:, 0] = 1.0
    return binary, torch.randn(n, modelconfigs.get_num_global_input_features(cfg))


def test_the_converted_network_computes_what_it_did_whatever_the_new_inputs():
    ckpt = checkpoint()
    old = Model(ckpt["config"], blueprints.MAX_BOARD_LEN)
    old.load_state_dict(ckpt["model"])
    old_swa = AveragedModel(Model(ckpt["config"], blueprints.MAX_BOARD_LEN))
    old_swa.load_state_dict(ckpt["swa_model"])
    touched = add_extended_inputs.convert(ckpt)
    assert len(touched) == 6, touched  # 2 weights in the model, the SWA model and the optimizer
    assert modelconfigs.get_inputs_version(ckpt["config"]) == 10
    new = Model(ckpt["config"], blueprints.MAX_BOARD_LEN)
    new.load_state_dict(ckpt["model"])
    new_swa = AveragedModel(Model(ckpt["config"], blueprints.MAX_BOARD_LEN))
    new_swa.load_state_dict(ckpt["swa_model"])
    binary, global_ = inputs(ckpt["config"], 3)
    # The rows of version 9 are the first planes and the first global inputs of version 10's.
    old_binary, old_global = binary[:, :90], global_[:, :24]
    old.eval(), new.eval(), old_swa.eval(), new_swa.eval()
    with torch.no_grad():
        for a, b in ((old, new), (old_swa, new_swa)):
            expected = flat(a(old_binary, old_global))
            for fill in (0.0, 1.0, None):  # the new inputs at 0, at 1, and random
                x, g = binary.clone(), global_.clone()
                if fill is not None:
                    x[:, 90:], g[:, 24:] = fill, fill
                got = flat(b(x, g))
                assert len(got) == len(expected)
                for e, o in zip(expected, got):
                    torch.testing.assert_close(o, e, rtol=1e-5, atol=1e-5)


def test_the_optimizer_state_gets_zero_columns_and_loads():
    ckpt = checkpoint()
    before = {i: {k: v.clone() for k, v in s.items()} for i, s in ckpt["optimizer"]["state"].items()}
    add_extended_inputs.convert(ckpt)
    new = Model(ckpt["config"], blueprints.MAX_BOARD_LEN)
    params = dict(new.named_parameters())
    optimizer = torch.optim.SGD([{"params": [params[n] for n in names]} for _, names in group_names(new)],
                                lr=0.01, momentum=0.9)
    optimizer.load_state_dict(ckpt["optimizer"])
    widened = 0
    for i, state in ckpt["optimizer"]["state"].items():
        for k, v in state.items():
            old = before[i][k]
            if v.shape != old.shape:
                widened += 1
                assert torch.equal(v.narrow(1, 0, old.shape[1]), old)
                assert not v.narrow(1, old.shape[1], v.shape[1] - old.shape[1]).any()
            else:
                assert torch.equal(v, old)
    assert widened == 2


def test_a_checkpoint_of_a_network_alone_converts():
    ckpt = checkpoint()
    alone = {"config": ckpt["config"], "model": ckpt["model"]}
    assert len(add_extended_inputs.convert(alone)) == 2
    Model(alone["config"], blueprints.MAX_BOARD_LEN).load_state_dict(alone["model"])


def test_the_converter_refuses_other_inputs():
    ckpt = checkpoint()
    add_extended_inputs.convert(ckpt)
    with pytest.raises(ValueError, match="version 10, not 9"):
        add_extended_inputs.convert(ckpt)
    with pytest.raises(ValueError, match="no extended inputs"):
        add_extended_inputs.convert({"config": config("enclosure", 9)})


def test_the_export_writes_model_version_19(tmp_path):
    ckpt = checkpoint()
    add_extended_inputs.convert(ckpt)
    torch.save(ckpt, tmp_path / "checkpoint.ckpt")
    args = dict(export.args, export_dir=str(tmp_path / "out"), model_name="t", filename_prefix="model",
                checkpoint=str(tmp_path / "checkpoint.ckpt"), export_random_initialized_model=None, game=None)
    export.main(args)
    header = (tmp_path / "out" / "model.bin").read_bytes().split(b"\n", 4)
    assert header[1] == b"19"
    assert (int(header[2]), int(header[3])) == (90 + 12, 24 + 20)  # the input channels, then the global ones


# The inputs version 11: Blueprints' 8 turn inputs after version 10's 44 global inputs.
V10_GLOBALS, TURNS = 24 + 20, 8


def test_version_11_adds_the_games_turn_inputs_after_version_10s():
    assert modelconfigs.get_num_bin_input_features(config("blueprints", 11)) == 90 + 12
    assert modelconfigs.get_num_global_input_features(config("blueprints", 11)) == V10_GLOBALS + TURNS
    game = games.get("blueprints")
    assert games.row_bytes(game, 11) - games.row_bytes(game, 10) == 4 * TURNS
    # A game without turn inputs: version 11 is version 10.
    assert games.turn_global_inputs(games.get("enclosure")) == 0
    assert modelconfigs.get_num_global_input_features(config("enclosure", 11)) == 7
    assert games.row_bytes(games.get("enclosure"), 11) == games.row_bytes(games.get("enclosure"), 10)


def test_the_trainer_maps_the_window_boxes_of_version_11_rows_and_not_the_turn_inputs():
    rows = torch.randn(3, V10_GLOBALS + TURNS)
    out = games.apply_symmetry_global_inputs(blueprints, rows, 3)
    boxes = rows[:, 24:V10_GLOBALS]
    assert torch.equal(out[:, :24], rows[:, :24])
    assert torch.equal(out[:, 24:V10_GLOBALS], blueprints.apply_symmetry_extended_global_inputs(boxes, 3))
    assert torch.equal(out[:, V10_GLOBALS:], rows[:, V10_GLOBALS:])


def dropout_before(game, binary):
    """The history dropout as it was before the inputs version 11 (data_processing_pytorch.apply_history_dropout)."""
    num_history = game.NUM_HISTORY_PLANES
    start = 1 + game.HISTORY_PLANE
    n = binary.shape[0]
    should_stop_history = torch.rand((n, num_history), device=binary.device) >= 0.98
    include_history = (torch.cumsum(should_stop_history, axis=1, dtype=torch.float32) <= 0.1).to(binary.dtype)
    out = binary.clone()
    out[:, start:start + num_history] *= include_history.view(n, num_history, 1, 1)
    return out


def test_the_history_dropout_drops_each_turns_inputs_with_its_planes():
    from katago.train.data_processing_pytorch import apply_history_dropout
    game = games.get("blueprints")
    n = 4000
    binary = torch.ones(n, 1 + game.NUM_SPATIAL_INPUTS + game.NUM_EXTENDED_SPATIAL_INPUTS, 1, 1)
    # The rows of version 10: the planes as before, from the same draws, and the global inputs as they are.
    v10 = torch.ones(n, V10_GLOBALS)
    torch.manual_seed(1)
    before = dropout_before(game, binary)
    torch.manual_seed(1)
    planes, globals_ = apply_history_dropout(game, binary, v10)
    assert torch.equal(planes, before) and globals_ is v10
    # The rows of version 11: turn input i with the history plane i (8 planes, 8 inputs), from the same draws.
    v11 = torch.ones(n, V10_GLOBALS + TURNS)
    torch.manual_seed(1)
    planes, globals_ = apply_history_dropout(game, binary, v11)
    assert torch.equal(planes, before)
    kept = planes[:, 1 + game.HISTORY_PLANE:1 + game.HISTORY_PLANE + game.NUM_HISTORY_PLANES, 0, 0]
    assert torch.equal(globals_[:, V10_GLOBALS:], kept)
    assert torch.equal(globals_[:, :V10_GLOBALS], v11[:, :V10_GLOBALS])
    assert 0 < int((kept == 0).any(dim=1).sum()) < n  # some rows drop history, most do not
    # A game without turn inputs: its global inputs as they are.
    enclosure = games.get("enclosure")
    rows = torch.ones(5, 7)
    assert apply_history_dropout(enclosure, torch.ones(5, 1 + enclosure.NUM_SPATIAL_INPUTS, 1, 1), rows)[1] is rows


def test_the_export_writes_model_version_20(tmp_path):
    torch.save(checkpoint(11), tmp_path / "checkpoint.ckpt")
    args = dict(export.args, export_dir=str(tmp_path / "out"), model_name="t", filename_prefix="model",
                checkpoint=str(tmp_path / "checkpoint.ckpt"), export_random_initialized_model=None, game=None)
    export.main(args)
    header = (tmp_path / "out" / "model.bin").read_bytes().split(b"\n", 4)
    assert header[1] == b"20"
    assert (int(header[2]), int(header[3])) == (90 + 12, V10_GLOBALS + TURNS)
