"""The inputs version 10: the game's extended inputs after the inputs version 9's (modelconfigs, katago/train/games), the
trainer's map of the extended global inputs, the export as model version 19, and add_extended_inputs.py, which gives a
checkpoint of version 9 the extended inputs as zero weights, so that its network computes what it did. The inputs
version 11: the game's turn inputs after version 10's global inputs, the trainer's maps and history dropout of its
rows, the export as model version 20, and add_extended_inputs.py's conversion of a checkpoint of version 10, with its
score factor for a new score scale."""
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
# The inputs version 11: Blueprints' 8 turn inputs after version 10's 44 global inputs.
V10_GLOBALS, TURNS = 24 + 20, 8
# Each conversion of add_extended_inputs.py: the checkpoint's inputs version, the version that it gives, and the
# weights that it widens.
CONVERSIONS = [(9, 10, 2), (10, 11, 1)]


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


def sgd(model):
    """An SGD optimizer with momentum over the model's parameters, in train.get_param_groups' order."""
    params = dict(model.named_parameters())
    return torch.optim.SGD([{"params": [params[n] for n in names]} for _, names in group_names(model)],
                           lr=0.01, momentum=0.9)


def checkpoint(inputs_version=9, kind=KIND):
    """A checkpoint of a random Blueprints model of the inputs version 9 (or another), as train.py saves one: the
    model, its SWA model and an SGD optimizer with momentum after one step, in train.get_param_groups' order."""
    torch.manual_seed(0)
    cfg = config("blueprints", inputs_version, kind)
    model = Model(cfg, blueprints.MAX_BOARD_LEN)
    model.initialize()
    optimizer = sgd(model)
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


def networks(ckpt):
    """The checkpoint's network and its SWA network, in eval mode."""
    model = Model(ckpt["config"], blueprints.MAX_BOARD_LEN)
    model.load_state_dict(ckpt["model"])
    swa = AveragedModel(Model(ckpt["config"], blueprints.MAX_BOARD_LEN))
    swa.load_state_dict(ckpt["swa_model"])
    return model.eval(), swa.eval()


def assert_same_outputs(olds, news, cfg, planes, global_inputs, outputs=lambda net, out: flat(out)):
    """The networks news, on rows of cfg's inputs, give what olds give on the rows' first planes and first
    global_inputs, whatever the inputs after these hold. outputs(net, out): what is compared (default: every output)."""
    binary, global_ = inputs(cfg, 3)
    with torch.no_grad():
        for a, b in zip(olds, news):
            expected = outputs(a, a(binary[:, :planes], global_[:, :global_inputs]))
            for fill in (0.0, 1.0, None):  # the new inputs at 0, at 1, and random
                x, g = binary.clone(), global_.clone()
                if fill is not None:
                    x[:, planes:], g[:, global_inputs:] = fill, fill
                got = outputs(b, b(x, g))
                assert len(got) == len(expected)
                for e, o in zip(expected, got):
                    torch.testing.assert_close(o, e, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("version, target, weights", CONVERSIONS)
def test_the_converted_network_computes_what_it_did_whatever_the_new_inputs(version, target, weights):
    ckpt = checkpoint(version)
    olds = networks(ckpt)
    touched = add_extended_inputs.convert(ckpt, target)
    assert len(touched) == 3 * weights, touched  # each widened weight in the model, the SWA model and the optimizer
    assert modelconfigs.get_inputs_version(ckpt["config"]) == target
    # The rows of the old version are the first planes and the first global inputs of the new version's. Version 11
    # keeps version 10's planes.
    planes, global_inputs = {9: (90, 24), 10: (90 + 12, V10_GLOBALS)}[version]
    assert_same_outputs(olds, networks(ckpt), ckpt["config"], planes, global_inputs)


@pytest.mark.parametrize("version, target, weights", CONVERSIONS)
def test_the_optimizer_state_gets_zero_columns_and_loads(version, target, weights):
    ckpt = checkpoint(version)
    before = {i: {k: v.clone() for k, v in s.items()} for i, s in ckpt["optimizer"]["state"].items()}
    add_extended_inputs.convert(ckpt, target)
    sgd(Model(ckpt["config"], blueprints.MAX_BOARD_LEN)).load_state_dict(ckpt["optimizer"])
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
    assert widened == weights


@pytest.mark.parametrize("version, target, weights", CONVERSIONS)
def test_a_checkpoint_of_a_network_alone_converts(version, target, weights):
    ckpt = checkpoint(version)
    alone = {"config": ckpt["config"], "model": ckpt["model"]}
    assert len(add_extended_inputs.convert(alone, target)) == weights
    Model(alone["config"], blueprints.MAX_BOARD_LEN).load_state_dict(alone["model"])


def test_the_converter_refuses_other_inputs():
    ckpt = checkpoint()
    with pytest.raises(ValueError, match="version 9, not 10"):
        add_extended_inputs.convert(ckpt, 11)
    add_extended_inputs.convert(ckpt, 10)
    with pytest.raises(ValueError, match="version 10, not 9"):
        add_extended_inputs.convert(ckpt, 10)
    with pytest.raises(ValueError, match="this gives 10 or 11 or 12"):
        add_extended_inputs.convert(ckpt, 13)
    for version, target, _ in CONVERSIONS:  # a game without the target's new inputs
        with pytest.raises(ValueError, match=f"no inputs new in version {target}"):
            add_extended_inputs.convert({"config": config("enclosure", version)}, target)


def test_the_export_writes_model_version_19(tmp_path):
    ckpt = checkpoint()
    add_extended_inputs.convert(ckpt, 10)
    torch.save(ckpt, tmp_path / "checkpoint.ckpt")
    args = dict(export.args, export_dir=str(tmp_path / "out"), model_name="t", filename_prefix="model",
                checkpoint=str(tmp_path / "checkpoint.ckpt"), export_random_initialized_model=None, game=None)
    export.main(args)
    header = (tmp_path / "out" / "model.bin").read_bytes().split(b"\n", 4)
    assert header[1] == b"19"
    assert (int(header[2]), int(header[3])) == (90 + 12, 24 + 20)  # the input channels, then the global ones


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


def test_the_history_dropout_drops_each_turn_with_its_planes_and_its_turn_inputs():
    from katago.train.data_processing_pytorch import apply_history_dropout
    game = games.get("blueprints")
    n = 4000
    binary = torch.ones(n, 1 + game.NUM_SPATIAL_INPUTS + game.NUM_EXTENDED_SPATIAL_INPUTS, 1, 1)
    history = slice(1 + game.HISTORY_PLANE, 1 + game.HISTORY_PLANE + game.NUM_HISTORY_PLANES)
    # The rows of version 10: a draw for each plane, as before, and the global inputs as they are.
    v10 = torch.ones(n, V10_GLOBALS)
    torch.manual_seed(1)
    before = dropout_before(game, binary)
    torch.manual_seed(1)
    planes, globals_ = apply_history_dropout(game, binary, v10, 10)
    assert torch.equal(planes, before) and globals_ is v10
    # The rows of version 11: a draw for each of the 4 turns, which clears the turn's 2 planes and its 2 turn inputs
    # together, and never a newer turn without the older ones.
    v11 = torch.ones(n, V10_GLOBALS + TURNS)
    torch.manual_seed(1)
    planes, globals_ = apply_history_dropout(game, binary, v11, 11)
    kept = planes[:, history, 0, 0].reshape(n, 4, 2)  # turn, then its stamped and taken planes
    assert torch.equal(kept[:, :, 0], kept[:, :, 1])
    assert torch.equal(globals_[:, V10_GLOBALS:].reshape(n, 4, 2), kept)  # turn, then its moves' edit flags
    torch.manual_seed(1)
    stops = torch.rand((n, 4)) >= 0.98
    assert torch.equal(kept[:, :, 0], (torch.cumsum(stops, axis=1, dtype=torch.float32) <= 0.1).float())
    assert 0 < int((kept == 0).any(dim=(1, 2)).sum()) < n  # some rows drop history, most do not
    others = torch.ones(planes.shape[1], dtype=torch.bool)
    others[history] = False
    assert torch.equal(planes[:, others], binary[:, others])
    assert torch.equal(globals_[:, :V10_GLOBALS], v11[:, :V10_GLOBALS])
    # A game with turn inputs names its turns, which divide its planes and its turn inputs.
    for turns in (None, 3):
        with pytest.MonkeyPatch.context() as patch:
            if turns is None:
                patch.delattr(blueprints, "NUM_HISTORY_TURNS")
            else:
                patch.setattr(blueprints, "NUM_HISTORY_TURNS", turns)
            with pytest.raises(ValueError, match="NUM_HISTORY_TURNS"):
                games.get("blueprints")
    # A game without turn inputs: its global inputs as they are.
    enclosure = games.get("enclosure")
    rows = torch.ones(5, 7)
    assert apply_history_dropout(enclosure, torch.ones(5, 1 + enclosure.NUM_SPATIAL_INPUTS, 1, 1), rows, 11)[1] is rows


def test_the_export_writes_model_version_20(tmp_path):
    torch.save(checkpoint(11), tmp_path / "checkpoint.ckpt")
    args = dict(export.args, export_dir=str(tmp_path / "out"), model_name="t", filename_prefix="model",
                checkpoint=str(tmp_path / "checkpoint.ckpt"), export_random_initialized_model=None, game=None)
    export.main(args)
    header = (tmp_path / "out" / "model.bin").read_bytes().split(b"\n", 4)
    assert header[1] == b"20"
    assert (int(header[2]), int(header[3])) == (90 + 12, V10_GLOBALS + TURNS)


# The score factor: Blueprints' SCORE_SCALE from 12 to 20 multiplies the multipliers of its outputs in points by
# 20 / 12.
FACTOR = 12.0 / 20.0


def points(net, outputs):
    """The network's outputs as the trainer reads them (Model.postprocess_output), the scores in points."""
    model = net.module if isinstance(net, AveragedModel) else net
    return flat(model.postprocess_output(outputs))


@pytest.mark.parametrize("kind, heads", [(KIND, 1), (KIND + "-ih", 2)])  # -ih: with an intermediate head
def test_the_score_factor_keeps_the_outputs_in_points_under_the_new_multipliers(kind, heads):
    ckpt = checkpoint(10, kind)
    olds = networks(ckpt)
    add_extended_inputs.convert(ckpt, 11)
    touched = add_extended_inputs.scale_scores(ckpt, FACTOR)
    # The weight and the bias of 2 layers of each value head, in the model, the SWA model and the optimizer.
    assert len(touched) == 3 * 4 * heads, touched
    news = networks(ckpt)
    # The trainer of the new score scale: the multipliers of the score mean, the lead and the TD scores over FACTOR.
    for net in news:
        model = net.module if isinstance(net, AveragedModel) else net
        for name in ("scoremean_multiplier", "lead_multiplier", "td_score_multiplier"):
            setattr(model, name, getattr(model, name) / FACTOR)
    assert_same_outputs(olds, news, ckpt["config"], 90 + 12, V10_GLOBALS, points)


def test_the_score_factor_scales_the_momentum_of_its_rows_and_it_loads():
    ckpt = checkpoint(10)
    before = {i: {k: v.clone() for k, v in s.items()} for i, s in ckpt["optimizer"]["state"].items()}
    add_extended_inputs.scale_scores(ckpt, FACTOR)
    model = Model(ckpt["config"], blueprints.MAX_BOARD_LEN)
    sgd(model).load_state_dict(ckpt["optimizer"])
    names = [n for _, group in group_names(model) for n in group]  # by the optimizer's indices
    scaled = 0
    for i, state in ckpt["optimizer"]["state"].items():
        assert list(state) == ["momentum_buffer"]
        expected = before[i]["momentum_buffer"].clone()
        rows = add_extended_inputs.SCORE_ROWS.get(names[i].split(".")[-2])
        if rows is not None:
            expected[rows] *= FACTOR
            scaled += 1
        assert torch.equal(state["momentum_buffer"], expected), names[i]
    assert scaled == 4  # the weight and the bias of the value head's 2 layers


def test_the_score_factor_refuses_other_optimizers_factors_and_move_margins():
    ckpt = checkpoint(10)
    names = [n for _, group in group_names(Model(ckpt["config"], blueprints.MAX_BOARD_LEN)) for n in group]
    state = ckpt["optimizer"]["state"][names.index("value_head.linear_miscvaluehead.bias")]
    state["exp_avg"] = state.pop("momentum_buffer")  # as Adam's first moment
    weights = {k: v.clone() for k, v in ckpt["model"].items()}
    with pytest.raises(ValueError, match="exp_avg"):
        add_extended_inputs.scale_scores(ckpt, FACTOR)
    assert all(torch.equal(ckpt["model"][k], v) for k, v in weights.items())  # refused before any change
    alone = {"config": ckpt["config"], "model": ckpt["model"]}
    for factor in (0.0, -FACTOR, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="not a positive number"):
            add_extended_inputs.scale_scores(alone, factor)
    with pytest.raises(ValueError, match="policy output 7"):
        add_extended_inputs.scale_scores({"config": dict(ckpt["config"], version=16)}, FACTOR)


def test_the_command_line_writes_a_new_checkpoint(tmp_path, monkeypatch, capsys):
    torch.save(checkpoint(10), tmp_path / "in.ckpt")
    out = tmp_path / "out.ckpt"
    monkeypatch.setattr(sys, "argv", ["add_extended_inputs.py", str(tmp_path / "in.ckpt"), str(out),
                                      "-inputs-version", "11", "-old-score-scale", "12"])
    # The code of the checkpoint's score scale: refused, and nothing written.
    monkeypatch.setattr(blueprints, "SCORE_SCALE", 12.0)
    with pytest.raises(SystemExit, match="SCORE_SCALE is 12.0 too"):
        add_extended_inputs.main()
    assert not out.exists()
    # The code of Blueprints' new score scale, 20.
    monkeypatch.setattr(blueprints, "SCORE_SCALE", 20.0)
    add_extended_inputs.main()
    assert modelconfigs.get_inputs_version(torch.load(out, weights_only=False)["config"]) == 11
    assert modelconfigs.get_inputs_version(torch.load(tmp_path / "in.ckpt", weights_only=False)["config"]) == 10
    assert "rows [0, 2] x 0.6" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="exists"):
        add_extended_inputs.main()
