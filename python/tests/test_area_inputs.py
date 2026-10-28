"""The inputs version 12: the game's planes of closed areas after version 11's planes (modelconfigs,
katago/train/games), their map at the symmetries (pixels), the rows of versions 11 and 12 in one window (the shuffle and
the reader), the export as model version 21, and add_extended_inputs.py's conversion of a checkpoint of version 11, so
that its network computes what it did whatever the new planes hold."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.optim.swa_utils import AveragedModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# export_model_pytorch parses its command line at import.
sys.argv = [sys.argv[0], "-export-dir", "/nonexistent", "-model-name", "x", "-filename-prefix", "x"]
import add_extended_inputs  # noqa: E402
import export_model_pytorch as export  # noqa: E402
import shuffle  # noqa: E402
from enclosure_add_regret_heads import group_names  # noqa: E402
from katago.train import games, modelconfigs  # noqa: E402
from katago.train.data_processing_pytorch import (  # noqa: E402
    GLOBAL_TARGETS_NC_CHANNELS, VALUE_TARGETS_NCHW_CHANNELS, fit_binary_input_nchw, read_npz_training_data)
from katago.train.games import blueprints  # noqa: E402
from katago.train.model_pytorch import Model, policy_len  # noqa: E402
# The checkpoints and the models' inputs of the conversion's tests of version 10.
from test_extended_inputs import checkpoint, config, flat, inputs  # noqa: E402

V11_PLANES, AREAS, V11_GLOBALS = 1 + 89 + 12, 2, 22 + 2 + 20 + 8


def test_version_12_adds_the_games_planes_of_closed_areas_after_version_11s():
    assert modelconfigs.get_num_bin_input_features(config("blueprints", 11)) == V11_PLANES
    assert modelconfigs.get_num_bin_input_features(config("blueprints", 12)) == V11_PLANES + AREAS
    assert modelconfigs.get_num_global_input_features(config("blueprints", 12)) == V11_GLOBALS
    assert (blueprints.MY_CLOSED, blueprints.THEIR_CLOSED) == (V11_PLANES - 1, V11_PLANES)  # the game's planes, from 0
    game = games.get("blueprints")
    points = game.MAX_BOARD_LEN ** 2
    assert games.row_bytes(game, 12) - games.row_bytes(game, 11) == AREAS * ((points + 7) // 8)
    # A game without them: version 12 is version 11.
    enclosure = games.get("enclosure")
    assert games.area_spatial_inputs(enclosure) == 0
    assert modelconfigs.get_num_bin_input_features(config("enclosure", 12)) == 215
    assert games.row_bytes(enclosure, 12) == games.row_bytes(enclosure, 11)


def test_the_planes_of_closed_areas_move_as_pixels():
    torch.manual_seed(3)
    binary = (torch.rand(5, V11_PLANES + AREAS, 67, 67) < 0.2).float()
    for s in (3, 4, 7):
        out = blueprints.apply_symmetry_inputs(binary, s)
        assert torch.equal(out[:, V11_PLANES:], games.apply_symmetry_points(binary[:, V11_PLANES:], s))
        # The planes before them map as in the rows of version 11.
        assert torch.equal(out[:, :V11_PLANES], blueprints.apply_symmetry_inputs(binary[:, :V11_PLANES], s))


def models(ckpt):
    """The model and the SWA model of a checkpoint, in eval mode."""
    model = Model(ckpt["config"], blueprints.MAX_BOARD_LEN)
    model.load_state_dict(ckpt["model"])
    swa = AveragedModel(Model(ckpt["config"], blueprints.MAX_BOARD_LEN))
    swa.load_state_dict(ckpt["swa_model"])
    return model.eval(), swa.eval()


def test_the_converted_network_computes_what_it_did_whatever_the_planes_of_closed_areas():
    ckpt = checkpoint(11)
    old, old_swa = models(ckpt)
    touched = add_extended_inputs.convert(ckpt, 12)
    assert len(touched) == 3, touched  # conv_spatial in the model, the SWA model and the optimizer
    assert modelconfigs.get_inputs_version(ckpt["config"]) == 12
    new, new_swa = models(ckpt)
    binary, global_ = inputs(ckpt["config"], 3)
    assert binary.shape[1] == V11_PLANES + AREAS and global_.shape[1] == V11_GLOBALS
    with torch.no_grad():
        for a, b in ((old, new), (old_swa, new_swa)):
            expected = flat(a(binary[:, :V11_PLANES], global_))
            for fill in (0.0, 1.0, None):  # the new planes at 0, at 1, and random
                x = binary.clone()
                if fill is not None:
                    x[:, V11_PLANES:] = fill
                got = flat(b(x, global_))
                assert len(got) == len(expected)
                for e, o in zip(expected, got):
                    torch.testing.assert_close(o, e, rtol=1e-5, atol=1e-5)


def test_the_optimizer_state_gets_zero_channels_and_loads():
    ckpt = checkpoint(11)
    before = {i: {k: v.clone() for k, v in s.items()} for i, s in ckpt["optimizer"]["state"].items()}
    add_extended_inputs.convert(ckpt, 12)
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
                assert v.shape[1] == old.shape[1] + AREAS
                assert torch.equal(v.narrow(1, 0, old.shape[1]), old)
                assert not v.narrow(1, old.shape[1], AREAS).any()
            else:
                assert torch.equal(v, old)
    assert widened == 1  # conv_spatial's momentum: version 12 adds no global input


def test_a_checkpoint_of_a_network_alone_converts():
    ckpt = checkpoint(11)
    alone = {"config": ckpt["config"], "model": ckpt["model"]}
    assert len(add_extended_inputs.convert(alone, 12)) == 1
    Model(alone["config"], blueprints.MAX_BOARD_LEN).load_state_dict(alone["model"])


def test_the_converter_refuses_other_inputs():
    ckpt = checkpoint(11)
    add_extended_inputs.convert(ckpt, 12)
    with pytest.raises(ValueError, match="version 12, not 11"):
        add_extended_inputs.convert(ckpt, 12)
    with pytest.raises(ValueError, match="version 10, not 11"):
        add_extended_inputs.convert({"config": config("blueprints", 10)}, 12)
    with pytest.raises(ValueError, match="no inputs new in version 12"):
        add_extended_inputs.convert({"config": config("enclosure", 11)}, 12)


def test_the_command_line_gives_version_12_and_10_as_before(tmp_path):
    torch.save(checkpoint(11), tmp_path / "v11.ckpt")
    script = os.path.join(os.path.dirname(__file__), "..", "add_extended_inputs.py")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    run = subprocess.run([sys.executable, script, str(tmp_path / "v11.ckpt"), str(tmp_path / "v12.ckpt"),
                          "-inputs-version", "12"], capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    assert "with the inputs version 12" in run.stdout
    out = torch.load(tmp_path / "v12.ckpt", map_location="cpu", weights_only=False)
    assert modelconfigs.get_inputs_version(out["config"]) == 12
    # Version 10 from 9, as before: a checkpoint of version 11 is refused.
    run = subprocess.run([sys.executable, script, str(tmp_path / "v11.ckpt"), str(tmp_path / "v10.ckpt"),
                          "-inputs-version", "10"], capture_output=True, text=True, env=env)
    assert run.returncode != 0 and "not 9" in run.stderr
    assert not (tmp_path / "v10.ckpt").exists()


def test_the_export_writes_model_version_21(tmp_path):
    ckpt = checkpoint(11)
    add_extended_inputs.convert(ckpt, 12)
    torch.save(ckpt, tmp_path / "checkpoint.ckpt")
    args = dict(export.args, export_dir=str(tmp_path / "out"), model_name="t", filename_prefix="model",
                checkpoint=str(tmp_path / "checkpoint.ckpt"), export_random_initialized_model=None, game=None)
    export.main(args)
    header = (tmp_path / "out" / "model.bin").read_bytes().split(b"\n", 4)
    assert header[1] == b"21"
    assert (int(header[2]), int(header[3])) == (V11_PLANES + AREAS, V11_GLOBALS)  # the input channels, then the global


def write_rows(path, planes, marked):
    """Four empty Blueprints rows of the inputs version 11 (planes = V11_PLANES) or 12, each with KataGo's on-board
    plane and, in rows of version 12, the plane of closed areas `marked` (the mover's 0, the opponent's 1) set."""
    game = games.get("blueprints")
    points = game.MAX_BOARD_LEN ** 2
    moves = policy_len(game.MAX_BOARD_LEN, game.POLICY_CHANNELS)
    binary = np.zeros((4, planes, points), dtype=np.uint8)
    binary[:, 0] = 1
    if planes > V11_PLANES:
        binary[:, V11_PLANES + marked] = 1
    global_targets = np.zeros((4, GLOBAL_TARGETS_NC_CHANNELS), dtype=np.float32)
    global_targets[:, game.TARGETS_VERSION_TARGET] = 2
    np.savez_compressed(
        path,
        binaryInputNCHWPacked=np.packbits(binary, axis=2),
        globalInputNC=np.zeros((4, V11_GLOBALS), dtype=np.float32),
        policyTargetsNCMove=np.zeros((4, 2, moves), dtype=np.int16),
        globalTargetsNC=global_targets,
        scoreDistrN=np.zeros((4, 2 * points + 2 * games.ROW_EXTRA_SCORE_RADIUS), dtype=np.int8),
        valueTargetsNCHW=np.zeros((4, VALUE_TARGETS_NCHW_CHANNELS, game.MAX_BOARD_LEN, game.MAX_BOARD_LEN),
                                  dtype=np.int8),
        qValueTargetsNCMove=np.zeros((4, games.ROW_QVALUE_TARGETS, moves), dtype=np.int16),
    )


def read(paths, inputs_version):
    cfg = config("blueprints", inputs_version)
    return list(read_npz_training_data([str(p) for p in paths], batch_size=4, world_size=1, rank=0, pos_len=67,
                                       device="cpu", randomize_symmetries=False, include_meta=False, model_config=cfg))


def test_a_model_of_version_12_reads_rows_of_version_11_with_the_planes_at_0(tmp_path):
    write_rows(tmp_path / "v11.npz", V11_PLANES, 0)
    write_rows(tmp_path / "v12.npz", V11_PLANES + AREAS, 1)
    v11, v12 = read([tmp_path / "v11.npz", tmp_path / "v12.npz"], 12)
    assert v11["binaryInputNCHW"].shape[1] == v12["binaryInputNCHW"].shape[1] == V11_PLANES + AREAS
    assert not v11["binaryInputNCHW"][:, V11_PLANES:].any()
    assert v12["binaryInputNCHW"][:, V11_PLANES + 1].all() and not v12["binaryInputNCHW"][:, V11_PLANES].any()
    # And a model of version 11 reads a row of version 12 without them.
    (only,) = read([tmp_path / "v12.npz"], 11)
    assert only["binaryInputNCHW"].shape[1] == V11_PLANES


def test_the_fit_changes_the_count_by_the_planes_of_closed_areas_only():
    game = games.get("blueprints")
    rows = np.ones((2, V11_PLANES, 9), dtype=np.uint8)
    fitted = fit_binary_input_nchw(rows, V11_PLANES + AREAS, game)
    assert fitted.shape == (2, V11_PLANES + AREAS, 9) and not fitted[:, V11_PLANES:].any()
    assert np.array_equal(fit_binary_input_nchw(fitted, V11_PLANES, game), rows)
    # Rows of version 9 (90 planes) stay as they are, and the reader refuses them for a model of version 12.
    v9 = np.ones((2, 90, 9), dtype=np.uint8)
    assert fit_binary_input_nchw(v9, V11_PLANES + AREAS, game) is v9
    assert fit_binary_input_nchw(rows, V11_PLANES + AREAS, games.get("enclosure")) is rows


def test_the_shuffle_merges_rows_of_both_counts_at_the_largest():
    narrow = np.ones((3, V11_PLANES, 5), dtype=np.uint8)
    wide = np.full((2, V11_PLANES + AREAS, 5), 7, dtype=np.uint8)
    rows = [narrow, wide]
    shuffle.pad_planes_to_widest(rows, [np.zeros((3, V11_GLOBALS)), np.zeros((2, V11_GLOBALS))])
    assert rows[1] is wide
    assert rows[0].shape == (3, V11_PLANES + AREAS, 5)
    assert np.array_equal(rows[0][:, :V11_PLANES], narrow) and not rows[0][:, V11_PLANES:].any()


def test_the_shuffle_refuses_rows_of_other_versions():
    # Rows of version 9 (90 planes, 24 global inputs) with rows of version 10 (102, 44), and rows of version 10 with
    # rows of version 12: their planes and their global inputs differ.
    for (p0, g0), (p1, g1) in (((90, 24), (102, 44)), ((102, 44), (V11_PLANES + AREAS, V11_GLOBALS))):
        with pytest.raises(ValueError, match="other inputs versions"):
            shuffle.pad_planes_to_widest([np.ones((3, p0, 5)), np.ones((2, p1, 5))],
                                         [np.zeros((3, g0)), np.zeros((2, g1))])
    # Rows of versions 10 and 11 have the same planes: their global inputs merge, as before.
    rows = [np.ones((3, 102, 5)), np.ones((2, 102, 5))]
    shuffle.pad_planes_to_widest(rows, [np.zeros((3, 44)), np.zeros((2, V11_GLOBALS))])
    assert [r.shape[1] for r in rows] == [102, 102]
