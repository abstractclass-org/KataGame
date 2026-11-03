"""The inputs version 15: the game's planes of reach after version 14's planes (modelconfigs, katago/train/games), their
map at the symmetries (pixels), the rows of versions 14 and 15 in one window (the shuffle and the reader), the export as
model version 25 (with hybrid3's patch block too), and add_extended_inputs.py's conversion of a checkpoint of version
14, so that its network computes on the rows of version 15 what it computed on the rows of version 14, whatever the
new planes hold. The engine repository's games/blueprints/tests/test_reach_inputs.py checks the planes themselves
against the C++ kernel."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# export_model_pytorch parses its command line at import.
sys.argv = [sys.argv[0], "-export-dir", "/nonexistent", "-model-name", "x", "-filename-prefix", "x"]
import add_extended_inputs  # noqa: E402
import shuffle  # noqa: E402
from katago.train import games, modelconfigs  # noqa: E402
from katago.train.data_processing_pytorch import fit_binary_input_nchw  # noqa: E402
from katago.train.games import blueprints  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402
from test_diversity_members import export_header  # noqa: E402
from test_extended_inputs import checkpoint, config, flat, inputs, networks, sgd  # noqa: E402
from test_npz_reader_batches import write_rows  # noqa: E402
from test_patch_transformer import hybrid3  # noqa: E402

# Blueprints' rows: version 14's planes (KataGo's on-board plane, the game's 89, the extended 12, the closed areas 2)
# and global inputs (the game's 22, the playout's 2, the extended 20, the turns' 8, the latent 5).
V14_PLANES, REACH, V14_GLOBALS = 1 + 89 + 12 + 2, 9, 22 + 2 + 20 + 8 + 5
# The rows of version 12 have version 14's planes and fewer global inputs (no latent).
V12_GLOBALS = V14_GLOBALS - 5


def test_version_15_adds_the_games_planes_of_reach_after_version_14s():
    assert modelconfigs.get_num_bin_input_features(config("blueprints", 14)) == V14_PLANES
    assert modelconfigs.get_num_bin_input_features(config("blueprints", 15)) == V14_PLANES + REACH
    assert modelconfigs.get_num_global_input_features(config("blueprints", 15)) == V14_GLOBALS
    assert modelconfigs.get_num_global_input_features(config("blueprints", 14)) == V14_GLOBALS
    # The game's planes, from 0 (KataGo's on-board plane is plane 0 of a row).
    assert (blueprints.MY_REACH, blueprints.THEIR_REACH, blueprints.REACH_UNKNOWN) == (103, 107, 111)
    assert blueprints.MY_REACH == V14_PLANES - 1
    assert blueprints.REACH_UNKNOWN + 1 - blueprints.MY_REACH == blueprints.NUM_REACH_SPATIAL_INPUTS == REACH
    assert blueprints.REACH_LEVELS == (1, 2, 3, 6)
    game = games.get("blueprints")
    points = game.MAX_BOARD_LEN ** 2
    assert games.row_bytes(game, 15) - games.row_bytes(game, 14) == REACH * ((points + 7) // 8)
    # A game without them: version 15 is version 14.
    enclosure = games.get("enclosure")
    assert games.reach_spatial_inputs(enclosure) == 0
    assert modelconfigs.get_num_bin_input_features(config("enclosure", 15)) == 215
    assert games.row_bytes(enclosure, 15) == games.row_bytes(enclosure, 14)


def test_the_planes_of_reach_move_as_pixels():
    torch.manual_seed(5)
    binary = (torch.rand(5, V14_PLANES + REACH, 67, 67) < 0.2).float()
    for s in (3, 4, 7):
        out = blueprints.apply_symmetry_inputs(binary, s)
        assert torch.equal(out[:, V14_PLANES:], games.apply_symmetry_points(binary[:, V14_PLANES:], s))
        # The planes before them map as in the rows of version 14.
        assert torch.equal(out[:, :V14_PLANES], blueprints.apply_symmetry_inputs(binary[:, :V14_PLANES], s))


def assert_computes_what_it_did(olds, news, cfg):
    """The networks news, on rows of cfg's inputs (version 15), give what olds give on the rows' first V14_PLANES
    planes, whatever the planes of reach hold: at 0, at 1 and random."""
    binary, global_ = inputs(cfg, 3)
    assert binary.shape[1] == V14_PLANES + REACH and global_.shape[1] == V14_GLOBALS
    with torch.no_grad():
        for a, b in zip(olds, news):
            expected = flat(a(binary[:, :V14_PLANES], global_))
            for fill in (0.0, 1.0, None):
                x = binary.clone()
                if fill is not None:
                    x[:, V14_PLANES:] = fill
                got = flat(b(x, global_))
                assert len(got) == len(expected)
                for e, o in zip(expected, got):
                    torch.testing.assert_close(o, e, rtol=1e-5, atol=1e-5)


def test_the_converted_network_computes_what_it_did_whatever_the_planes_of_reach():
    ckpt = checkpoint(14)
    olds = networks(ckpt)
    touched = add_extended_inputs.convert(ckpt, 15)
    assert len(touched) == 3, touched  # conv_spatial in the model, the SWA model and the optimizer
    assert modelconfigs.get_inputs_version(ckpt["config"]) == 15
    assert_computes_what_it_did(olds, networks(ckpt), ckpt["config"])


def test_bpj5_s_shapes_convert_too():
    """hybrid3 with bpj5's heads (the regret heads and AZdb's intrinsic value head), a network alone."""
    model = hybrid3(14).eval()
    ckpt = {"config": model.config, "model": model.state_dict()}
    assert len(add_extended_inputs.convert(ckpt, 15)) == 1
    new = Model(ckpt["config"], blueprints.MAX_BOARD_LEN)
    new.load_state_dict(ckpt["model"])
    assert_computes_what_it_did([model], [new.eval()], ckpt["config"])


def test_the_optimizer_state_gets_zero_channels_and_loads():
    ckpt = checkpoint(14)
    before = {i: {k: v.clone() for k, v in s.items()} for i, s in ckpt["optimizer"]["state"].items()}
    add_extended_inputs.convert(ckpt, 15)
    optimizer = sgd(Model(ckpt["config"], blueprints.MAX_BOARD_LEN))
    optimizer.load_state_dict(ckpt["optimizer"])
    widened = 0
    for i, state in ckpt["optimizer"]["state"].items():
        for k, v in state.items():
            old = before[i][k]
            if v.shape != old.shape:
                widened += 1
                assert v.shape[1] == old.shape[1] + REACH
                assert torch.equal(v.narrow(1, 0, old.shape[1]), old)
                assert not v.narrow(1, old.shape[1], REACH).any()
            else:
                assert torch.equal(v, old)
    assert widened == 1  # conv_spatial's momentum: version 15 adds no global input


def test_the_converter_refuses_other_inputs():
    ckpt = checkpoint(14)
    add_extended_inputs.convert(ckpt, 15)
    with pytest.raises(ValueError, match="version 15, not 14"):
        add_extended_inputs.convert(ckpt, 15)
    with pytest.raises(ValueError, match="version 13, not 14"):
        add_extended_inputs.convert({"config": config("blueprints", 13)}, 15)
    with pytest.raises(ValueError, match="no inputs new in version 15"):
        add_extended_inputs.convert({"config": config("enclosure", 14)}, 15)
    with pytest.raises(ValueError, match="this gives"):
        add_extended_inputs.convert(checkpoint(13), 14)  # version 14 is add_diversity_members.py's


def test_the_command_line_gives_version_15(tmp_path):
    torch.save(checkpoint(14), tmp_path / "v14.ckpt")
    script = os.path.join(os.path.dirname(__file__), "..", "add_extended_inputs.py")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    run = subprocess.run([sys.executable, script, str(tmp_path / "v14.ckpt"), str(tmp_path / "v15.ckpt"),
                          "-inputs-version", "15"], capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    assert "with the inputs version 15" in run.stdout
    out = torch.load(tmp_path / "v15.ckpt", map_location="cpu", weights_only=False)
    assert modelconfigs.get_inputs_version(out["config"]) == 15


def test_the_export_writes_model_version_25_with_or_without_the_patch_block(tmp_path):
    ckpt = checkpoint(14)
    add_extended_inputs.convert(ckpt, 15)
    lines = export_header(tmp_path / "b6c96", ckpt)
    assert lines[1] == b"25"
    assert (int(lines[2]), int(lines[3])) == (V14_PLANES + REACH, V14_GLOBALS)  # the input channels, the global
    ckpt = checkpoint(14, "hybrid3")
    add_extended_inputs.convert(ckpt, 15)
    lines = export_header(tmp_path / "hybrid3", ckpt)
    assert lines[1] == b"25" and b"patch_transformer_block" in lines
    # Version 14 as before: 23, and 24 with the patch block.
    assert export_header(tmp_path / "v14", checkpoint(14))[1] == b"23"
    assert export_header(tmp_path / "v14-hybrid3", checkpoint(14, "hybrid3"))[1] == b"24"


def test_a_model_of_version_15_reads_rows_of_version_14_with_the_planes_at_0(tmp_path):
    cfg14, cfg15 = config("blueprints", 14), config("blueprints", 15)
    v14 = write_rows(tmp_path / "v14.npz", cfg14, 16, seed=1)
    v15 = write_rows(tmp_path / "v15.npz", cfg15, 16, seed=2)
    batches = read_batches([v14, v15], cfg15)
    assert [b["binaryInputNCHW"].shape[1] for b in batches] == [V14_PLANES + REACH] * 2
    assert not batches[0]["binaryInputNCHW"][:, V14_PLANES:].any()
    assert batches[1]["binaryInputNCHW"][:, V14_PLANES:].any()
    # And a model of version 14 reads a row of version 15 without them.
    (only,) = read_batches([v15], cfg14)
    assert only["binaryInputNCHW"].shape[1] == V14_PLANES


def test_a_model_of_version_15_refuses_the_rows_of_version_12(tmp_path):
    cfg15 = config("blueprints", 15)
    v12 = write_rows(tmp_path / "v12.npz", config("blueprints", 12), 16, seed=3)
    with pytest.raises(ValueError, match="without the latent"):
        read_batches([v12], cfg15)


def read_batches(paths, cfg):
    """The batches of 16 rows of the files, without symmetries, on the CPU."""
    from katago.train.data_processing_pytorch import read_npz_training_data
    return list(read_npz_training_data(paths, batch_size=16, world_size=1, rank=0, pos_len=67, device="cpu",
                                       randomize_symmetries=False, include_meta=False, model_config=cfg))


def test_the_fit_changes_the_count_by_the_planes_of_reach_or_of_closed_areas_only():
    game = games.get("blueprints")
    rows = np.ones((2, V14_PLANES, 9), dtype=np.uint8)
    fitted = fit_binary_input_nchw(rows, V14_PLANES + REACH, game)
    assert fitted.shape == (2, V14_PLANES + REACH, 9) and not fitted[:, V14_PLANES:].any()
    assert np.array_equal(fit_binary_input_nchw(fitted, V14_PLANES, game), rows)
    # The rows of version 11 for a model of version 12, as before.
    v11 = np.ones((2, V14_PLANES - 2, 9), dtype=np.uint8)
    assert fit_binary_input_nchw(v11, V14_PLANES, game).shape == (2, V14_PLANES, 9)
    # Other counts stay as they are (the reader refuses them): the rows of version 11 for a model of version 15 lack
    # both groups of planes. A game without the planes fits nothing.
    assert fit_binary_input_nchw(v11, V14_PLANES + REACH, game) is v11
    assert fit_binary_input_nchw(rows, V14_PLANES + REACH, games.get("enclosure")) is rows


def test_the_shuffle_merges_rows_of_versions_14_and_15_at_the_largest():
    narrow = np.ones((3, V14_PLANES, 5), dtype=np.uint8)
    wide = np.full((2, V14_PLANES + REACH, 5), 7, dtype=np.uint8)
    rows = [narrow, wide]
    shuffle.pad_planes_to_widest(rows, [np.zeros((3, V14_GLOBALS)), np.zeros((2, V14_GLOBALS))])
    assert rows[1] is wide
    assert rows[0].shape == (3, V14_PLANES + REACH, 5)
    assert np.array_equal(rows[0][:, :V14_PLANES], narrow) and not rows[0][:, V14_PLANES:].any()


def test_the_shuffle_refuses_rows_of_version_12_with_rows_of_version_15():
    with pytest.raises(ValueError, match="other inputs versions"):
        shuffle.pad_planes_to_widest([np.ones((3, V14_PLANES, 5)), np.ones((2, V14_PLANES + REACH, 5))],
                                     [np.zeros((3, V12_GLOBALS)), np.zeros((2, V14_GLOBALS))])
