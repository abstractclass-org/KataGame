"""The inputs version 13: version 12's inputs with the game's labels in their canonical order (modelconfigs,
katago/train/games), the same counts of planes and of global inputs, the export as model version 22, and
add_extended_inputs.py's conversion of a checkpoint of version 12, which changes no weight. The engine repository's
games/blueprints/tests check Blueprints' order and the augmentation of its rows against the C++ encoding."""
import os
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# export_model_pytorch parses its command line at import.
sys.argv = [sys.argv[0], "-export-dir", "/nonexistent", "-model-name", "x", "-filename-prefix", "x"]
import add_extended_inputs  # noqa: E402
import export_model_pytorch as export  # noqa: E402
from katago.train import games, modelconfigs  # noqa: E402
from katago.train.games import blueprints  # noqa: E402
# The checkpoints and the models' inputs of the conversion's tests of versions 10 and 12.
from test_area_inputs import AREAS, V11_GLOBALS, V11_PLANES, models  # noqa: E402
from test_extended_inputs import checkpoint, config, flat, inputs  # noqa: E402


def test_version_13_has_version_12s_inputs():
    for game in ("blueprints", "enclosure"):
        for count in (modelconfigs.get_num_bin_input_features, modelconfigs.get_num_global_input_features):
            assert count(config(game, 13)) == count(config(game, 12))
        assert games.row_bytes(games.get(game), 13) == games.row_bytes(games.get(game), 12)
    assert modelconfigs.get_num_bin_input_features(config("blueprints", 13)) == V11_PLANES + AREAS
    assert games.canonical_labels(blueprints)
    assert not games.canonical_labels(games.get("enclosure"))  # a game without the order: version 13 is version 12


def test_the_conversion_sets_the_version_and_changes_no_weight():
    ckpt = checkpoint(12)
    before = {part: {k: v.clone() for k, v in ckpt[part].items()} for part in ("model", "swa_model")}
    old, old_swa = models(ckpt)
    assert add_extended_inputs.convert(ckpt, 13) == []
    assert modelconfigs.get_inputs_version(ckpt["config"]) == 13
    for part, weights in before.items():
        assert weights.keys() == ckpt[part].keys()
        assert all(torch.equal(v, ckpt[part][k]) for k, v in weights.items())
    new, new_swa = models(ckpt)
    binary, global_ = inputs(ckpt["config"], 3)
    assert binary.shape[1] == V11_PLANES + AREAS and global_.shape[1] == V11_GLOBALS
    with torch.no_grad():
        for a, b in ((old, new), (old_swa, new_swa)):
            for e, o in zip(flat(a(binary, global_)), flat(b(binary, global_))):
                torch.testing.assert_close(o, e, rtol=0, atol=0)


def test_the_converter_refuses_other_inputs():
    ckpt = checkpoint(12)
    add_extended_inputs.convert(ckpt, 13)
    with pytest.raises(ValueError, match="version 13, not 12"):
        add_extended_inputs.convert(ckpt, 13)
    with pytest.raises(ValueError, match="version 11, not 12"):
        add_extended_inputs.convert({"config": config("blueprints", 11)}, 13)
    with pytest.raises(ValueError, match="no inputs new in version 13"):
        add_extended_inputs.convert({"config": config("enclosure", 12)}, 13)


def test_the_command_line_gives_version_13(tmp_path):
    torch.save(checkpoint(12), tmp_path / "v12.ckpt")
    script = os.path.join(os.path.dirname(__file__), "..", "add_extended_inputs.py")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    run = subprocess.run([sys.executable, script, str(tmp_path / "v12.ckpt"), str(tmp_path / "v13.ckpt"),
                          "-inputs-version", "13"], capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr
    assert "with the inputs version 13" in run.stdout
    out = torch.load(tmp_path / "v13.ckpt", map_location="cpu", weights_only=False)
    assert modelconfigs.get_inputs_version(out["config"]) == 13


def test_the_export_writes_model_version_22(tmp_path):
    ckpt = checkpoint(12)
    add_extended_inputs.convert(ckpt, 13)
    torch.save(ckpt, tmp_path / "checkpoint.ckpt")
    args = dict(export.args, export_dir=str(tmp_path / "out"), model_name="t", filename_prefix="model",
                checkpoint=str(tmp_path / "checkpoint.ckpt"), export_random_initialized_model=None, game=None)
    export.main(args)
    header = (tmp_path / "out" / "model.bin").read_bytes().split(b"\n", 4)
    assert header[1] == b"22"
    assert (int(header[2]), int(header[3])) == (V11_PLANES + AREAS, V11_GLOBALS)  # the input channels, then the global
