"""The model kinds of the shape speed screen (the engine repository's abscls-ra93.9; the screen is abscls-ra93.7): reg6c128m64,
nbt5c128m64 and hybrid3l8d256 of the learning test (abscls-ra93.8).

- Each kind is b6c96 (hybrid3l8d256: hybrid3) with only the screen's keys changed.
- Each builds with bpj5's game settings (Blueprints, the inputs version 14, the regret and intrinsic heads) and has the
  parameters that the screen counted for it.
- The names have no hyphen: a hyphen starts a variant suffix, and the kinds are not variants of another kind."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from katago.train import modelconfigs  # noqa: E402
from katago.train.games import blueprints  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402
from test_extended_inputs import config  # noqa: E402

BPJ5_HEADS = ("regret", "intrinsic")

# kind: (the kind it is made from, the keys that differ from it, the parameters with bpj5's heads)
KINDS = {
    "reg6c128m64": ("b6c96", {"trunk_num_channels": 128, "mid_num_channels": 64}, 1_011_025),
    "nbt5c128m64": ("b6c96", {"trunk_num_channels": 128, "mid_num_channels": 64, "block_kind": [
        ["rconv1", "bottlenest2"], ["rconv2", "bottlenest2gpool"], ["rconv3", "bottlenest2"],
        ["rconv4", "bottlenest2gpool"], ["rconv5", "bottlenest2"]]}, 984_017),
    "hybrid3l8d256": ("hybrid3", {"patch_channels": 256, "patch_pairs": 8, "transformer_heads": 8,
                                  "transformer_ffn_channels": 768}, 7_454_801),
}


@pytest.mark.parametrize("kind", KINDS)
def test_a_kind_differs_from_its_base_in_the_screens_keys_only(kind):
    base, change, _ = KINDS[kind]
    expected = {**modelconfigs.config_of_name[base], **change}
    assert modelconfigs.config_of_name[kind] == expected
    assert "-" not in kind


@pytest.mark.parametrize("kind", KINDS)
def test_a_kind_builds_with_bpj5s_settings_and_has_the_screens_parameters(kind):
    _, _, parameters = KINDS[kind]
    torch.manual_seed(0)
    model = Model(modelconfigs.with_heads(config("blueprints", 14, kind), BPJ5_HEADS), blueprints.MAX_BOARD_LEN)
    model.initialize()
    assert sum(p.numel() for p in model.parameters()) == parameters
