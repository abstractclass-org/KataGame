"""The extra heads of a new model: modelconfigs.HEADS and train.py's -heads."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from enclosure_rgsc_analysis import SOURCES  # noqa: E402
from katago.train import modelconfigs  # noqa: E402
from katago.train.games import blueprints  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402
from test_extended_inputs import config, inputs  # noqa: E402


def test_a_new_model_with_the_regret_and_intrinsic_heads_has_their_modules():
    heads = modelconfigs.parse_heads("regret,intrinsic")
    model = Model(modelconfigs.with_heads(config("blueprints", 14, "b1c6nbt"), heads), blueprints.MAX_BOARD_LEN)
    names = {n.split(".")[1] for n, _ in model.named_parameters() if n.startswith("value_head.")}
    assert {"regret_value", "regret_ranking", "intrinsic_value"} <= names
    assert "swing_value" not in names


def test_the_heads_read_back_from_the_config_in_one_order():
    plain = config("blueprints", 14, "b10c128")
    assert modelconfigs.get_heads(plain) == ()
    heads = modelconfigs.parse_heads("intrinsic, regret")
    assert heads == ("regret", "intrinsic")
    configured = modelconfigs.with_heads(plain, heads)
    assert modelconfigs.get_heads(configured) == ("regret", "intrinsic")
    assert "swing_heads" not in configured and "regret_heads" not in plain
    assert modelconfigs.parse_heads("") == ()


def test_the_rgsc_tools_read_each_sources_own_heads():
    # The tools of RGSC's sources (enclosure_rgsc_selectors.py) read a source's value and ranking heads by index. Heads
    # that give a constant each, the same at every position, tell them apart.
    heads = modelconfigs.parse_heads("regret,swing,intrinsic")
    cfg = modelconfigs.with_heads(config("blueprints", 14, "b1c6nbt"), heads)
    model = Model(cfg, blueprints.MAX_BOARD_LEN)
    model.initialize()
    constants = {"regret_value": -0.5, "regret_ranking": 2.0, "swing_value": 0.25, "swing_ranking": 3.0}
    with torch.no_grad():
        for name, value in constants.items():
            out = getattr(model.value_head, name).linear_out
            out.weight.zero_()
            out.bias.fill_(value)
        outputs = model.postprocess_output(model(*inputs(cfg, 2)))[0]
    for source, prefix in (("vanilla", "regret"), ("swing", "swing")):
        value, ranking = (outputs[i] for i in SOURCES[source]["heads"])
        regret = 4.0 * torch.sigmoid(torch.tensor(constants[prefix + "_value"]))  # in [0, 4]
        assert torch.allclose(value, regret.expand(2)), source
        assert torch.allclose(ranking, torch.full((2,), constants[prefix + "_ranking"])), source


@pytest.mark.parametrize("text", ["regret,value", "intrinsic,swing", "regret_heads"])
def test_an_unknown_head_and_swing_without_regret_are_refused(text):
    with pytest.raises(ValueError):
        modelconfigs.parse_heads(text)
