"""The extra heads of a new model (modelconfigs.HEADS, train.py's -heads): RGSC's regret pair (regret), its swing pair
(swing) and AZdb's intrinsic value head (intrinsic). A network trained from scratch beside a run's network gets the
run network's heads at its start, where enclosure_add_regret_heads.py and add_diversity_members.py add them to a
checkpoint trained before (the engine repository's games/blueprints/docs/decisions/port.md, abscls-luzl.15)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from katago.train import modelconfigs  # noqa: E402
from katago.train.games import blueprints  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402
from test_extended_inputs import config  # noqa: E402


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
    assert configured["regret_heads"] is True and configured["intrinsic_head"] is True
    assert "swing_heads" not in configured and "regret_heads" not in plain
    assert modelconfigs.parse_heads("") == ()


@pytest.mark.parametrize("text", ["regret,value", "intrinsic,swing", "regret_heads"])
def test_an_unknown_head_and_swing_without_regret_are_refused(text):
    with pytest.raises(ValueError):
        modelconfigs.parse_heads(text)
