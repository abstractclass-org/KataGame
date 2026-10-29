"""AZdb's members in the trainer (the engine repository's games/blueprints/docs/research/diversity-players.md): the
inputs version 14, version 13's with the latent after its global inputs (modelconfigs, katago/train/games, the reader's
history dropout and symmetries), the rows' 4 targets of the intrinsic value head (C80-83: the reader, the shuffle and
the loss), the intrinsic value head on the shared trunk, its export as model version 23, and add_diversity_members.py,
which gives a checkpoint of version 13 the latent and the head while member 0 computes what the checkpoint's network
computed."""
import os
import sys

import numpy as np
import pytest
import torch
from torch.optim.swa_utils import AveragedModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# export_model_pytorch parses its command line at import.
sys.argv = [sys.argv[0], "-export-dir", "/nonexistent", "-model-name", "x", "-filename-prefix", "x"]
import add_diversity_members  # noqa: E402
import export_model_pytorch as export  # noqa: E402
import shuffle  # noqa: E402
from enclosure_add_regret_heads import group_names  # noqa: E402
from katago.train import data_processing_pytorch, games, modelconfigs  # noqa: E402
from katago.train.games import blueprints  # noqa: E402
from katago.train.metrics_pytorch import Metrics  # noqa: E402
from katago.train.model_pytorch import Model  # noqa: E402
from test_area_inputs import AREAS, V11_PLANES, read, write_rows  # noqa: E402
from test_extended_inputs import checkpoint, config, flat, inputs, networks, sgd  # noqa: E402
from test_game_module import random_batch  # noqa: E402

MEMBERS = modelconfigs.DIVERSITY_MEMBERS
# Blueprints' global inputs of version 13: the game's 22, the 2 playout inputs, the 20 window boxes, the 8 turn inputs.
V13_GLOBALS = 22 + 2 + 20 + 8
BOXES, TURNS = slice(24, 44), slice(44, 52)


def latent(n, member):
    """n rows of the latent of a member."""
    out = torch.zeros(n, MEMBERS)
    out[:, member] = 1.0
    return out


def test_version_14_adds_the_latent_after_version_13s_global_inputs():
    for game, globals_13 in (("blueprints", V13_GLOBALS), ("enclosure", 7)):
        assert modelconfigs.get_num_global_input_features(config(game, 14)) == globals_13 + MEMBERS
        assert modelconfigs.get_num_global_input_features(config(game, 13)) == globals_13
        assert modelconfigs.get_num_bin_input_features(config(game, 14)) == \
            modelconfigs.get_num_bin_input_features(config(game, 13))
        # A row of version 14: the latent's inputs and AZdb's 4 targets, float32 each.
        assert games.row_bytes(games.get(game), 14) - games.row_bytes(games.get(game), 13) == 4 * (MEMBERS + 4)


def test_the_trainer_maps_the_boxes_of_version_14_rows_and_not_the_latent():
    rows = torch.cat([torch.randn(3, V13_GLOBALS), latent(3, 2)], dim=1)
    out = games.apply_symmetry_global_inputs(blueprints, rows, 3)
    assert torch.equal(out[:, BOXES], blueprints.apply_symmetry_extended_global_inputs(rows[:, BOXES], 3))
    assert not torch.equal(out[:, BOXES], rows[:, BOXES])
    keep = [c for c in range(rows.shape[1]) if not BOXES.start <= c < BOXES.stop]
    assert torch.equal(out[:, keep], rows[:, keep])


def test_history_dropout_clears_the_turn_inputs_and_not_the_latent():
    n = 4000
    binary = torch.ones(n, 1 + blueprints.NUM_SPATIAL_INPUTS + 14, 2, 2)
    globals_13 = torch.ones(n, V13_GLOBALS)
    rows_14 = torch.cat([globals_13, latent(n, 1)], dim=1)
    torch.manual_seed(5)
    _, out_13 = data_processing_pytorch.apply_history_dropout(blueprints, binary, globals_13, 13)
    torch.manual_seed(5)
    _, out_14 = data_processing_pytorch.apply_history_dropout(blueprints, binary, rows_14, 14)
    # The same draws clear the same turn inputs in both versions, and some rows lose some.
    assert torch.equal(out_14[:, :V13_GLOBALS], out_13)
    assert (out_13[:, TURNS] == 0).any() and torch.equal(out_13[:, :TURNS.start], globals_13[:, :TURNS.start])
    assert torch.equal(out_14[:, V13_GLOBALS:], rows_14[:, V13_GLOBALS:])


def test_the_reader_and_the_shuffle_take_rows_of_80_and_84_global_targets():
    old, new = np.ones((2, 80), dtype=np.float32), np.ones((3, 84), dtype=np.float32)
    padded = data_processing_pytorch.pad_global_targets_nc(old)
    assert padded.shape == (2, 84) and not padded[:, 80:].any() and padded[:, :80].all()
    assert data_processing_pytorch.pad_global_targets_nc(new) is new
    # The shuffle keeps each width, and refuses to merge files of 80 with files of 84: the run of version 14 trains on
    # its own rows only.
    assert shuffle.pad_global_targets_nc(old) is old and shuffle.pad_global_targets_nc(new) is new
    assert shuffle.pad_global_targets_nc(np.ones((1, 64), dtype=np.float32)).shape == (1, 80)
    shuffle.refuse_mixed_global_targets([old, old.copy()])
    shuffle.refuse_mixed_global_targets([new, new.copy()])
    with pytest.raises(ValueError, match=r"\[80, 84\] global targets"):
        shuffle.refuse_mixed_global_targets([old, new])


def test_a_model_of_version_14_reads_the_rows_of_its_version_only(tmp_path):
    write_rows(tmp_path / "v13.npz", V11_PLANES + AREAS, 0)
    with np.load(tmp_path / "v13.npz") as z:
        arrays = dict(z)
    arrays["globalInputNC"] = np.concatenate([arrays["globalInputNC"], latent(4, 3).numpy()], axis=1)
    np.savez_compressed(tmp_path / "v14.npz", **arrays)
    (rows,) = read([tmp_path / "v14.npz"], 14)
    assert torch.equal(rows["globalInputNC"][:, V13_GLOBALS:], latent(4, 3))
    # A row of version 13 holds no member's latent, and a model of version 13 still reads it.
    with pytest.raises(ValueError, match="without the latent"):
        read([tmp_path / "v13.npz"], 14)
    (rows,) = read([tmp_path / "v13.npz"], 13)
    assert rows["globalInputNC"].shape[1] == V13_GLOBALS


def intrinsic_model(seed=1, kind="b6c96"):
    torch.manual_seed(seed)
    model = Model(dict(config("blueprints", 14, kind), intrinsic_head=True), blueprints.MAX_BOARD_LEN)
    model.initialize()
    return model


def test_the_intrinsic_head_gives_a_and_b_on_the_shared_trunk():
    model = intrinsic_model(kind="b1c6nbt")
    batch = random_batch(model)
    outputs = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))[0]
    pred = outputs[19]
    assert pred.shape == (2, 2) and pred.abs().min() > 0
    # Its gradient reaches the trunk, unlike RGSC's heads on the detached trunk.
    pred.sum().backward()
    assert model.conv_spatial.weight.grad.abs().sum() > 0
    # A model without the head gives zeros there.
    torch.manual_seed(1)
    plain = Model(config("blueprints", 14, "b1c6nbt"), blueprints.MAX_BOARD_LEN)
    plain.initialize()
    assert not plain.postprocess_output(plain(batch["binaryInputNCHW"], batch["globalInputNC"]))[0][19].any()


def intrinsic_losses(model, weights):
    """The intrinsic value head's losses on a random batch of 2 rows whose C80-83 hold targets 0.5 and -0.5 with the
    weights `weights` (A's, B's)."""
    batch = random_batch(model)
    batch["globalTargetsNC"][:, 80:84] = torch.tensor([0.5, weights[0], -0.5, weights[1]])
    outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
    metrics = Metrics(1, model).metrics_dict_batchwise(
        model, model.postprocess_output(outputs), None, batch, is_training=False,
        soft_policy_weight_scale=8.0, disable_optimistic_policy=False, meta_kata_only_soft_policy=False,
        value_loss_scale=0.6, td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=1.0,
        variance_time_loss_scale=1.0, main_loss_scale=None, intermediate_loss_scale=None)
    pred = model.postprocess_output(outputs)[0][19].detach()
    return metrics, pred


def test_the_loss_is_the_squared_error_on_the_rows_with_targets():
    model = intrinsic_model()
    with torch.no_grad():
        metrics, pred = intrinsic_losses(model, (1.0, 1.0))
    # The value loss's weight, 1.20, in the sample-wise loss (and value_loss_scale in the sum).
    expected_a = 1.20 * float(torch.square(pred[:, 0] - 0.5).sum())
    expected_b = 1.20 * float(torch.square(pred[:, 1] + 0.5).sum())
    assert float(metrics["intaloss_sum"]) == pytest.approx(expected_a, rel=1e-5)
    assert float(metrics["intbloss_sum"]) == pytest.approx(expected_b, rel=1e-5)
    assert float(metrics["intarows_sum"]) == 2 and float(metrics["intbrows_sum"]) == 2
    with torch.no_grad():
        none, _ = intrinsic_losses(model, (0.0, 0.0))
    assert float(none["intaloss_sum"]) == 0 and float(none["intbloss_sum"]) == 0
    # The loss sum holds them times value_loss_scale.
    assert float(metrics["loss_sum"]) - float(none["loss_sum"]) == pytest.approx(0.6 * (expected_a + expected_b),
                                                                                rel=1e-4)


def converted(seed=0, **options):
    ckpt = checkpoint(13)
    olds = networks(ckpt)
    touched = add_diversity_members.convert(ckpt, seed, **options)
    return ckpt, olds, touched


def outputs_without_intrinsic(net, out):
    """Every output but the intrinsic value head's columns (4 and 5 of the extra heads' outputs, the last one)."""
    out = flat(out)
    return out[:-1] + [out[-1][:, :4]]


@pytest.mark.parametrize("random_last_layer", [False, True])
def test_member_0_computes_what_the_network_did(random_last_layer):
    ckpt, olds, touched = converted(random_last_layer=random_last_layer)
    assert len(touched) == 3, touched  # the model, the SWA model, the optimizer
    assert modelconfigs.get_inputs_version(ckpt["config"]) == 14 and ckpt["config"]["intrinsic_head"]
    news = networks(ckpt)
    binary, global_13 = inputs(config("blueprints", 13, "b1c6nbt"), 3)
    with torch.no_grad():
        for old, new in zip(olds, news):
            expected = outputs_without_intrinsic(old, old(binary, global_13))
            # Member 0's latent, and no latent at all: member 0's column is 0.
            for row_latent in (latent(3, 0), torch.zeros(3, MEMBERS)):
                got = outputs_without_intrinsic(new, new(binary, torch.cat([global_13, row_latent], dim=1)))
                assert len(got) == len(expected)
                for e, o in zip(expected, got):
                    torch.testing.assert_close(o, e, rtol=1e-5, atol=1e-5)
            # The other members differ a little. A head with a random last layer gives each member its own A and B, and
            # a head whose last layer is 0 gives 0.
            heads = []
            for member in range(MEMBERS):
                out = flat(new(binary, torch.cat([global_13, latent(3, member)], dim=1)))
                heads.append(out[-1][:, 4:6])
                if member > 0:
                    assert not torch.equal(out[1], expected[1])  # the value logits
            if random_last_layer:
                assert all(not torch.equal(heads[0], h) for h in heads[1:]) and heads[0].abs().min() > 0
            else:
                assert not any(h.any() for h in heads)


def test_the_heads_last_layer_starts_at_0_and_the_trunk_gets_no_gradient_through_it():
    ckpt, _, touched = converted()
    assert all("its last layer at 0" in t for t in touched[:2]), touched
    random, _, _ = converted(random_last_layer=True)
    for part in ("model", "swa_model"):
        for k in add_diversity_members.LAST_LAYER:
            key = add_diversity_members.key_of(ckpt[part], k)
            assert not ckpt[part][key].any(), key
            assert random[part][key].abs().min() > 0, key
        # The head's other layers are KataGo's initialization, the same with either choice.
        for k in ckpt[part]:
            if add_diversity_members.HEAD in k and not any(k.endswith(last) for last in add_diversity_members.LAST_LAYER):
                assert torch.equal(ckpt[part][k], random[part][k]), k
    model = Model(ckpt["config"], blueprints.MAX_BOARD_LEN)
    model.load_state_dict(ckpt["model"])
    batch = random_batch(model)
    batch["globalInputNC"][:, V13_GLOBALS:] = latent(batch["globalInputNC"].shape[0], 3)
    pred = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))[0][19]
    assert not pred.any()
    # A loss on A and B trains the last layer, and gives the trunk and the head's other layers no gradient.
    torch.square(pred - 1.0).sum().backward()
    assert model.value_head.intrinsic_value.linear_out.weight.grad.abs().sum() > 0
    assert not model.value_head.intrinsic_value.linear2.weight.grad.any()
    assert model.conv_spatial.weight.grad is None or not model.conv_spatial.weight.grad.any()


def test_the_latent_columns_are_member_0s_zeros_and_small_draws():
    ckpt, _, _ = converted()
    model = Model(ckpt["config"], blueprints.MAX_BOARD_LEN)
    model.load_state_dict(ckpt["model"])
    columns = model.linear_global.weight[:, V13_GLOBALS:]
    assert columns.shape[1] == MEMBERS and not columns[:, 0].any() and columns[:, 1:].abs().min() > 0
    # A tenth of KataGo's initialization of the layer (Model.initialize), of the seed's fresh network.
    fresh = Model(ckpt["config"], blueprints.MAX_BOARD_LEN)
    torch.manual_seed(0)
    fresh.initialize()
    expected = fresh.linear_global.weight[:, V13_GLOBALS:] * add_diversity_members.LATENT_SCALE
    assert add_diversity_members.LATENT_SCALE == 0.1
    torch.testing.assert_close(columns[:, 1:], expected[:, 1:], rtol=0, atol=0)
    # The SWA model gets the same new weights, and the seed repeats them.
    swa = AveragedModel(Model(ckpt["config"], blueprints.MAX_BOARD_LEN))
    swa.load_state_dict(ckpt["swa_model"])
    assert torch.equal(swa.module.linear_global.weight[:, V13_GLOBALS:], columns)
    again, _, _ = converted(0)
    other, _, _ = converted(1)
    key = add_diversity_members.key_of(again["model"], add_diversity_members.GLOBAL)
    assert torch.equal(again["model"][key], ckpt["model"][key])
    assert not torch.equal(other["model"][key], ckpt["model"][key])


def test_the_optimizer_keeps_its_state_and_loads():
    ckpt = checkpoint(13)
    old_model = Model(ckpt["config"], blueprints.MAX_BOARD_LEN)
    old_names = [n for _, group in group_names(old_model) for n in group]
    before = {old_names[i]: {k: v.clone() for k, v in s.items()} for i, s in ckpt["optimizer"]["state"].items()}
    add_diversity_members.convert(ckpt)
    model = Model(ckpt["config"], blueprints.MAX_BOARD_LEN)
    sgd(model).load_state_dict(ckpt["optimizer"])
    names = [n for _, group in group_names(model) for n in group]
    after = {names[i]: s for i, s in ckpt["optimizer"]["state"].items()}
    assert set(after) == set(before)
    assert not any(n.startswith(add_diversity_members.HEAD) for n in after)
    assert any(n.startswith(add_diversity_members.HEAD) for n in names)
    for n, state in after.items():
        for k, v in state.items():
            old = before[n][k]
            if n == add_diversity_members.GLOBAL:
                assert torch.equal(v[:, :V13_GLOBALS], old) and not v[:, V13_GLOBALS:].any()
            else:
                assert torch.equal(v, old), n


def test_a_checkpoint_of_a_network_alone_converts():
    ckpt = checkpoint(13)
    alone = {"config": ckpt["config"], "model": ckpt["model"]}
    assert len(add_diversity_members.convert(alone)) == 1
    Model(alone["config"], blueprints.MAX_BOARD_LEN).load_state_dict(alone["model"])


def test_the_converter_refuses_other_checkpoints():
    with pytest.raises(ValueError, match="version 12, not 13"):
        add_diversity_members.convert({"config": config("blueprints", 12)})
    ckpt = checkpoint(13)
    with pytest.raises(ValueError, match="not 0 or more"):
        add_diversity_members.convert(ckpt, latent_scale=-1.0)
    add_diversity_members.convert(ckpt)
    with pytest.raises(ValueError, match="version 14, not 13"):
        add_diversity_members.convert(ckpt)
    with pytest.raises(ValueError, match="intrinsic value head already"):
        add_diversity_members.convert({"config": dict(config("blueprints", 13), intrinsic_head=True)})


def export_header(tmp_path, ckpt):
    """The exported model file's lines: its header, and every line after it."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, tmp_path / "checkpoint.ckpt")
    args = dict(export.args, export_dir=str(tmp_path / "out"), model_name="t", filename_prefix="model",
                checkpoint=str(tmp_path / "checkpoint.ckpt"), export_random_initialized_model=None, game=None)
    export.main(args)
    return (tmp_path / "out" / "model.bin").read_bytes().split(b"\n")


def value_head_options(lines):
    i = lines.index(b"model.value_head")
    return [int(x) for x in lines[i + 1:i + 4]]


def test_the_export_writes_model_version_23_with_the_head(tmp_path):
    ckpt, _, _ = converted()
    lines = export_header(tmp_path / "v14", ckpt)
    assert lines[1] == b"23"
    assert (int(lines[2]), int(lines[3])) == (1 + blueprints.NUM_SPATIAL_INPUTS + 14, V13_GLOBALS + MEMBERS)
    assert value_head_options(lines) == [2, 1, 0]
    assert b"model.value_head.intrinsic_value" in lines
    # A model without the head writes its file as before: option B is 0.
    old = export_header(tmp_path / "v13", checkpoint(13))
    assert old[1] == b"22" and value_head_options(old) == [2, 0, 0]
    assert b"model.value_head.intrinsic_value" not in old
