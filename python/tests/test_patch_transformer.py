"""hybrid3 in the trainer (the engine repository's abscls-ra93.2): the patch block (model_pytorch.PatchTransformerBlock)
and the model config hybrid3, Blueprints' legal moves (games.legal_moves) and holdout_losses.py.

- The packing maps each hex of the view to one token and back, and the 4 symmetries of Blueprints map the 3 x 3
  patches to patches.
- hybrid3 builds with bpj5's heads, gives the trainer's outputs at 67 x 67, and its patch block starts as the identity
  (its output map at 0). One step of SGD on the trainer's loss reaches the block, and then its attention.
- Every parameter is in one of the optimizer's groups, once.
- add_diversity_members.py converts a checkpoint of hybrid3 of the inputs version 13: it keeps every tensor but
  linear_global's, whose new columns hold member 0's at 0, and member 0 computes what it did within fp32's rounding.
- legal_moves gives the placements on the anchors' planes while no pick waits, the picks and drops on the planes of
  legal hand actions, and no pass.
- holdout_losses.evaluate gives the same measures for the same network and seed, and the policy's mass outside the
  legal moves.
- The export (the engine repository's abscls-ra93.3) writes hybrid3 as model version 24 with the patch block's lines
  (neuralnet/desc.cpp, PatchTransformerBlockDesc), and a model without the patch block as before (version 23)."""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import add_diversity_members  # noqa: E402
import holdout_losses  # noqa: E402
from enclosure_add_regret_heads import group_names  # noqa: E402
from katago.train import games, modelconfigs  # noqa: E402
from katago.train.data_processing_pytorch import GLOBAL_TARGETS_NC_CHANNELS, VALUE_TARGETS_NCHW_CHANNELS  # noqa: E402
from katago.train.games import blueprints  # noqa: E402
from katago.train.metrics_pytorch import Metrics  # noqa: E402
from katago.train.model_pytorch import Model, PatchTransformerBlock, patch_padding, policy_len  # noqa: E402
from test_diversity_members import export_header, latent, outputs_without_intrinsic  # noqa: E402
from test_extended_inputs import checkpoint, config, flat, inputs, networks  # noqa: E402
from test_game_module import random_batch  # noqa: E402

SIZE = blueprints.MAX_BOARD_LEN
BPJ5_HEADS = ("regret", "intrinsic")


def hybrid3(inputs_version=14, heads=BPJ5_HEADS, seed=0):
    torch.manual_seed(seed)
    model = Model(modelconfigs.with_heads(config("blueprints", inputs_version, "hybrid3"), heads), SIZE)
    model.initialize()
    return model


def patch_block(model):
    (block,) = [b for b in model.blocks if isinstance(b, PatchTransformerBlock)]
    return block


def test_the_padding_is_symmetric_for_3_and_one_sided_for_2():
    assert patch_padding(67, 3) == (1, 1) and patch_padding(67, 2) == (0, 1) and patch_padding(69, 3) == (0, 0)


def test_the_packing_maps_each_hex_to_one_token_and_back():
    block = patch_block(hybrid3())
    assert block.grid == 23
    x = torch.randn(2, 5, SIZE, SIZE, dtype=torch.float64)
    packed = block.pack_points(x)
    assert tuple(packed.shape) == (2, 45, 23, 23)
    assert torch.equal(block.unpack_points(packed), x)
    ids = block.pack_points(torch.arange(1, SIZE * SIZE + 1, dtype=torch.float64).view(1, 1, SIZE, SIZE))
    real = ids[ids > 0]
    assert real.numel() == SIZE * SIZE and len(set(real.tolist())) == SIZE * SIZE
    assert (ids > 0).any(dim=1).all()  # each token holds a hex of the view: the attention needs no mask


def test_the_four_symmetries_map_patches_to_patches():
    block = patch_block(hybrid3())
    ids = torch.arange(1, SIZE * SIZE + 1, dtype=torch.float64).view(1, 1, SIZE, SIZE)
    for symmetry in [s for s in range(8) if blueprints.SYMMETRIES & (1 << s)]:
        moved = block.pack_points(games.apply_symmetry_points(ids, symmetry)).sort(dim=1).values
        tokens = games.apply_symmetry_points(block.pack_points(ids).sort(dim=1).values, symmetry)
        assert torch.equal(moved, tokens), symmetry


def test_hybrid3_gives_the_trainers_outputs_and_starts_as_the_identity():
    model = hybrid3().eval()
    assert [kind for _, kind in model.config["block_kind"]] == ["regular", "regular", "patchtransformerropesg",
                                                               "regular"]
    assert not model.supports_per_block_compile()
    binary, global_ = inputs(model.config, 2)
    with torch.no_grad():
        policy = model(binary, global_)[0][0]
        assert policy.shape[0] == 2 and policy.shape[-1] == policy_len(SIZE, blueprints.POLICY_CHANNELS)
        block = patch_block(model)
        trunk = torch.randn(2, 64, SIZE, SIZE)
        mask = binary[:, 0:1]
        residual = block(trunk, mask=mask, mask_sum_hw=mask.sum(dim=(2, 3), keepdim=True), mask_sum=mask.sum(),
                         extra_outputs=None)
        assert residual.shape == trunk.shape and not residual.any()


def test_a_step_on_the_trainers_loss_reaches_the_patch_block():
    model = hybrid3()
    block = patch_block(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics_obj = Metrics(1, model)
    batch = random_batch(model)

    def loss():
        outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
        metrics = metrics_obj.metrics_dict_batchwise(
            model, model.postprocess_output(outputs), None, batch, is_training=True, soft_policy_weight_scale=8.0,
            disable_optimistic_policy=False, meta_kata_only_soft_policy=False, value_loss_scale=0.6,
            td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=1.0, variance_time_loss_scale=1.0,
            main_loss_scale=None, intermediate_loss_scale=None, include_model_norms=False)
        assert torch.isfinite(metrics["loss_sum"])
        return metrics["loss_sum"]

    loss().backward()
    assert block.unpack.weight.grad.abs().sum() > 0
    q = block.blockstack[0].q_proj.weight
    assert q.grad is None or not q.grad.any()  # no gradient through an output map of 0
    optimizer.step()
    optimizer.zero_grad()
    loss().backward()
    assert q.grad.abs().sum() > 0


def test_every_parameter_is_in_one_group_once():
    model = hybrid3()
    names = [n for _, group in group_names(model) for n in group]
    assert len(names) == len(set(names))
    assert set(names) == {n for n, _ in model.named_parameters()}


def test_a_checkpoint_of_hybrid3_of_version_13_converts_with_member_0_as_it_was():
    """The conversion changes linear_global's columns and adds the head only: the patch block's tensors and every
    other one stay, and member 0's column is 0. The outputs then agree within fp32's rounding: the wider linear_global
    sums in another order, and a deep random network carries that difference to the heads (a b6c96 converted the
    same way differs as much; test_diversity_members.py checks 1e-5 on the small b1c6nbt)."""
    ckpt = checkpoint(13, "hybrid3")
    before = {part: {k: v.clone() for k, v in ckpt[part].items()} for part in ("model", "swa_model")}
    olds = networks(ckpt)
    add_diversity_members.convert(ckpt, 0)
    for part, old in before.items():
        new = ckpt[part]
        for key, value in old.items():
            if key.endswith("linear_global.weight"):
                old_n = value.shape[1]
                assert torch.equal(new[key][:, :old_n], value) and not new[key][:, old_n].any()
            else:
                assert torch.equal(new[key], value), key
        assert any("blocks.2.blockstack." in k for k in new)
    news = networks(ckpt)
    binary, global_13 = inputs(config("blueprints", 13, "hybrid3"), 2)
    with torch.no_grad():
        for old, new in zip(olds, news):
            expected = outputs_without_intrinsic(old, old(binary, global_13))
            got = outputs_without_intrinsic(new, new(binary, torch.cat([global_13, latent(2, 0)], dim=1)))
            for e, o in zip(expected, got):
                torch.testing.assert_close(o, e, rtol=1e-3, atol=1e-2)


def test_the_export_writes_hybrid3_as_model_version_24(tmp_path):
    """The header of version 23's format, version 24, and the patch block's lines in the order that
    PatchTransformerBlockDesc reads them: the board, the patch and its padding, the stack's length, the input map, the
    attention and FFN blocks, the RMSNorm and the output map. A b6c96 of the same inputs keeps version 23."""
    lines = export_header(tmp_path / "hybrid3", checkpoint(14, "hybrid3"))
    assert lines[1] == b"24"
    i = lines.index(b"patch_transformer_block")
    assert lines[i + 1] == b"model.blocks.2"
    assert [int(x) for x in lines[i + 2:i + 7]] == [SIZE, 3, 1, 1, 8]  # the board, the patch, padding, 4 pairs
    assert lines[i + 7] == b"model.blocks.2.pack" and [int(x) for x in lines[i + 8:i + 10]] == [64 * 9, 128]
    kinds = [x for x in lines if x in (b"transformer_attention_block", b"transformer_ffn_block")]
    assert kinds == [b"transformer_attention_block", b"transformer_ffn_block"] * 4
    assert b"model.blocks.2.blockstack.7" in lines
    n = lines.index(b"model.blocks.2.norm")
    assert [float(x) for x in lines[n + 1:n + 3]] == [128, 1e-6]
    u = next(j for j in range(n, len(lines)) if lines[j] == b"model.blocks.2.unpack")
    assert [int(x) for x in lines[u + 1:u + 3]] == [128, 64 * 9]
    old = export_header(tmp_path / "b6c96", checkpoint(14))
    assert old[1] == b"23" and b"patch_transformer_block" not in old


def blank_rows(n, planes, globals_):
    """n rows of the batch's float layout with KataGo's on-board plane and nothing else."""
    binary = torch.zeros(n, planes, SIZE, SIZE)
    binary[:, 0] = 1.0
    return binary, torch.zeros(n, globals_)


def move(x, y, channel):
    return (y * SIZE + x) * blueprints.POLICY_CHANNELS + channel


def test_the_legal_moves_are_the_anchors_and_the_legal_hand_actions():
    cfg = config("blueprints", 13)
    binary, global_ = blank_rows(2, modelconfigs.get_num_bin_input_features(cfg),
                                 modelconfigs.get_num_global_input_features(cfg))
    binary[:, 1 + blueprints.MY_PLACES + 2, 10, 5] = 1.0   # blueprint 3 fits with its anchor at x 5, y 10
    binary[:, 1 + blueprints.LEGAL_HAND + 1, 1, 33] = 1.0  # a pick or a drop of blueprint 2 at x 33, y 1
    global_[1, blueprints.PICKED_INPUT] = 1.0               # row 1: a pick waits for its drop
    legal = games.legal_moves(blueprints, binary, global_)
    assert legal.shape == (2, policy_len(SIZE, blueprints.POLICY_CHANNELS)) and legal.dtype == torch.bool
    assert sorted(torch.nonzero(legal[0]).flatten().tolist()) == sorted([move(5, 10, 2), move(33, 1, 4)])
    assert torch.nonzero(legal[1]).flatten().tolist() == [move(33, 1, 4)]
    assert not legal[:, -1].any()  # no pass
    old = config("blueprints", 9)  # no plane of legal hand actions before the inputs version 10
    assert games.legal_moves(blueprints, *blank_rows(1, modelconfigs.get_num_bin_input_features(old),
                                                    modelconfigs.get_num_global_input_features(old))) is None
    assert games.legal_moves(games.get("enclosure"), binary, global_) is None


def write_rows(path, n, seed):
    """n random Blueprints rows of the inputs version 13, each with one target move that its planes make legal (a
    placement of blueprint 1), in 3 games."""
    rng = np.random.default_rng(seed)
    cfg = config("blueprints", 13)
    planes = modelconfigs.get_num_bin_input_features(cfg)
    bits = (rng.random((n, planes, SIZE * SIZE)) < 0.05).astype(np.uint8)
    bits[:, 0] = 1
    x, y = rng.integers(0, SIZE, n), rng.integers(0, SIZE, n)
    bits[np.arange(n), 1 + blueprints.MY_PLACES, y * SIZE + x] = 1
    policy = np.zeros((n, 2, policy_len(SIZE, blueprints.POLICY_CHANNELS)), dtype=np.int16)
    policy[np.arange(n), 0, (y * SIZE + x) * blueprints.POLICY_CHANNELS] = 10
    targets = np.zeros((n, games.ROW_GLOBAL_TARGETS), dtype=np.float32)
    targets[:, [0, 25, 26, 27]] = 1.0
    targets[:, 41:47] = rng.integers(0, 3, n)[:, None]
    score = np.zeros((n, 2 * SIZE * SIZE + 2 * games.ROW_EXTRA_SCORE_RADIUS), dtype=np.int8)
    score[:, score.shape[1] // 2] = 100
    np.savez_compressed(
        path, binaryInputNCHWPacked=np.packbits(bits, axis=2),
        globalInputNC=rng.standard_normal((n, modelconfigs.get_num_global_input_features(cfg))).astype(np.float32),
        policyTargetsNCMove=policy, globalTargetsNC=targets, scoreDistrN=score,
        valueTargetsNCHW=np.zeros((n, VALUE_TARGETS_NCHW_CHANNELS, SIZE, SIZE), dtype=np.int8))


def test_the_holdouts_measures_repeat_and_count_the_mass_outside_the_legal_moves(tmp_path):
    write_rows(tmp_path / "rows.npz", 8, 0)
    torch.manual_seed(0)
    model = Model(config("blueprints", 13, "b1c6nbt"), SIZE)
    model.initialize()
    model.eval()
    with torch.no_grad():
        first = holdout_losses.evaluate({"a": model, "b": model}, [str(tmp_path / "rows.npz")], 4, "cpu", seed=3)
        again = holdout_losses.evaluate({"a": model}, [str(tmp_path / "rows.npz")], 4, "cpu", seed=3)
    # The same text, NaN too: a measure of a target that no row holds (p1loss here) is NaN.
    assert json.dumps(first["a"]) == json.dumps(first["b"]) == json.dumps(again["a"])
    m = first["a"]
    assert m["rows"] == 8 and 0.0 <= m["pacc1"] <= 1.0 and 0.0 < m["illegal"] < 1.0
    assert all(np.isfinite(m[k]) for k in ("p0loss", "vloss", "oloss", "loss"))
    # A uniform policy puts on the illegal moves their share of the moves.
    batch = {"binaryInputNCHW": torch.zeros(2, 104, SIZE, SIZE), "globalInputNC": torch.zeros(2, 52),
             "globalTargetsNC": torch.zeros(2, GLOBAL_TARGETS_NC_CHANNELS),
             "policyTargetsNCMove": torch.zeros(2, 2, policy_len(SIZE, blueprints.POLICY_CHANNELS))}
    batch["binaryInputNCHW"][:, 1 + blueprints.MY_PLACES, :4, :] = 1.0
    batch["globalTargetsNC"][:, [25, 26]] = 1.0
    batch["policyTargetsNCMove"][:, 0, 0] = 1.0
    uniform = ((torch.zeros(2, 2, policy_len(SIZE, blueprints.POLICY_CHANNELS)),),)
    legal = games.legal_moves(blueprints, batch["binaryInputNCHW"], batch["globalInputNC"])
    total = holdout_losses.illegal_mass(uniform, legal, batch)
    weight = holdout_losses.batch_weights(batch)["p0w"]
    moves = policy_len(SIZE, blueprints.POLICY_CHANNELS)
    assert weight == 2.0
    torch.testing.assert_close(total / weight, torch.tensor(1.0 - 4 * SIZE / moves))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
