"""The trainer's reader of rows (katago/train/data_processing_pytorch.read_npz_training_data) keeps each file's arrays
in their dtypes on disk and expands each batch on its device. Its batches are the old reader's (reference_npz_reader.py,
which expands each file to float32 before the copy), bit for bit, for a fixed seed: with the symmetries and the history
dropout on, every tensor of every batch, with its dtype, shape and strides. The rows are random, in the dtypes and
shapes that the self-play writes, for each game, with the old rows' widths that the reader pads or cuts, the ranks of
a distributed run, the prefetch, the blanked planes, the metadata and the q-value targets. On the CPU, and on a GPU
when torch sees one."""
import os
import sys
import threading
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from katago.train import data_processing_pytorch, games, modelconfigs  # noqa: E402
from katago.train.data_processing_pytorch import (  # noqa: E402
    GLOBAL_TARGETS_NC_CHANNELS, VALUE_TARGETS_NCHW_CHANNELS, read_npz_training_data)
from katago.train.model_pytorch import policy_len  # noqa: E402
import reference_npz_reader  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def config(game, inputs_version, kind="b6c96", **extra):
    return dict(modelconfigs.with_game(modelconfigs.config_of_name[kind], game), inputs_version=inputs_version,
                **extra)


def write_rows(path, cfg, rows, seed, planes=None, global_inputs=None, global_targets=GLOBAL_TARGETS_NC_CHANNELS,
               value_channels=VALUE_TARGETS_NCHW_CHANNELS, meta=False, qvalues=False):
    """A file of `rows` random rows for the model config `cfg`, in the arrays and dtypes of the self-play's files:
    sparse planes with the on-board plane set, sparse policy targets, and global targets of small integers, so that
    the games' masks meet both cases. planes, global_inputs, global_targets and value_channels give older widths."""
    rng = np.random.default_rng(seed)
    game = modelconfigs.get_game(cfg)
    size = game.MAX_BOARD_LEN
    points = size * size
    moves = policy_len(size, game.POLICY_CHANNELS)
    planes = planes or modelconfigs.get_num_bin_input_features(cfg)
    global_inputs = global_inputs or modelconfigs.get_num_global_input_features(cfg)
    binary = (rng.random((rows, planes, points)) < 0.1).astype(np.uint8)
    binary[:, 0] = 1
    policy = np.zeros((rows, 2, moves), dtype=np.int16)
    for row in range(rows):
        for channel in range(2):
            picked = rng.choice(moves, size=12, replace=False)
            policy[row, channel, picked] = rng.integers(1, 400, size=12)
    arrays = dict(
        binaryInputNCHWPacked=np.packbits(binary, axis=2),
        globalInputNC=(rng.integers(-1, 3, size=(rows, global_inputs)) * 0.5).astype(np.float32),
        policyTargetsNCMove=policy,
        globalTargetsNC=rng.integers(-2, 3, size=(rows, global_targets)).astype(np.float32),
        scoreDistrN=rng.integers(-100, 100, size=(rows, 2 * points + 2 * games.ROW_EXTRA_SCORE_RADIUS), dtype=np.int8),
        valueTargetsNCHW=rng.integers(-100, 100, size=(rows, value_channels, size, size), dtype=np.int8),
    )
    if meta:
        arrays["metadataInputNC"] = rng.random((rows, modelconfigs.get_num_meta_encoder_input_features(1)),
                                               dtype=np.float32)
    if qvalues:
        arrays["qValueTargetsNCMove"] = rng.integers(-3000, 3000, size=(rows, games.ROW_QVALUE_TARGETS, moves),
                                                     dtype=np.int16)
    np.savez_compressed(path, **arrays)
    return str(path)


def read(reader, files, cfg, device, seed, monkeypatch, batch_size=16, world_size=1, rank=0, symmetries=True,
         meta=False, prefetch_depth=1):
    """Each batch of the reader, as {key: (dtype, shape, strides, device, values on the CPU)}, for the seed: the
    reader's generator of symmetries (os.urandom) and torch's (the history dropout)."""
    monkeypatch.setattr(os, "urandom", lambda n: bytes((seed + i) % 256 for i in range(n)))
    torch.manual_seed(seed)
    out = []
    for batch in reader(files, batch_size, world_size, rank, pos_len=modelconfigs.get_game(cfg).MAX_BOARD_LEN,
                        device=device, randomize_symmetries=symmetries, include_meta=meta, model_config=cfg,
                        prefetch_depth=prefetch_depth):
        out.append({key: (t.dtype, tuple(t.shape), t.stride(), t.device, t.cpu()) for key, t in batch.items()})
    return out


def assert_same_batches(files, cfg, device, monkeypatch, seed=7, **kw):
    new = read(read_npz_training_data, files, cfg, device, seed, monkeypatch, **kw)
    old = read(reference_npz_reader.read_npz_training_data, files, cfg, device, seed, monkeypatch, **kw)
    assert len(new) == len(old) > 0
    for n, (a, b) in enumerate(zip(new, old)):
        assert a.keys() == b.keys(), n
        for key in a:
            assert a[key][:4] == b[key][:4], (n, key)
            assert torch.equal(a[key][4], b[key][4]), (n, key)
    return new


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("game,inputs_version,kind", [
    ("blueprints", 14, "hybrid3"), ("blueprints", 14, "b6c96"), ("blueprints", 13, "b6c96"),
    ("enclosure", 9, "b6c96"), ("enclosure", 11, "b6c96"), ("connect4", 9, "b6c96")])
def test_the_batches_are_the_old_readers(tmp_path, monkeypatch, device, game, inputs_version, kind):
    cfg = config(game, inputs_version, kind)
    # Three files, the last two with rows after their last whole batch, which the reader leaves out.
    files = [write_rows(tmp_path / f"f{i}.npz", cfg, rows, seed=i) for i, rows in enumerate((48, 37, 21))]
    new = assert_same_batches(files, cfg, device, monkeypatch)
    assert len(new) == 3 + 2 + 1
    # The symmetries change the batches: the check met them.
    plain = read(read_npz_training_data, files, cfg, device, 7, monkeypatch, symmetries=False)
    assert any(not torch.equal(a["binaryInputNCHW"][4], b["binaryInputNCHW"][4]) for a, b in zip(new, plain))


@pytest.mark.parametrize("device", DEVICES)
def test_the_batches_of_older_rows_padded_or_cut_are_the_old_readers(tmp_path, monkeypatch, device):
    # Rows of the inputs version 11 for a model of version 12 (the planes of closed areas at 0), with 80 global
    # targets and 7 value channels (zero-padded), and more global inputs than the model reads (cut).
    cfg = config("blueprints", 12)
    planes = modelconfigs.get_num_bin_input_features(cfg) - games.area_spatial_inputs(games.get("blueprints"))
    files = [write_rows(tmp_path / f"f{i}.npz", cfg, 40, seed=i, planes=planes, global_targets=80, value_channels=7,
                        global_inputs=modelconfigs.get_num_global_input_features(cfg) + 3) for i in range(2)]
    assert_same_batches(files, cfg, device, monkeypatch)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prefetch_depth", [1, 3])
def test_the_batches_of_each_rank_and_prefetch_are_the_old_readers(tmp_path, monkeypatch, device, prefetch_depth):
    cfg = config("blueprints", 14)
    files = [write_rows(tmp_path / f"f{i}.npz", cfg, rows, seed=10 + i) for i, rows in enumerate((70, 64, 35, 50))]
    for rank in (0, 1):
        assert_same_batches(files, cfg, device, monkeypatch, seed=rank, world_size=2, rank=rank,
                            prefetch_depth=prefetch_depth)


@pytest.mark.parametrize("device", DEVICES)
def test_the_batches_with_blanked_planes_metadata_and_qvalues_are_the_old_readers(tmp_path, monkeypatch, device):
    monkeypatch.setattr(data_processing_pytorch, "ENCLOSURE_ZERO_INPUT_CHANNELS", [1, 5, 9])
    cfg = config("blueprints", 14, version=17, predict_q_values=True)
    files = [write_rows(tmp_path / f"f{i}.npz", cfg, 40, seed=20 + i, meta=True, qvalues=True) for i in range(2)]
    new = assert_same_batches(files, cfg, device, monkeypatch, meta=True)
    assert {"metadataInputNC", "qValueTargetsNCMove"} <= new[0].keys()
    assert not new[0]["binaryInputNCHW"][4][:, [1, 5, 9]].any()


def test_a_reader_stopped_early_drops_the_files_still_to_load(tmp_path, monkeypatch):
    """The validation stops its reader after its rows: the reader does not wait for the file in its loading thread,
    and does not load the next one, which waits in the queue."""
    cfg = config("connect4", 9)
    files = [write_rows(tmp_path / f"f{i}.npz", cfg, 16, seed=i) for i in range(4)]
    loaded, second_started, release = [], threading.Event(), threading.Event()
    original = np.load

    def held_load(path, *args, **kwargs):
        # The thread holds the second file until the test releases it (or 5 s pass).
        loaded.append(path)
        if path == files[1]:
            second_started.set()
            release.wait(timeout=5)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", held_load)
    reader = read_npz_training_data(files, 16, 1, 0, pos_len=7, device="cpu", randomize_symmetries=True,
                                    include_meta=False, model_config=cfg, prefetch_depth=1)
    next(reader)  # the first file's batch, while the thread holds the second, and the third waits
    assert second_started.wait(timeout=5)
    start = time.monotonic()
    reader.close()
    closed = time.monotonic() - start
    release.set()
    time.sleep(0.5)  # a third load, if the reader had kept it, starts now
    assert closed < 2  # a close that waited for the held load would take 5 s
    assert loaded == files[:2]
