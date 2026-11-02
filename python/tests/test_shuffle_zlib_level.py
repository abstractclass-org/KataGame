"""The shuffle writes its files (the shards and the output files) at zlib's level 1 (shuffle.savez_zlib_level), where
np.savez_compressed writes at level 6. The arrays read back are the same, bit for bit, and every reader of the shuffle's
files reads the new files as it read the old ones: np.load (the trainer's, the phases' and the tools'), zipfile (the
row counts of rowindex.py and the shuffle's own headers), and the trainer's batches. The rows are random, in the dtypes
and shapes that the self-play writes. The files differ in their bytes: level 1 gives larger files."""
import os
import sys
import zipfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import shuffle  # noqa: E402
from katago.train.data_processing_pytorch import read_npz_training_data  # noqa: E402
from test_npz_reader_batches import assert_equal_batches, config, read, write_rows  # noqa: E402

KEYS = ("binaryInputNCHWPacked", "globalInputNC", "policyTargetsNCMove", "globalTargetsNC", "scoreDistrN",
        "valueTargetsNCHW")


def arrays_of(path):
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.keys()}


def rows_as_arrays(tmp_path, cfg, rows, seed, **kw):
    """The arrays of a random file of rows, as a dict (the file is written once and read back)."""
    return arrays_of(write_rows(tmp_path / f"source{seed}.npz", cfg, rows, seed=seed, **kw))


def write_both(tmp_path, arrays, name="f"):
    """The same arrays written at level 6 (np.savez_compressed) and at level 1 (the shuffle's)."""
    old, new = str(tmp_path / f"old_{name}.npz"), str(tmp_path / f"new_{name}.npz")
    np.savez_compressed(old, **arrays)
    shuffle.savez_zlib_level(new, shuffle.SHUFFLE_ZLIB_LEVEL, **arrays)
    return old, new


def assert_same_arrays(a, b):
    assert list(a) == list(b)
    for key in a:
        assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, key
        assert a[key].tobytes() == b[key].tobytes(), key


@pytest.mark.parametrize("meta,qvalues", [(False, False), (True, False), (False, True), (True, True)])
def test_np_load_reads_the_same_arrays(tmp_path, meta, qvalues):
    cfg = config("blueprints", 14)
    arrays = rows_as_arrays(tmp_path, cfg, 37, seed=3, meta=meta, qvalues=qvalues)
    old, new = write_both(tmp_path, arrays)
    assert_same_arrays(arrays_of(old), arrays_of(new))
    assert_same_arrays(arrays, arrays_of(new))


def test_the_zip_files_differ_only_in_their_deflate(tmp_path):
    cfg = config("enclosure", 9)
    old, new = write_both(tmp_path, rows_as_arrays(tmp_path, cfg, 64, seed=1))
    with zipfile.ZipFile(old) as zold, zipfile.ZipFile(new) as znew:
        assert znew.namelist() == zold.namelist() == [key + ".npy" for key in KEYS]
        for a, b in zip(zold.infolist(), znew.infolist()):
            assert a.compress_type == b.compress_type == zipfile.ZIP_DEFLATED
            assert (b.file_size, b.CRC) == (a.file_size, a.CRC)
            assert (b.extract_version, b.flag_bits & ~0x6, b.create_system) == \
                (a.extract_version, a.flag_bits & ~0x6, a.create_system)
        assert znew.testzip() is None
        # Level 1 is the level that was asked for: its files are larger than level 6's, here by a clear margin.
        assert os.path.getsize(new) > 1.05 * os.path.getsize(old)
        assert sum(i.compress_size for i in znew.infolist()) > sum(i.compress_size for i in zold.infolist())


def test_the_zip_readers_read_the_same_headers_and_counts(tmp_path):
    cfg = config("blueprints", 14)
    old, new = write_both(tmp_path, rows_as_arrays(tmp_path, cfg, 41, seed=2, meta=True, qvalues=True))
    assert shuffle.get_numpy_npz_headers(new) == shuffle.get_numpy_npz_headers(old) is not None
    for path in (old, new):
        # rowindex.py counts the rows from the first array's header, in the zip.
        with zipfile.ZipFile(path) as z, z.open("binaryInputNCHWPacked.npy") as fh:
            version = np.lib.format.read_magic(fh)
            assert version in ((1, 0), (2, 0))
            header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)(fh)
            assert header[0][0] == 41


def test_the_shuffles_save_function_writes_level_1(tmp_path):
    cfg = config("blueprints", 14)
    arrays = rows_as_arrays(tmp_path, cfg, 50, seed=4, meta=True, qvalues=True)
    parts = [arrays[k] for k in KEYS] + [arrays["metadataInputNC"], arrays["qValueTargetsNCMove"]]
    saved = str(tmp_path / "saved.npz")
    shuffle.save_output_npz(saved, parts, True, True, 5, 45)
    sliced = {k: v[5:45] for k, v in arrays.items()}
    expected = str(tmp_path / "expected.npz")
    shuffle.savez_zlib_level(expected, 1, **sliced)
    assert_same_arrays(arrays_of(saved), sliced)
    with open(saved, "rb") as a, open(expected, "rb") as b:
        assert a.read() == b.read()  # the same bytes as a level 1 write: the save function binds the level 1
    with zipfile.ZipFile(saved) as z:
        assert {i.compress_type for i in z.infolist()} == {zipfile.ZIP_DEFLATED}


@pytest.mark.parametrize("game,inputs_version", [("blueprints", 14), ("enclosure", 9), ("connect4", 9)])
def test_the_trainers_batches_are_the_same(tmp_path, monkeypatch, game, inputs_version):
    cfg = config(game, inputs_version)
    olds, news = [], []
    for i, rows in enumerate((48, 37, 21)):
        old, new = write_both(tmp_path, rows_as_arrays(tmp_path, cfg, rows, seed=i), name=str(i))
        olds.append(old)
        news.append(new)
    batches = [read(read_npz_training_data, files, cfg, "cpu", 7, monkeypatch) for files in (olds, news)]
    assert_equal_batches(batches[1], batches[0])


def test_a_file_of_no_rows_and_of_one_row_reads_the_same(tmp_path):
    cfg = config("blueprints", 14)
    arrays = rows_as_arrays(tmp_path, cfg, 3, seed=5)
    for rows in (0, 1):
        old, new = write_both(tmp_path, {k: v[:rows] for k, v in arrays.items()}, name=str(rows))
        assert_same_arrays(arrays_of(old), arrays_of(new))
