"""The shuffle writes its files (the shards and the output files) at zlib's level 1 (shuffle.savez_zlib_level), where
np.savez_compressed writes at level 6. Every reader of the files (np.load in the trainer, the phases and the tools, the
shuffle's own headers, rowindex.py) reads the members through zipfile, which inflates any level. So the test compares
the zip's members: at level 6 the writer writes numpy's file, byte for byte, and at level 1 each member holds the same
bytes in a larger deflate. The rows are random, in the dtypes and shapes that the self-play writes, with every array."""
import os
import pathlib
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import shuffle  # noqa: E402
from test_npz_reader_batches import config, write_rows  # noqa: E402

KEYS = ("binaryInputNCHWPacked", "globalInputNC", "policyTargetsNCMove", "globalTargetsNC", "scoreDistrN",
        "valueTargetsNCHW", "metadataInputNC", "qValueTargetsNCMove")


def written_by_numpy(tmp_path, rows):
    """A file of random rows that np.savez_compressed wrote, and its arrays."""
    path = write_rows(tmp_path / "numpy.npz", config("blueprints", 14), rows, seed=rows, meta=True, qvalues=True)
    with np.load(path) as npz:
        assert tuple(npz.keys()) == KEYS
        return path, {key: npz[key] for key in KEYS}


def members(path):
    """Each member's zip fields but the size of its deflate, then that size and the member's bytes."""
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        return [((i.filename, i.date_time, i.compress_type, i.file_size, i.CRC, i.flag_bits, i.extract_version,
                  i.create_system), i.compress_size, z.read(i)) for i in z.infolist()]


def test_at_level_6_it_writes_numpys_file(tmp_path):
    old, arrays = written_by_numpy(tmp_path, 41)
    new = str(tmp_path / "new.npz")
    shuffle.savez_zlib_level(new, 6, **arrays)
    # zipfile gives each member the same fixed time (1980-01-01), so the clock does not change the file.
    assert pathlib.Path(new).read_bytes() == pathlib.Path(old).read_bytes()


@pytest.mark.parametrize("rows", [0, 1, 41])
def test_at_level_1_the_members_hold_the_same_bytes(tmp_path, rows):
    old, arrays = written_by_numpy(tmp_path, rows)
    new = str(tmp_path / "new.npz")
    shuffle.savez_zlib_level(new, 1, **arrays)
    old_members, new_members = members(old), members(new)
    assert [(fields, data) for fields, _, data in new_members] == [(fields, data) for fields, _, data in old_members]
    if rows > 0:
        # The level binds: level 1's deflate is larger.
        assert sum(size for _, size, _ in new_members) > sum(size for _, size, _ in old_members)
    with np.load(new) as npz:
        for key in KEYS:
            assert npz[key].dtype == arrays[key].dtype and npz[key].shape == arrays[key].shape, key
            assert npz[key].tobytes() == arrays[key].tobytes(), key


def test_the_shuffles_save_function_writes_level_1(tmp_path):
    _, arrays = written_by_numpy(tmp_path, 50)
    saved, expected = str(tmp_path / "saved.npz"), str(tmp_path / "expected.npz")
    shuffle.save_output_npz(saved, [arrays[key] for key in KEYS], include_meta=True, include_qvalues=True, start=5,
                            stop=45)
    shuffle.savez_zlib_level(expected, 1, **{key: value[5:45] for key, value in arrays.items()})
    assert pathlib.Path(saved).read_bytes() == pathlib.Path(expected).read_bytes()
