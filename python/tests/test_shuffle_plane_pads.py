"""The shuffle merges files of other counts of planes only for a pair that shuffle.PLANE_PADS names (abscls-l4qm):
Blueprints' inputs versions 11 and 12, whose rows differ only in the planes of closed areas that version 12 adds after
version 11's. It refuses every other mix of plane counts, and names each kind's counts and a file of each. Files of one
count of planes merge as before, their global inputs at the widest. The files run through shardify and merge_bucket:
one sharding group of both kinds (shardify meets the mix), or a group of each kind (merge_bucket meets it)."""
import os
import pkgutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import shuffle  # noqa: E402
from katago.train import games, modelconfigs  # noqa: E402
from test_npz_reader_batches import config, write_rows  # noqa: E402


def planes(game, inputs_version):
    return modelconfigs.get_num_bin_input_features(config(game, inputs_version))


def global_inputs(game, inputs_version):
    return modelconfigs.get_num_global_input_features(config(game, inputs_version))


def shuffled(tmp_path, groups):
    """The rows of the files after shardify (one call for each group of files) and merge_bucket."""
    bucket = tmp_path / "bucket"
    os.makedirs(bucket)
    for index, group in enumerate(groups):
        assert shuffle.shardify(index, [str(path) for path in group], 1, [str(bucket)], keep_prob=1.0,
                                include_meta=False, include_qvalues=False) == 0
    out_file = str(tmp_path / "out.npz")
    shuffle.merge_bucket([out_file], len(groups), str(bucket), include_meta=False, include_qvalues=False)
    return loaded(out_file)


def by_policy(rows, sources):
    """Each source row's output row, by its policy targets (random, so each row's own)."""
    out = {target.tobytes(): row for row, target in enumerate(rows["policyTargetsNCMove"])}
    return [[out[target.tobytes()] for target in arrays["policyTargetsNCMove"]] for arrays in sources]


def loaded(path):
    with np.load(path) as npz:
        return dict(npz)


GROUPINGS = {"one group": lambda a, b: [[a, b]], "a group each": lambda a, b: [[a], [b]]}


def test_the_list_holds_blueprints_planes_of_closed_areas_by_their_counts():
    key = (planes("blueprints", 11), planes("blueprints", 12), global_inputs("blueprints", 11))
    assert key == (102, 104, 52) and global_inputs("blueprints", 12) == 52
    assert set(shuffle.PLANE_PADS) == {key}
    # No other game's rows show either kind of a pair, at any inputs version: the counts tell the games apart.
    names = [m.name for m in pkgutil.iter_modules(games.__path__)]
    assert "tak" in names and "enclosure" in names
    listed = {(p, g) for narrow, wide, g in shuffle.PLANE_PADS for p in (narrow, wide)}
    for name in set(names) - {"blueprints"}:
        for version in modelconfigs.INPUTS_VERSIONS:
            assert (planes(name, version), global_inputs(name, version)) not in listed, (name, version)


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_blueprints_rows_of_versions_11_and_12_merge_as_before(tmp_path, grouping):
    v11 = write_rows(tmp_path / "v11.npz", config("blueprints", 11), 3, seed=1)
    v12 = write_rows(tmp_path / "v12.npz", config("blueprints", 12), 2, seed=2)
    sources = [loaded(v11), loaded(v12)]
    rows = shuffled(tmp_path, GROUPINGS[grouping](v11, v12))
    assert rows["binaryInputNCHWPacked"].shape[1] == 104 and len(rows["binaryInputNCHWPacked"]) == 5
    (old, new) = by_policy(rows, sources)
    out = rows["binaryInputNCHWPacked"]
    assert np.array_equal(out[old, :102], sources[0]["binaryInputNCHWPacked"]) and not out[old, 102:].any()
    assert np.array_equal(out[new], sources[1]["binaryInputNCHWPacked"])
    assert np.array_equal(rows["globalInputNC"][old + new],
                          np.concatenate([sources[0]["globalInputNC"], sources[1]["globalInputNC"]]))


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_tak_rows_of_36_planes_do_not_merge_with_rows_of_59(tmp_path, grouping):
    """Tak's planes changed in the middle at 59 (abscls-shfz.35), with the same 21 global inputs: rows of 36 planes
    padded to 59 would be wrong inputs. These files are made here: no game wrote them."""
    assert (planes("tak", 15), global_inputs("tak", 15)) == (59, 21)
    old = write_rows(tmp_path / "old36.npz", config("tak", 15), 3, seed=3, planes=36)
    new = write_rows(tmp_path / "new59.npz", config("tak", 15), 2, seed=4)
    with pytest.raises(ValueError) as refused:
        shuffled(tmp_path, GROUPINGS[grouping](old, new))
    message = str(refused.value)
    # shardify names the files; merge_bucket names its shards, each the rows of one group of files of one kind.
    if grouping == "one group":
        first, second = str(old), str(new)
    else:
        first, second = (os.path.join(str(tmp_path / "bucket"), f"{index}.npz") for index in (0, 1))
    assert f"36 planes and 21 global inputs ({first}); 59 planes and 21 global inputs ({second})" in message, message
    assert "not a pair on shuffle.PLANE_PADS" in message


def test_rows_whose_planes_and_global_inputs_both_differ_stay_refused():
    # Blueprints' versions 9 (90 planes, 24 global inputs) and 10 (102, 44).
    with pytest.raises(ValueError, match=r"90 planes and 24 global inputs \(a\.npz\); 102 planes and 44 global inputs "
                                         r"\(b\.npz\)\. .*not a pair on shuffle\.PLANE_PADS"):
        shuffle.pad_planes_to_widest([np.ones((3, 90, 5)), np.ones((2, 102, 5))],
                                     [np.zeros((3, 24)), np.zeros((2, 44))], ["a.npz", "b.npz"])


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_files_of_one_count_of_planes_merge_as_before(tmp_path, grouping):
    # Blueprints' versions 10 and 11: 102 planes, and 44 and 52 global inputs, the narrower padded with zeros.
    v10 = write_rows(tmp_path / "v10.npz", config("blueprints", 10), 3, seed=5)
    v11 = write_rows(tmp_path / "v11.npz", config("blueprints", 11), 2, seed=6)
    sources = [loaded(v10), loaded(v11)]
    rows = shuffled(tmp_path, GROUPINGS[grouping](v10, v11))
    (old, new) = by_policy(rows, sources)
    assert np.array_equal(rows["binaryInputNCHWPacked"][old + new],
                          np.concatenate([sources[0]["binaryInputNCHWPacked"], sources[1]["binaryInputNCHWPacked"]]))
    assert rows["globalInputNC"].shape[1] == 52 and not rows["globalInputNC"][old, 44:].any()
    assert np.array_equal(rows["globalInputNC"][old, :44], sources[0]["globalInputNC"])
    # Files of the same counts: the arrays as they are.
    same = [np.ones((3, 59, 5)), np.ones((2, 59, 5))]
    arrays = list(same)
    shuffle.pad_planes_to_widest(arrays, [np.zeros((3, 21)), np.zeros((2, 21))], ["a.npz", "b.npz"])
    assert arrays[0] is same[0] and arrays[1] is same[1]
