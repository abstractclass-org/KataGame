"""The legal moves in the rows (katago/train/games: LEGAL_MOVES_IN_ROWS) and the static mask of the policy by the size
of board (policy_mask), in the loader, the shuffle and the losses. The rows hold two planes: the legal moves of the
row's position (plane 0) and of its next move's position (plane 1, abscls-shfz.33), each the floor of its soft policy
target. The game is the test game slides (slides_game.py, cpp/tests/games/slides): 469 slots, a 5 x 5 board in the
6 x 6 network, and all 8 symmetries. The test computes the legal moves from the positions alone (legal_slots). A game
without the members trains as before (test_game_module pins the games' losses and gradients)."""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import shuffle  # noqa: E402
from katago.train import data_processing_pytorch, games, modelconfigs  # noqa: E402
from katago.train.data_processing_pytorch import (  # noqa: E402
    LEGAL_MOVES_ARRAY, finish_batch, read_npz_training_data, unpack_legal_moves)
from katago.train.metrics_pytorch import Metrics  # noqa: E402
from katago.train.model_pytorch import Model, policy_len  # noqa: E402
import slides_game  # noqa: E402
from test_game_module import loss_sums, random_batch, stub_game  # noqa: E402

SIZE = slides_game.MAX_BOARD_LEN
CHANNELS = slides_game.POLICY_CHANNELS
MOVES = policy_len(SIZE, CHANNELS)
TURN = 51  # the global target that holds the row's turn


@pytest.fixture(autouse=True)
def slides(monkeypatch):
    """The trainer's module of slides, with both members."""
    return slides_game.register(monkeypatch)


def legal_slots(cells, size, opening):
    """The slots (6 * 6 * 13 + 1 booleans) that a position of a size x size board makes legal for the mover. cells holds
    the pieces (1 the mover's, -1 the opponent's, 0 none)."""
    legal = np.zeros(MOVES, dtype=bool)
    for y in range(size):
        for x in range(size):
            point = (y * SIZE + x) * CHANNELS
            if cells[y, x] == 0:
                legal[point] = True
            elif cells[y, x] == 1 and not opening:
                for d, (dx, dy) in enumerate(slides_game.DIRECTIONS):
                    for k in range(1, slides_game.MAX_SLIDE + 1):
                        ex, ey = x + dx * k, y + dy * k
                        if not (0 <= ex < size and 0 <= ey < size) or cells[ey, ex] != 0:
                            break
                        legal[point + 1 + 4 * (k - 1) + d] = True
    return legal


def random_position(rng, size, opening=False):
    cells = np.zeros((SIZE, SIZE), dtype=np.int8)
    if opening:
        return cells
    occupied = rng.random((size, size)) < 0.35
    owners = rng.choice([1, -1], size=(size, size))
    cells[:size, :size] = np.where(occupied, owners, 0)
    return cells


NO_NEXT = 5  # the rows 4, 9, 14 ... of random_rows have no next move's policy target


def planes_of(cells, size):
    """The planes (on board, the mover's, the opponent's) of a position."""
    planes = np.zeros((3, SIZE, SIZE), dtype=np.uint8)
    planes[0, :size, :size] = 1
    planes[1] = cells == 1
    planes[2] = cells == -1
    return planes


def random_rows(rng, count, with_mask=True, size=None):
    """The arrays of `count` rows as the self-play writes them, with their positions (cells, size, opening, and the
    next move's position: its cells and its opening, or None for a row without a next move's policy target)."""
    positions = []
    binary = np.zeros((count, 3, SIZE, SIZE), dtype=np.uint8)
    global_inputs = np.zeros((count, modelconfigs.get_num_global_input_features(config())), dtype=np.float32)
    policy = np.zeros((count, 2, MOVES), dtype=np.int16)
    legal = np.zeros((count, 2, MOVES), dtype=bool)
    global_targets = np.zeros((count, data_processing_pytorch.GLOBAL_TARGETS_NC_CHANNELS), dtype=np.float32)
    global_targets[:, [25, 26, 28]] = 1.0  # the row's weight, and the policy targets'
    for row in range(count):
        n = size or (SIZE if rng.random() < 0.5 else SIZE - 1)
        opening = rng.random() < 0.1
        cells = random_position(rng, n, opening)
        binary[row] = planes_of(cells, n)
        global_inputs[row, :2] = rng.random(2) < 0.5
        global_inputs[row, 1] = opening
        legal[row, 0] = legal_slots(cells, n, opening)
        visited = rng.choice(np.flatnonzero(legal[row, 0]), size=4, replace=False)
        policy[row, 0, visited] = rng.integers(1, 200, size=4)
        # The next move of the next mover, in a position of its own on the same board (any position: the loader does
        # not see how the two follow each other), or none: a uniform target of weight 0 and no mask.
        following = None
        if row % NO_NEXT == NO_NEXT - 1:
            policy[row, 1] = 1
            global_targets[row, 28] = 0.0
        else:
            next_opening = opening and rng.random() < 0.5
            following = (random_position(rng, n, next_opening), next_opening)
            legal[row, 1] = legal_slots(following[0], n, next_opening)
            visited = rng.choice(np.flatnonzero(legal[row, 1]), size=4, replace=False)
            policy[row, 1, visited] = rng.integers(1, 200, size=4)
        positions.append((cells, n, opening, following))
    global_targets[:, TURN] = np.arange(count) + 10
    arrays = dict(
        binaryInputNCHWPacked=np.packbits(binary.reshape(count, 3, -1), axis=2),
        globalInputNC=global_inputs,
        policyTargetsNCMove=policy,
        globalTargetsNC=global_targets,
        scoreDistrN=np.zeros((count, 2 * (SIZE * SIZE + games.ROW_EXTRA_SCORE_RADIUS)), dtype=np.int8),
        valueTargetsNCHW=np.zeros((count, 8, SIZE, SIZE), dtype=np.int8),
    )
    if with_mask:
        arrays[LEGAL_MOVES_ARRAY] = np.packbits(legal, axis=2)
    return arrays, positions


def static_slots(size):
    """The slots that some position of a size x size board makes legal, computed from all its points and paths."""
    static = np.zeros(MOVES, dtype=bool)
    for y in range(size):
        for x in range(size):
            cells = np.zeros((SIZE, SIZE), dtype=np.int8)
            cells[y, x] = 1
            static |= legal_slots(cells, size, False)
    return static | legal_slots(np.zeros((SIZE, SIZE), dtype=np.int8), size, True)


def config(**extra):
    return dict(modelconfigs.with_game(modelconfigs.config_of_name["b1c6nbt"], "slides"), **extra)


def write(path, arrays):
    np.savez_compressed(path, **arrays)
    return str(path)


def read(files, symmetries=False, batch_size=8, **extra):
    """The batches of the files, with the symmetries of the loader on or off."""
    cfg = config(**extra)
    return list(read_npz_training_data(files, batch_size, 1, 0, pos_len=SIZE, device="cpu",
                                       randomize_symmetries=symmetries, include_meta=False, model_config=cfg))


def slots_of_planes(planes, opening):
    """The legal slots of a batch's row from its own planes (on board, the mover's, the opponent's)."""
    size = int(planes[0].sum(0).max().item()) if planes[0].any() else 0
    cells = np.asarray(planes[1] - planes[2], dtype=np.int8)
    # The board's corner after the symmetry.
    ys, xs = np.nonzero(np.asarray(planes[0]))
    y0, x0 = ys.min(), xs.min()
    moved = np.zeros((SIZE, SIZE), dtype=np.int8)
    moved[:size, :size] = cells[y0:y0 + size, x0:x0 + size]
    at_corner = legal_slots(moved, size, opening)
    shifted = np.zeros(MOVES, dtype=bool)
    for slot in np.flatnonzero(at_corner):
        point, channel = divmod(int(slot), CHANNELS)
        shifted[((point // SIZE + y0) * SIZE + point % SIZE + x0) * CHANNELS + channel] = True
    return shifted


def image_of_planes(planes, symm):
    """The planes (3, y, x) at the symmetry symm (None: none), by the map of the inputs, not of the policy."""
    planes = torch.from_numpy(planes.astype(np.float32))[None]
    return (planes if not symm else slides_game.apply_symmetry_inputs(planes, symm))[0].numpy()


def next_slots(position, symm):
    """The legal slots of a row's next move's position (random_rows' positions) at the symmetry symm, from the image of
    its planes; none for a row without a next move's policy target."""
    _, n, _, following = position
    if following is None:
        return np.zeros(MOVES, dtype=bool)
    return slots_of_planes(image_of_planes(planes_of(following[0], n), symm), following[1])


def assert_batch_matches(batch, positions=None, symm=None):
    """Each row's mask of its position (plane 0) is the legal slots of its own planes, its visited moves are in it, and
    it has some. Its mask of the next position (plane 1) has the next move's visits, the static mask holds it, and it
    is empty where the row has no next move's target (C28 is 0). With the rows' positions (random_rows) and the batch's
    symmetry, plane 1 is the legal slots of the image of the next position."""
    static = slides_game.policy_mask(batch["binaryInputNCHW"][:, 0]).numpy()
    for row in range(batch["binaryInputNCHW"].shape[0]):
        planes = batch["binaryInputNCHW"][row].numpy()
        expected = slots_of_planes(planes, bool(batch["globalInputNC"][row, 1] > 0))
        mask, mask_next = batch["legalMovesNCMove"][row].numpy()
        assert mask.dtype == bool and mask.any()
        assert np.array_equal(mask, expected), (row, int((mask ^ expected).sum()))
        visited, visited_next = batch["policyTargetsNCMove"][row].numpy() > 0
        assert not (visited & ~mask).any()
        has_next = bool(batch["globalTargetsNC"][row, 28] != 0)
        assert mask_next.any() == has_next, row
        if has_next:
            assert not (visited_next & ~mask_next).any() and not (mask_next & ~static[row]).any(), row
        if positions is not None:
            expected_next = next_slots(positions[row], symm)
            assert np.array_equal(mask_next, expected_next), (row, int((mask_next ^ expected_next).sum()))


def symmetry_of(batch, arrays, rows):
    """The symmetry that the loader drew for the batch of the arrays' rows: the one whose image of their planes is the
    batch's."""
    n = arrays["globalTargetsNC"].shape[0]
    binary = np.unpackbits(arrays["binaryInputNCHWPacked"], axis=2)[:, :, :SIZE * SIZE].reshape(n, 3, SIZE, SIZE)
    planes = torch.from_numpy(binary[rows].astype(np.float32))
    found = [s for s in range(8) if torch.equal(slides_game.apply_symmetry_inputs(planes, s) if s else planes,
                                                batch["binaryInputNCHW"])]
    assert len(found) == 1, found
    return found[0]


def unpacked_masks(arrays):
    """The arrays' legal moves (N, 2, moves) as booleans."""
    return np.unpackbits(arrays[LEGAL_MOVES_ARRAY], axis=2, count=MOVES).astype(bool)


def row_of(arrays, symm):
    """The tensors of `finish_batch` of the arrays' rows before any symmetry, and the symmetry's batch."""
    n = arrays["globalTargetsNC"].shape[0]
    binary = np.unpackbits(arrays["binaryInputNCHWPacked"], axis=2)[:, :, :SIZE * SIZE].reshape(n, 3, SIZE, SIZE)
    return finish_batch(
        slides_game, 9, symm,
        torch.from_numpy(binary.astype(np.float32)), torch.from_numpy(arrays["globalInputNC"]),
        torch.from_numpy(arrays["policyTargetsNCMove"].astype(np.float32)),
        torch.from_numpy(arrays["globalTargetsNC"]), torch.from_numpy(arrays["scoreDistrN"].astype(np.float32)),
        torch.from_numpy(arrays["valueTargetsNCHW"].astype(np.float32)), None, None,
        legalMovesNCMove=torch.from_numpy(unpacked_masks(arrays)))


# ---------------------------------------------------------------------------------------------------------------------
# The mask's size, its bits and the static mask.


def test_a_row_grows_by_the_bytes_of_its_two_masks():
    assert games.packed_policy_len(MOVES) == 59
    assert games.row_bytes(slides_game) - games.row_bytes(type("Plain", (), dict(
        vars(slides_game), LEGAL_MOVES_IN_ROWS=False))) == 2 * 59
    # Tak's 6 x 6: 36 points of 255 channels and pass.
    assert games.packed_policy_len(36 * 255 + 1) == 1148


@pytest.mark.parametrize("moves", [1, 7, 8, 9, 469, 9181])
def test_the_bits_are_numpys_packbits(moves):
    bits = np.random.default_rng(moves).random((5, 2, moves)) < 0.3
    packed = np.packbits(bits, axis=2)
    assert packed.shape[1:] == (2, games.packed_policy_len(moves))
    assert np.array_equal(unpack_legal_moves(torch.from_numpy(packed), moves).numpy(), bits)


@pytest.mark.parametrize("size", [SIZE, SIZE - 1])
def test_the_static_mask_is_the_slots_that_some_position_of_the_board_makes_legal(size):
    on_board = torch.zeros(1, SIZE, SIZE)
    on_board[:, :size, :size] = 1.0
    mask = slides_game.policy_mask(on_board)
    assert mask.dtype == torch.bool and mask.shape == (1, MOVES)
    assert np.array_equal(mask[0].numpy(), static_slots(size))
    assert not mask[0, -1]
    assert 0 < int(mask.sum()) < MOVES


def test_the_games_have_the_members_that_they_give(monkeypatch):
    for name in ("enclosure", "connect4", "blueprints"):
        game = games.get(name)
        assert not games.legal_moves_in_rows(game)
        assert getattr(game, "policy_mask", None) is None
    assert games.legal_moves_in_rows(slides_game)


# ---------------------------------------------------------------------------------------------------------------------
# The symmetries.


@pytest.mark.parametrize("symm", range(8))
def test_the_masks_follow_the_policys_map_at_each_symmetry(symm):
    rng = np.random.default_rng(3 + symm)
    arrays, positions = random_rows(rng, 16)
    batch = row_of(arrays, symm)
    assert_batch_matches(batch, positions, symm)
    assert batch["legalMovesNCMove"][:, 1].any(dim=1).sum() == 13  # 3 of the 16 rows have no next move's target
    # The static mask of the image's board is the image of the row's static mask, and holds the legal moves.
    static = slides_game.policy_mask(batch["binaryInputNCHW"][:, 0])
    assert not (batch["legalMovesNCMove"] & ~static.unsqueeze(1)).any()


@pytest.mark.parametrize("plane", [0, 1])
@pytest.mark.parametrize("wrong", ["inverse", "none", "no direction"])
def test_a_wrong_map_of_either_mask_is_found_by_the_same_check(wrong, plane):
    inverse = [0, 1, 2, 3, 4, 6, 5, 7]
    right = slides_game.apply_symmetry_policy

    def broken(policy, symmetry):
        if wrong == "inverse":
            return right(policy, inverse[symmetry], None, None)
        if wrong == "none":
            return policy
        planes = policy[:, :, :-1].reshape(policy.shape[0], policy.shape[1], SIZE, SIZE, CHANNELS)
        planes = games.apply_symmetry_points(planes.permute(0, 1, 4, 2, 3), symmetry).permute(0, 1, 3, 4, 2)
        return torch.cat((planes.reshape(policy.shape[0], policy.shape[1], -1), policy[:, :, -1:]), dim=2)

    # The map of the policy targets and of the other plane stays right, so that the map of one plane alone is wrong.
    def finish(arrays, symm):
        batch = row_of(arrays, symm)
        mask = torch.from_numpy(unpacked_masks(arrays)[:, plane:plane + 1]).float()
        batch["legalMovesNCMove"][:, plane] = broken(mask, symm)[:, 0] > 0
        return batch

    rng = np.random.default_rng(5)
    arrays, positions = random_rows(rng, 16)
    failed = []
    for symm in range(8):
        try:
            assert_batch_matches(finish(arrays, symm), positions, symm)
        except AssertionError:
            failed.append(symm)
    if wrong == "inverse":
        assert failed == [5, 6]  # the two rotations by 90 degrees are the symmetries that are not their own inverses
    else:
        assert set(range(1, 8)) >= set(failed) and failed and 0 not in failed


# A row made by hand: a 5 x 5 board in the corner of the 6 x 6 network, after the opening, with the mover's pieces on
# (1, 0) and (4, 4) and the opponent's on (1, 2). Its legal moves (the point where each starts, its channel and the point
# where it ends, the same point for a placement) are written out, and so are the 8 images of a point.
HAND_MOVES = [((x, y), 0, (x, y)) for y in range(5) for x in range(5) if (x, y) not in ((1, 0), (4, 4), (1, 2))] + [
    ((1, 0), 1, (1, 1)),                                              # up, and (1, 2) is taken
    ((1, 0), 3, (2, 0)), ((1, 0), 7, (3, 0)), ((1, 0), 11, (4, 0)),   # right, to the edge of the 5 x 5 board
    ((1, 0), 4, (0, 0)),                                              # left
    ((4, 4), 2, (4, 3)), ((4, 4), 6, (4, 2)), ((4, 4), 10, (4, 1)),   # down
    ((4, 4), 4, (3, 4)), ((4, 4), 8, (2, 4)), ((4, 4), 12, (1, 4)),   # left
]
HAND_IMAGES = [
    lambda x, y: (x, y), lambda x, y: (x, 5 - y), lambda x, y: (5 - x, y), lambda x, y: (5 - x, 5 - y),
    lambda x, y: (y, x), lambda x, y: (5 - y, x), lambda x, y: (y, 5 - x), lambda x, y: (5 - y, 5 - x),
]


def hand_slot(start, end):
    """The slot of the move from a point to another (a placement: the same point), from the points alone."""
    (x, y), (ex, ey) = start, end
    steps = max(abs(ex - x), abs(ey - y))
    channel = 0
    if steps:
        channel = 4 * (steps - 1) + {(0, 1): 1, (0, -1): 2, (1, 0): 3, (-1, 0): 4}[(ex - x) // steps, (ey - y) // steps]
    return (y * 6 + x) * 13 + channel


def hand_rows(count):
    """The arrays of `count` rows of the row made by hand, with 3 visits on (1, 0) to (4, 0) and 1 on a placement on
    (0, 4). Its next move's position is the same position, with the same visits and so the same mask."""
    arrays, _ = random_rows(np.random.default_rng(1), count, size=5)
    planes = np.zeros((3, 6, 6), dtype=np.uint8)
    planes[0, :5, :5] = 1
    planes[1, 0, 1] = planes[1, 4, 4] = planes[2, 2, 1] = 1
    arrays["binaryInputNCHWPacked"][:] = np.packbits(planes.reshape(3, 36), axis=1)
    arrays["globalInputNC"][:, :2] = 0.0
    arrays["policyTargetsNCMove"][:] = 0
    arrays["policyTargetsNCMove"][:, :, hand_slot((1, 0), (4, 0))] = 3
    arrays["policyTargetsNCMove"][:, :, hand_slot((0, 4), (0, 4))] = 1
    arrays["globalTargetsNC"][:, 28] = 1.0
    arrays[LEGAL_MOVES_ARRAY][:] = 0
    for start, channel, end in HAND_MOVES:
        slot = (start[1] * 6 + start[0]) * 13 + channel
        assert slot == hand_slot(start, end)
        arrays[LEGAL_MOVES_ARRAY][:, :, slot // 8] |= 128 >> (slot % 8)
    return arrays


@pytest.mark.parametrize("symm", range(8))
def test_the_loader_maps_a_row_made_by_hand_at_each_symmetry(tmp_path, monkeypatch, symm):
    monkeypatch.setattr(games, "symmetries", lambda game: [symm])
    (batch,) = read([write(tmp_path / "hand.npz", hand_rows(8))], symmetries=True)
    image = HAND_IMAGES[symm]
    mine = {(x, y) for y, x in batch["binaryInputNCHW"][0, 1].nonzero().tolist()}
    assert mine == {image(1, 0), image(4, 4)}
    expected = np.zeros(MOVES, dtype=bool)
    expected[[hand_slot(image(*start), image(*end)) for start, _, end in HAND_MOVES]] = True
    assert expected.sum() == 33
    target = np.zeros(MOVES)
    target[hand_slot(image(1, 0), image(4, 0))] = 3
    target[hand_slot(image(0, 4), image(0, 4))] = 1
    for row in range(8):
        for plane in range(2):
            assert np.array_equal(batch["legalMovesNCMove"][row, plane].numpy(), expected)
            assert np.array_equal(batch["policyTargetsNCMove"][row, plane].numpy(), target)
    # The 25 placements and 4 x (20 + 15 + 10) slides of a 5 x 5 board, in the corner where the board stands.
    static = slides_game.policy_mask(batch["binaryInputNCHW"][:, 0])
    assert static.sum(dim=1).tolist() == [205] * 8
    assert not (batch["legalMovesNCMove"] & ~static.unsqueeze(1)).any()


def test_the_loader_gives_each_batch_its_own_mask_at_every_symmetry(tmp_path):
    rng = np.random.default_rng(17)
    files, rows = [], []
    for number in range(3):
        arrays, positions = random_rows(rng, 64)
        files.append(write(tmp_path / f"rows{number}.npz", arrays))
        rows.append((arrays, positions))
    seen, drawn = 0, set()
    for batch in read(files, symmetries=True, batch_size=16):
        # The files' rows in order, 4 batches a file.
        arrays, positions = rows[seen // 4]
        part = slice(16 * (seen % 4), 16 * (seen % 4 + 1))
        symm = symmetry_of(batch, arrays, part)
        assert_batch_matches(batch, positions[part], symm)
        drawn.add(symm)
        seen += 1
    assert seen == 3 * 4 and len(drawn) > 2


# ---------------------------------------------------------------------------------------------------------------------
# The stops, with the row's name.


def rows_file(tmp_path, change):
    arrays, positions = random_rows(np.random.default_rng(23), 16, size=SIZE)
    change(arrays, positions)
    return write(tmp_path / "rows.npz", arrays)


def set_bit(packed, row, move, value, plane=0):
    if value:
        packed[row, plane, move >> 3] |= 0x80 >> (move & 7)
    else:
        packed[row, plane, move >> 3] &= 0xFF ^ (0x80 >> (move & 7))


def test_a_visited_move_outside_the_rows_mask_stops_the_loader_with_the_rows_name(tmp_path):
    def change(arrays, positions):
        visited = int(np.flatnonzero(arrays["policyTargetsNCMove"][5, 0])[0])
        set_bit(arrays[LEGAL_MOVES_ARRAY], 5, visited, False)

    with pytest.raises(ValueError, match=r"rows\.npz, row 5 \(turn 15 of its game\) has visits on a move outside its "
                                         r"own legal moves"):
        read([rows_file(tmp_path, change)])


def test_a_visit_of_the_next_move_outside_its_positions_mask_stops_the_loader_with_the_rows_name(tmp_path):
    def change(arrays, positions):
        visited = int(np.flatnonzero(arrays["policyTargetsNCMove"][6, 1])[0])
        set_bit(arrays[LEGAL_MOVES_ARRAY], 6, visited, False, plane=1)

    with pytest.raises(ValueError, match=r"rows\.npz, row 6 \(turn 16 of its game\) has visits of the next move "
                                         r"outside the legal moves of its position"):
        read([rows_file(tmp_path, change)])


@pytest.mark.parametrize("plane, row, what", [
    (0, 9, "a legal move"), (1, 8, "a legal move of its next move's position")])
def test_a_legal_move_outside_the_static_mask_stops_the_loader_with_the_rows_name(tmp_path, plane, row, what):
    # A slide off the board: legal by no rule, so the static mask has not the slot.
    off_board = (0 * SIZE + 0) * CHANNELS + 1 + 4 * 0 + 1  # (0, 0) slides in -y

    def change(arrays, positions):
        set_bit(arrays[LEGAL_MOVES_ARRAY], row, off_board, True, plane)

    with pytest.raises(ValueError, match=rf"rows\.npz, row {row} \(turn {row + 10} of its game\) has {what} outside "
                                         r"the game's policy_mask: the move \d+ \(x 0, y 0, channel 2\)"):
        read([rows_file(tmp_path, change)])


def test_a_visit_outside_the_static_mask_stops_the_loader_for_either_target(tmp_path):
    for channel, what in ((0, "has visits on a move outside"), (1, "has visits of the next move outside")):
        def change(arrays, positions, channel=channel):
            # A slide that leaves the board at the corner, and a legal bit of its own in its target's plane, so that a
            # check of the static mask is the one that stops.
            move = (0 * SIZE + 0) * CHANNELS + 1 + 4 * 0 + 1
            arrays["policyTargetsNCMove"][3, channel, move] = 50
            set_bit(arrays[LEGAL_MOVES_ARRAY], 3, move, True, plane=channel)

        with pytest.raises(ValueError, match=rf"row 3 \(turn 13 of its game\) (has a legal move|{what})"):
            read([rows_file(tmp_path, change)])


def test_a_row_past_the_first_part_of_the_check_stops_the_loader_with_its_own_name(tmp_path, monkeypatch):
    monkeypatch.setattr(data_processing_pytorch, "CHECKED_ROWS_AT_A_TIME", 5)

    def change(arrays, positions):
        visited = int(np.flatnonzero(arrays["policyTargetsNCMove"][13, 0])[0])
        set_bit(arrays[LEGAL_MOVES_ARRAY], 13, visited, False)

    with pytest.raises(ValueError, match=r"rows\.npz, row 13 \(turn 23 of its game\) has visits on a move outside its "
                                         r"own legal moves"):
        read([rows_file(tmp_path, change)])
    assert len(read([rows_file(tmp_path, lambda arrays, positions: None)])) == 2


def test_the_targets_without_a_weight_are_not_visits(tmp_path):
    """A row whose policy target has no weight holds a uniform target over every move, and no visits."""
    def change(arrays, positions):
        arrays["globalTargetsNC"][4, [26, 28]] = 0.0
        arrays["policyTargetsNCMove"][4] = 1

    assert len(read([rows_file(tmp_path, change)])) == 2


@pytest.mark.parametrize("change, what", [
    (lambda packed: packed[:, :, :-1], r"uint8 of the shape \(16, 2, 58\), not uint8 of \(16, 2, 59\)"),
    (lambda packed: np.concatenate([packed, packed[:, :, :1]], axis=2),
     r"uint8 of the shape \(16, 2, 60\), not uint8 of \(16, 2, 59\)"),
    (lambda packed: packed[:-1], r"uint8 of the shape \(15, 2, 59\), not uint8 of \(16, 2, 59\)"),
    (lambda packed: packed.astype(np.int16), r"int16 of the shape \(16, 2, 59\), not uint8"),
    # One plane, as the rows of abscls-shfz.9 had it, and one plane of two.
    (lambda packed: packed[:, 0], r"uint8 of the shape \(16, 59\), not uint8 of \(16, 2, 59\)"),
    (lambda packed: packed[:, :1], r"uint8 of the shape \(16, 1, 59\), not uint8 of \(16, 2, 59\)"),
])
def test_a_mask_of_a_wrong_size_stops_the_loader(tmp_path, change, what):
    def wrong(arrays, positions):
        arrays[LEGAL_MOVES_ARRAY] = change(arrays[LEGAL_MOVES_ARRAY])

    with pytest.raises(ValueError, match=rf"rows\.npz: {LEGAL_MOVES_ARRAY} is {what}"):
        read([rows_file(tmp_path, wrong)])


def test_the_rows_of_a_game_without_the_member_that_hold_the_array_stop_the_loader(tmp_path, monkeypatch):
    path = rows_file(tmp_path, lambda arrays, positions: None)
    monkeypatch.setattr(slides_game, "LEGAL_MOVES_IN_ROWS", False)
    with pytest.raises(ValueError, match=r"rows\.npz: the rows hold legalMovesNCMovePacked, and the trainer's module of "
                                         r"the game 'slides' does not give LEGAL_MOVES_IN_ROWS"):
        read([path])


def test_a_row_without_a_mask_is_read_with_0_bits_and_its_static_checks_stay(tmp_path):
    arrays, _ = random_rows(np.random.default_rng(29), 16, with_mask=False)
    path = write(tmp_path / "old.npz", arrays)
    for batch in read([path]):
        assert batch["legalMovesNCMove"].dtype == torch.bool and not batch["legalMovesNCMove"].any()
        assert batch["legalMovesNCMove"].shape == (8, 2, MOVES)
    arrays["policyTargetsNCMove"][2, 0, MOVES - 1] = 5  # a pass, which slides has in no position
    with pytest.raises(ValueError, match=r"row 2 \(turn 12 of its game\) has visits on a move outside the game's "
                                         r"policy_mask: the move 468 \(pass\)"):
        read([write(tmp_path / "bad.npz", arrays)])


# ---------------------------------------------------------------------------------------------------------------------
# The shuffle: files with the array, without it, and both.


def shuffled(tmp_path, files, name):
    """The rows of the files after shardify and merge_bucket, as a training file's arrays."""
    out_tmp = str(tmp_path / (name + "_tmp"))
    os.makedirs(out_tmp)
    assert shuffle.shardify(0, files, 1, [out_tmp], keep_prob=1.0, include_meta=False, include_qvalues=False) == 0
    out_file = str(tmp_path / (name + ".npz"))
    shuffle.merge_bucket([out_file], 1, out_tmp, include_meta=False, include_qvalues=False)
    with np.load(out_file) as npz:
        return dict(npz)


def matched(rows, sources):
    """Each output row's index into the sources' rows (all sources' rows after each other), by its policy targets."""
    key = {}
    start = 0
    for arrays in sources:
        for row, target in enumerate(arrays["policyTargetsNCMove"]):
            key[target.tobytes()] = start + row
        start += len(arrays["policyTargetsNCMove"])
    return [key[target.tobytes()] for target in rows["policyTargetsNCMove"]]


def test_the_shuffle_keeps_each_rows_mask(tmp_path):
    sources = [random_rows(np.random.default_rng(41 + n), 20)[0] for n in range(3)]
    files = [write(tmp_path / f"part{n}.npz", arrays) for n, arrays in enumerate(sources)]
    rows = shuffled(tmp_path, files, "all")
    assert sorted(matched(rows, sources)) == list(range(60))
    masks = np.concatenate([arrays[LEGAL_MOVES_ARRAY] for arrays in sources])
    assert rows[LEGAL_MOVES_ARRAY].dtype == np.uint8
    assert np.array_equal(rows[LEGAL_MOVES_ARRAY], masks[matched(rows, sources)])


def test_the_shuffle_of_a_window_of_both_kinds_gives_the_older_rows_no_mask(tmp_path):
    new = [random_rows(np.random.default_rng(50 + n), 20)[0] for n in range(2)]
    old = [random_rows(np.random.default_rng(60 + n), 20, with_mask=False)[0] for n in range(2)]
    files = [write(tmp_path / f"new{n}.npz", arrays) for n, arrays in enumerate(new)]
    files += [write(tmp_path / f"old{n}.npz", arrays) for n, arrays in enumerate(old)]
    rows = shuffled(tmp_path, files, "mixed")
    index = np.array(matched(rows, new + old))
    masks = np.concatenate([arrays[LEGAL_MOVES_ARRAY] for arrays in new])
    for row, source in enumerate(index):
        expected = masks[source] if source < 40 else np.zeros((2, 59), dtype=np.uint8)
        assert np.array_equal(rows[LEGAL_MOVES_ARRAY][row], expected), (row, source)
    # The loader reads it: the rows of the two kinds have their own masks. 8 of the 40 newer rows have no next move's
    # policy target, and so no mask of its position.
    path = write(tmp_path / "window.npz", rows)
    masks_seen = torch.cat([batch["legalMovesNCMove"].any(dim=2) for batch in read([path])])
    assert masks_seen.sum(dim=0).tolist() == [40, 32]


def test_the_shuffle_of_rows_without_the_array_has_none(tmp_path):
    old = [random_rows(np.random.default_rng(70 + n), 20, with_mask=False)[0] for n in range(2)]
    files = [write(tmp_path / f"old{n}.npz", arrays) for n, arrays in enumerate(old)]
    rows = shuffled(tmp_path, files, "none")
    assert LEGAL_MOVES_ARRAY not in rows
    assert list(rows) == list(old[0])


def test_the_shuffle_refuses_masks_of_other_widths():
    targets = np.zeros((3, 80), dtype=np.float32)
    wide, narrow = np.zeros((3, 2, 60), dtype=np.uint8), np.ones((3, 2, 59), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"files of \[\(2, 59\), \(2, 60\)\] planes and bytes of legal moves"):
        shuffle.fill_missing_legal_moves([wide, narrow], [targets, targets])
    # The one plane of abscls-shfz.9's rows does not merge with the two either.
    with pytest.raises(ValueError, match=r"files of \[\(2, 59\), \(59,\)\] planes and bytes of legal moves"):
        shuffle.fill_missing_legal_moves([narrow[:, 0], narrow], [targets, targets])
    assert shuffle.fill_missing_legal_moves([None, None], [targets, targets]) is None
    filled = shuffle.fill_missing_legal_moves([None, narrow], [targets, targets])
    assert [f.shape for f in filled] == [(3, 2, 59), (3, 2, 59)] and not filled[0].any() and filled[1].all()


# ---------------------------------------------------------------------------------------------------------------------
# The losses.


@pytest.fixture
def model():
    torch.manual_seed(2)
    built = Model(config(), SIZE)
    built.initialize()
    return built


def batch_of(arrays):
    """The batch of the arrays' rows, as the loader gives it (no symmetry)."""
    return row_of(arrays, None)


def cross_entropy(logits, target, mask, weight=None):
    """The mean-free sum over the rows of -target * log of the softmax over the slots of the mask, by another code than
    the trainer's: log_softmax of the masked logits alone. A row of weight 0 (None: all 1) is left out."""
    total = torch.zeros([])
    for row in range(logits.shape[0]):
        if weight is not None and weight[row] == 0:
            continue
        kept = mask[row]
        log_probs = torch.log_softmax(logits[row][kept], dim=0)
        total = total - (1.0 if weight is None else weight[row]) * (target[row][kept] * log_probs).sum()
    return total


def soft_target(target, floor_mask):
    """KataGo's soft target with its floor on the mask's slots alone, and 0 on the others."""
    soft = torch.pow((target + 1e-7) * floor_mask, 0.25) * floor_mask
    return soft / soft.sum(dim=1, keepdim=True)


def floors(batch):
    """The floors of the two soft targets as the rows give them: the mover's on plane 0 of the rows' masks, and the
    static mask for a plane of 0 bits, the next mover's on the static mask."""
    static = slides_game.policy_mask(batch["binaryInputNCHW"][:, 0])
    legal = batch["legalMovesNCMove"]
    return [torch.where(legal[:, 0].any(dim=1, keepdim=True), legal[:, 0], static), static]


def expected_losses(model, batch, floor_player, floor_opponent):
    """The policy losses of the model on the batch (the sums, as the metrics give them) with a softmax on the static
    mask of the board's size, and the soft targets' floors on floor_player and floor_opponent (booleans). Each target
    has its weight (C26, C28): the next mover's is 0 in a row without a next move's target."""
    with torch.no_grad():
        outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
        logits = model.postprocess_output(outputs)[0][0]
    static = slides_game.policy_mask(batch["binaryInputNCHW"][:, 0])
    normalized = [batch["policyTargetsNCMove"][:, c] / batch["policyTargetsNCMove"][:, c].sum(dim=1, keepdim=True)
                  for c in range(2)]
    player, opponent = normalized
    weight_player, weight_opponent = batch["globalTargetsNC"][:, 26], batch["globalTargetsNC"][:, 28]
    soft_player = soft_target(player, floor_player.float())
    soft_opponent = soft_target(opponent, floor_opponent.float())
    return {
        "p0loss_sum": cross_entropy(logits[:, 0], player, static, weight_player),
        "p1loss_sum": 0.15 * cross_entropy(logits[:, 1], opponent, static, weight_opponent),
        "p0softloss_sum": cross_entropy(logits[:, 2], soft_player, static, weight_player),
        "p1softloss_sum": 0.15 * cross_entropy(logits[:, 3], soft_opponent, static, weight_opponent),
        # The optimistic policies are off in these sums (no weight of the game's outcome): 0.5 of the mover's.
        "p0lopt_sum": 0.5 * cross_entropy(logits[:, 4], player, static, weight_player),
        "p0sopt_sum": 0.5 * cross_entropy(logits[:, 5], player, static, weight_player),
    }


def metric_sums(model, batch, policy=None):
    """The metrics of the model on the batch, or with `policy` in place of its policy logits."""
    with torch.no_grad():
        heads = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))
        if policy is not None:
            heads = ((policy,) + tuple(heads[0][1:]),)
        metrics = Metrics(1, model).metrics_dict_batchwise(
            model, heads, None, batch, is_training=False,
            soft_policy_weight_scale=8.0, disable_optimistic_policy=True, meta_kata_only_soft_policy=False,
            value_loss_scale=0.6, td_value_loss_scales=[0.6, 0.6, 0.6], seki_loss_scale=1.0,
            variance_time_loss_scale=1.0, main_loss_scale=None, intermediate_loss_scale=None)
    return {k: float(v) for k, v in metrics.items() if k.endswith("_sum")}


POLICY_LOSSES = ("p0loss_sum", "p1loss_sum", "p0softloss_sum", "p1softloss_sum", "p0lopt_sum", "p0sopt_sum")


def test_each_policy_loss_reads_its_softmax_and_floor_masks(model):
    arrays, _ = random_rows(np.random.default_rng(81), 24)
    batch = batch_of(arrays)
    sums = metric_sums(model, batch)
    assert all(math.isfinite(v) for v in sums.values())
    legal = batch["legalMovesNCMove"]
    expected = expected_losses(model, batch, *floors(batch))
    for name in POLICY_LOSSES:
        assert sums[name] == pytest.approx(float(expected[name]), rel=2e-4), name
    # The rows 4, 9, 14 and 19 have no next move's target: no mask of its position, and the weight 0.
    assert sums["lmrows_sum"] == 24.0
    assert legal[:, 1].any(dim=1).tolist() == [row % NO_NEXT != NO_NEXT - 1 for row in range(24)]


def test_the_floor_of_a_row_without_a_mask_is_the_static_mask(model):
    arrays, _ = random_rows(np.random.default_rng(82), 24)
    batch = batch_of(arrays)
    batch["legalMovesNCMove"][10:] = False  # older rows: no mask
    sums = metric_sums(model, batch)
    static = slides_game.policy_mask(batch["binaryInputNCHW"][:, 0])
    floor_player, floor_opponent = floors(batch)
    assert torch.equal(floor_opponent[10:], static[10:]) and torch.equal(floor_player[10:], static[10:])
    expected = expected_losses(model, batch, floor_player, floor_opponent)
    for name in POLICY_LOSSES:
        assert sums[name] == pytest.approx(float(expected[name]), rel=2e-4), name
    assert sums["lmrows_sum"] == 10.0
    # And the floor is not the legal moves of the other rows, which differ.
    legal_everywhere = expected_losses(model, batch_of(arrays), *floors(batch_of(arrays)))
    for name in ("p0softloss_sum",):
        assert abs(float(legal_everywhere[name]) - sums[name]) > 1e-3, name


def test_a_row_without_a_next_moves_target_adds_nothing_to_the_next_movers_losses(model):
    """Its target is uniform and its mask of 0 bits: its soft target is normalized on the static mask, so that no row
    gives 0 / 0, and its weight is 0. The losses of the batch are those of its other rows."""
    arrays, _ = random_rows(np.random.default_rng(83), 10)
    batch = batch_of(arrays)
    assert batch["globalTargetsNC"][:, 28].tolist() == [0.0 if row in (4, 9) else 1.0 for row in range(10)]
    sums = metric_sums(model, batch)
    assert all(math.isfinite(v) for v in sums.values())
    kept = [row for row in range(10) if row not in (4, 9)]
    part = {k: v[kept] for k, v in batch.items()}
    part_sums = metric_sums(model, part)
    for name in ("p1loss_sum", "p1softloss_sum"):
        assert sums[name] == pytest.approx(part_sums[name], rel=1e-5), name


def test_the_losses_of_a_row_made_by_hand(model):
    """The row made by hand (33 legal moves, 205 static slots), with its masks and without them. Each output has the
    logit ln 3 on one slot and 0 on the 468 others, so that its loss is ln(205 + 2) - ln 3 * (the target on that slot).
    For the soft outputs the slot is a legal move without visits, whose target is the floor's share. The next move's
    position is the same position, and the next mover's floor is on the static mask."""
    batch = batch_of(hand_rows(2))
    batch["legalMovesNCMove"][1] = False  # an older row: no mask
    unvisited, visited = hand_slot((1, 0), (1, 1)), hand_slot((1, 0), (4, 0))
    policy = torch.zeros(2, 6, MOVES)
    policy[:, [0, 1, 4, 5], visited] = math.log(3.0)
    policy[:, [2, 3], unvisited] = math.log(3.0)
    sums = metric_sums(model, batch, policy)

    def loss(target):
        return math.log(207.0) - math.log(3.0) * target

    per_slot = 0.0177827941  # 1e-7 to the power 0.25
    visits = 0.75 ** 0.25 + 0.25 ** 0.25
    on_legal, on_static = per_slot / (visits + 31 * per_slot), per_slot / (visits + 203 * per_slot)
    expected = {"p0loss_sum": 2 * loss(0.75), "p1loss_sum": 0.15 * 2 * loss(0.75),
                "p0softloss_sum": loss(on_legal) + loss(on_static),
                "p1softloss_sum": 0.15 * 2 * loss(on_static),
                "p0lopt_sum": 0.5 * 2 * loss(0.75), "p0sopt_sum": 0.5 * 2 * loss(0.75)}
    for name in POLICY_LOSSES:
        assert sums[name] == pytest.approx(expected[name], rel=1e-6), name
    # The same row, and the two floors, are told apart.
    assert abs(expected["p0softloss_sum"] - 2 * loss(on_static)) > 1e-3
    assert sums["lmrows_sum"] == 1.0


def test_a_masked_slot_has_no_logit_and_no_gradient(model):
    on_board = torch.zeros(3, SIZE, SIZE)
    on_board[:, :SIZE - 1, :SIZE - 1] = 1.0  # the 5 x 5 board
    static = slides_game.policy_mask(on_board)
    logits = torch.randn(3, 6, MOVES, requires_grad=True)
    masked, floor = Metrics(1, model).apply_policy_mask(logits, torch.ones(3, MOVES), on_board)
    assert floor.dtype == torch.float32 and torch.equal(floor > 0, static)
    assert (masked.masked_select(~static.unsqueeze(1)) == -5000.0).all()
    # The other slots are as they were.
    assert torch.equal(masked.masked_select(static.unsqueeze(1)), logits.masked_select(static.unsqueeze(1)))
    target = static.float() / static.sum(dim=1, keepdim=True)
    loss = -(target.unsqueeze(1) * torch.log_softmax(masked, dim=2)).sum()
    loss.backward()
    grad = logits.grad
    assert float(grad.masked_select(~static.unsqueeze(1)).abs().max()) == 0.0
    assert float(grad.masked_select(static.unsqueeze(1)).abs().max()) > 0.0


def test_a_policy_head_that_puts_weight_outside_the_mask_does_not_change_the_losses(model):
    """The logits of the slots outside the mask are not in the losses: 1000 more in each of them changes nothing, and
    1000 more in a slot inside it changes each loss."""
    arrays, _ = random_rows(np.random.default_rng(84), 16)
    batch = batch_of(arrays)
    static = slides_game.policy_mask(batch["binaryInputNCHW"][:, 0])
    with torch.no_grad():
        policy = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))[0][0]
    plain = metric_sums(model, batch, policy)
    assert plain == metric_sums(model, batch)
    assert metric_sums(model, batch, policy + 1000.0 * (~static).unsqueeze(1)) == plain
    inside = policy.clone()
    inside[:, :, 0] += 1000.0  # the placement on (0, 0), which is in every mask
    changed = metric_sums(model, batch, inside)
    assert all(changed[name] != plain[name] for name in POLICY_LOSSES)


def test_the_value_of_a_game_without_the_members_is_katagos(model, monkeypatch):
    """A game without a static mask and without the member: the softmax on every slot of the board's points and the soft
    targets' floor on them, KataGo's, and no metric of the rows' masks."""
    monkeypatch.setattr(slides_game, "LEGAL_MOVES_IN_ROWS", False)
    monkeypatch.setattr(slides_game, "policy_mask", None, raising=False)
    arrays, _ = random_rows(np.random.default_rng(85), 16, with_mask=False)
    batch = row_of(dict(arrays, **{LEGAL_MOVES_ARRAY: np.zeros((16, 2, 59), dtype=np.uint8)}), None)
    del batch["legalMovesNCMove"]
    with torch.no_grad():
        outputs = model(batch["binaryInputNCHW"], batch["globalInputNC"])
        logits = model.postprocess_output(outputs)[0][0]
    on_board = batch["binaryInputNCHW"][:, 0]
    board_mask = torch.cat((on_board.reshape(16, -1).repeat_interleave(CHANNELS, dim=1), torch.ones(16, 1)), dim=1)
    player = batch["policyTargetsNCMove"][:, 0]
    player = player / player.sum(dim=1, keepdim=True)
    target = torch.pow((player + 1e-7) * board_mask, 0.25)
    target = target / target.sum(dim=1, keepdim=True)
    sums = metric_sums(model, batch)
    assert "lmrows_sum" not in sums
    log_probs = torch.log_softmax(logits[:, 2], dim=1)
    assert sums["p0softloss_sum"] == pytest.approx(float(-(target * log_probs).sum()), rel=1e-4)
    expected = float(-(player * torch.log_softmax(logits[:, 0], dim=1)).sum())
    assert sums["p0loss_sum"] == pytest.approx(expected, rel=1e-4)


def test_a_game_without_the_members_has_the_same_losses_as_before(monkeypatch):
    """The trainer's losses of a stub of Connect-4 with the members off, against Connect-4's own (no members)."""
    from katago.train.games import connect4
    stub_game(monkeypatch, "plainstub", connect4, LEGAL_MOVES_IN_ROWS=False)
    torch.manual_seed(3)
    own = Model(modelconfigs.with_game(modelconfigs.config_of_name["b1c6nbt"], "connect4"), connect4.MAX_BOARD_LEN)
    own.initialize()
    batch = random_batch(own, 4)
    reference = loss_sums(own, batch)
    torch.manual_seed(3)
    stub = Model(modelconfigs.with_game(modelconfigs.config_of_name["b1c6nbt"], "plainstub"), connect4.MAX_BOARD_LEN)
    stub.initialize()
    assert loss_sums(stub, batch) == reference
