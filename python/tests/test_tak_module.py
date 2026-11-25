"""Tak's module for the trainer (katago/train/games/tak.py) against the C++ encoding part's numbers, as its source gives
them, and the module's masks and images against a naive count. Also the loader's checks that Tak brought: rows of
another largest board, and rows of a game with classes of endings whose ending has no weight.

The engine repository's games/tak/tests/test_encoding.py compares the module with the C++ code through the rules module,
and its games/tak/tools/check_rows.py the rows and the batches with the games.
"""
import pathlib
import re

import numpy as np
import pytest
import torch

from katago.train import data_processing_pytorch, games, modelconfigs

CPP = pathlib.Path(__file__).resolve().parents[2] / "cpp"
SIDE = 6
MOVES = SIDE * SIDE * 255 + 1
DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))  # +, -, >, <

tak = games.get("tak")


def drop_lists(limit):
    """Every list of drops of a pickup of 1 to limit pieces, with its pattern: 2^(k-1) - 1 plus a bit 2^(i-1) for each
    step before the last that ends after the i-th piece."""
    out = []

    def extend(drops, left):
        for c in range(1, left + 1):
            steps = drops + [c]
            k = sum(steps)
            ends = np.cumsum(steps)[:-1]
            out.append((steps, 2 ** (k - 1) - 1 + sum(2 ** (int(e) - 1) for e in ends)))
            extend(steps, left - c)

    extend([], limit)
    return out


def slots_of(n, x0=0, y0=0):
    """The slots that a board of side n at the corner (x0, y0) can use, one by one."""
    slots = set()
    for y in range(n):
        for x in range(n):
            first = ((y + y0) * SIDE + x + x0) * 255
            slots |= {first, first + 1, first + 2}
            for d, (dx, dy) in enumerate(DIRECTIONS):
                for drops, pattern in drop_lists(n):
                    ex, ey = x + dx * len(drops), y + dy * len(drops)
                    if 0 <= ex < n and 0 <= ey < n:
                        slots.add(first + 3 + 4 * pattern + d)
    return slots


def constant(header, name):
    """A constexpr number of a C++ header."""
    text = (CPP / "game" / "tak" / header).read_text(encoding="utf-8")
    found = re.findall(rf"\bconstexpr \w+ (?:[\w ,=]*, )?{name} = ([^;,]+)[;,]", text)
    assert len(found) == 1, (name, found)
    return found[0]


def test_the_module_has_the_encoding_parts_numbers():
    for name, value in (("NUM_SPATIAL_INPUTS", tak.NUM_SPATIAL_INPUTS), ("NUM_GLOBAL_INPUTS", tak.NUM_GLOBAL_INPUTS),
                        ("HISTORY_PLANE", tak.HISTORY_PLANE), ("ENDING_CLASSES", tak.ENDING_CLASSES),
                        ("SCORE_SCALE", tak.SCORE_SCALE), ("GAME_LENGTH", tak.GAME_LENGTH),
                        ("TOP_PLANE", tak.MY_TOP), ("EDGE_PLANE", tak.EDGE_PLANE)):
        assert float(constant("gameencoding.h", name)) == value, name
    assert int(constant("gamerules.h", "HISTORY")) == tak.NUM_HISTORY_PLANES
    assert constant("gameencoding.h", "SYMMETRIES") == "0xFF" and tak.SYMMETRIES == 0xFF
    for name in ("DRAWS_AS_NO_RESULT", "LEGAL_MOVES_IN_ROWS"):
        assert constant("gameencoding.h", name) == "true" and getattr(tak, name) is True
    source = (CPP / "game" / "tak" / "gameencoding.h").read_text(encoding="utf-8")
    declared = re.search(r"unsigned TARGETS = ([^;]+);", source).group(1)
    assert sorted(re.findall(r"TARGET_\w+", declared)) == sorted(
        name for name in ("TARGET_OWNERSHIP", "TARGET_SEKI", "TARGET_FUTURE_PIECES", "TARGET_SCORING", "TARGET_INCOME",
                          "TARGET_NEXT_PIECES") if tak.TARGETS & getattr(games, name))


def test_every_game_gives_as_many_classes_of_endings_as_its_encoding_part():
    """With C++ alone having classes, the losses of other endings fall back to C27, which is 0 on them, with no word of
    it. With the trainer alone having them, G77 is another column."""
    for header in sorted((CPP / "game").glob("*/gameencoding.h")):
        found = re.search(r"constexpr int ENDING_CLASSES = (\d+);", header.read_text(encoding="utf-8"))
        assert games.ending_classes(games.get(header.parent.name)) == (int(found.group(1)) if found else 0), header


def test_the_networks_board_is_a_number_that_cmake_reads():
    """cpp/CMakeLists.txt takes the largest board of katago's build from this line."""
    source = pathlib.Path(tak.__file__).read_text(encoding="utf-8")
    cmake = (CPP / "CMakeLists.txt").read_text(encoding="utf-8")
    assert 'REGEX "^MAX_BOARD_LEN = [0-9]+"' in cmake
    assert re.findall(r"^MAX_BOARD_LEN = ([0-9]+)", source, re.MULTILINE) == [str(tak.MAX_BOARD_LEN)] == ["6"]
    assert tak.POLICY_CHANNELS == 255
    assert modelconfigs.get_game({"game": "tak"}) is tak


def test_the_policy_mask_counts_the_static_slots():
    on_board = torch.zeros((8, SIDE, SIDE))
    corners = [(6, 0, 0), (5, 0, 0), (5, 1, 0), (5, 0, 1), (5, 1, 1), (4, 2, 2), (3, 0, 3), (6, 0, 0)]
    for k, (n, x, y) in enumerate(corners):
        on_board[k, y:y + n, x:x + n] = 1
    masks = tak.policy_mask(on_board)
    for mask, (n, x, y) in zip(masks, corners):
        assert set(torch.nonzero(mask)[:, 0].tolist()) == slots_of(n, x, y), (n, x, y)
    assert [int(m.sum()) for m in masks[:2]] == [4572, 1575]
    assert not masks[:, -1].any()  # pass


def image(x, y, s):
    if s & 2:
        x = SIDE - 1 - x
    if s & 1:
        y = SIDE - 1 - y
    return (y, x) if s & 4 else (x, y)


def test_a_move_maps_to_its_squares_and_its_directions_image_at_each_symmetry():
    policy = torch.arange(MOVES, dtype=torch.float32).view(1, 1, MOVES)
    for s in range(8):
        mapped = tak.apply_symmetry_policy(policy, s, None, None)
        assert torch.equal(mapped, tak.apply_symmetry_policy_targets(policy, s, None, None, None, None))
        assert mapped[0, 0, -1] == MOVES - 1
        for y in range(SIDE):
            for x in range(SIDE):
                ix, iy = image(x, y, s)
                for c in (0, 1, 2, 3, 4, 5, 6, 3 + 4 * 37 + 2, 254):
                    to = c
                    if c >= 3:
                        pattern, d = divmod(c - 3, 4)
                        dx, dy = DIRECTIONS[d]
                        ex, ey = image(x + dx, y + dy, s)  # the first square of the spread, on the board or not
                        to = 3 + 4 * pattern + DIRECTIONS.index((ex - ix, ey - iy))
                    assert mapped[0, 0, (iy * SIDE + ix) * 255 + to] == (y * SIDE + x) * 255 + c, (s, x, y, c)


def test_a_plane_maps_as_pixels_but_the_edges_turn():
    planes = torch.zeros((1, 36, SIDE, SIDE))
    for p in range(36):
        planes[0, p, 1, 4] = p + 1
    for s in range(8):
        mapped = tak.apply_symmetry_inputs(planes, s)
        ix, iy = image(4, 1, s)
        want = list(range(36))
        for side in range(2):
            for d, (dx, dy) in enumerate(DIRECTIONS):
                ex, ey = image(4 + dx, 1 + dy, s)
                want[21 + 4 * side + DIRECTIONS.index((ex - ix, ey - iy))] = 21 + 4 * side + d
        assert mapped[0, :, iy, ix].tolist() == [p + 1 for p in want], s
        assert int((mapped != 0).sum()) == 36


def test_the_income_mask_is_the_top_pieces_of_the_opponent_and_of_the_mover():
    planes = torch.zeros((1, 36, SIDE, SIDE))
    for kind in range(3):
        planes[0, 1 + kind, 0, kind] = 1      # the mover's flat, wall and capstone
        planes[0, 4 + kind, 1, kind] = 1      # the opponent's
    planes[0, 7, 2, 0] = planes[0, 8, 2, 1] = 1  # pieces below the top
    mask = tak.income_mask(planes)
    assert mask[0, 0].nonzero().tolist() == [[1, 0], [1, 1], [1, 2]]
    assert mask[0, 1].nonzero().tolist() == [[0, 0], [0, 1], [0, 2]]


def targets(rows, **columns):
    out = np.zeros((rows, 84), np.float32)
    for name, values in columns.items():
        out[:, int(name[1:])] = values
    return out


def test_the_loader_refuses_a_score_target_without_its_endings_weight():
    good = targets(4, c27=[1, 0, 0.5, 0], c77=[1, 1, 0.5, 0])
    data_processing_pytorch.check_ending_weights("rows.npz", tak, good)
    bad = targets(4, c27=[1, 0, 1, 0], c77=[1, 1, 0, 0], c51=[7, 8, 9, 10])
    with pytest.raises(ValueError, match=r"rows.npz, row 2 \(turn 9 of its game\).*C27.*G77"):
        data_processing_pytorch.check_ending_weights("rows.npz", tak, bad)
    for other in ("enclosure", "connect4", "blueprints"):  # no classes: G77 is the game's own column
        data_processing_pytorch.check_ending_weights("rows.npz", games.get(other), bad)


def test_the_loader_refuses_rows_of_another_largest_board():
    def rows(side):
        moves = side * side * (3 + 4 * (2 ** side - 1)) + 1
        return np.zeros((2, 36, (side * side + 7) // 8), np.uint8), np.zeros((2, 2, moves), np.int16)

    data_processing_pytorch.check_board("rows.npz", tak, *rows(6))
    for side in (5, 7, 8):
        with pytest.raises(ValueError, match=r"rows.npz: .*board of side 6 with 5 and 9181: .*another largest board"):
            data_processing_pytorch.check_board("rows.npz", tak, *rows(side))
