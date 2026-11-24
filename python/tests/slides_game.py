"""The trainer's module of the test game slides (cpp/tests/games/slides), for the tests of the legal moves in the rows
and of the static masks (test_legal_moves.py): a 6 x 6 board, or a 5 x 5 one in its corner, and 13 channels of a move a
point: 0 a placement, and 1 + 4 * (k - 1) + d a slide of k = 1 to 3 points in the direction d (+y, -y, +x, -x). The network
reads the mover's pieces and the opponent's, and the global inputs the first player and the opening (the first 2 plies,
which have no slides).

A test registers it as the module of the game (register), because the trainer has no module of a test game. Its symmetries
are KataGo's 8 here. The C++ game has the identity alone, because no C++ test of its rows reads another."""
import sys

import torch

from katago.train.games import apply_symmetry_points

NAME = "slides"
MAX_BOARD_LEN = 6
POLICY_CHANNELS = 13
NUM_SPATIAL_INPUTS = 2
NUM_GLOBAL_INPUTS = 2
HISTORY_PLANE = 0
NUM_HISTORY_PLANES = 0
SYMMETRIES = 0xFF
TARGETS = 0
SCORE_SCALE = 1.0
GAME_LENGTH = 22.0

LEGAL_MOVES_IN_ROWS = True

DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))  # dx, dy
MAX_SLIDE = 3


def register(monkeypatch, **facts):
    """Registers this module (with `facts` in place of some of its members) as the game's, and returns it."""
    module = sys.modules[__name__]
    for name, value in facts.items():
        monkeypatch.setattr(module, name, value, raising=False)
    monkeypatch.setitem(sys.modules, "katago.train.games." + NAME, module)
    return module


def direction_image(direction, symmetry):
    """The direction's image under KataGo's symmetry (the flip of x, the flip of y, then the transpose)."""
    dx, dy = DIRECTIONS[direction]
    if symmetry & 2:
        dx = -dx
    if symmetry & 1:
        dy = -dy
    if symmetry & 4:
        dx, dy = dy, dx
    return DIRECTIONS.index((dx, dy))


def channel_image(channel, symmetry):
    """The channel of the image of a move of the channel."""
    if channel == 0:
        return 0
    length, direction = divmod(channel - 1, 4)
    return 1 + 4 * length + direction_image(direction, symmetry)


def apply_symmetry_inputs(binary_inputs, symmetry):
    return apply_symmetry_points(binary_inputs, symmetry)


def apply_symmetry_policy(policy, symmetry, binary_inputs, global_inputs):
    """Policy-shaped targets (N, C, 6 * 6 * 13 + 1), pass last."""
    if symmetry == 0:
        return policy
    n, c, _ = policy.shape
    size = MAX_BOARD_LEN
    planes = policy[:, :, :-1].reshape(n, c, size, size, POLICY_CHANNELS).permute(0, 1, 4, 2, 3)
    planes = apply_symmetry_points(planes, symmetry)
    image = torch.zeros_like(planes)
    for channel in range(POLICY_CHANNELS):
        image[:, :, channel_image(channel, symmetry)] = planes[:, :, channel]
    moves = image.permute(0, 1, 3, 4, 2).reshape(n, c, -1)
    return torch.cat((moves, policy[:, :, -1:]), dim=2)


def apply_symmetry_policy_targets(policy, symmetry, binary_inputs, global_inputs, value_targets, global_targets):
    return apply_symmetry_policy(policy, symmetry, binary_inputs, global_inputs)


def policy_mask(on_board):
    """The slots of a board of the rows' size: a placement on each point of the board, and each slide that stays on it."""
    on_board = on_board > 0
    n, size, _ = on_board.shape
    channels = [on_board]
    for length in range(1, MAX_SLIDE + 1):
        for dx, dy in DIRECTIONS:
            # The points whose point `length` steps on is on the board. The board is a square, so the path is on it too.
            x, y = dx * length, dy * length
            end = torch.zeros_like(on_board)
            end[:, max(0, -y):size - max(0, y), max(0, -x):size - max(0, x)] = \
                on_board[:, max(0, y):size - max(0, -y), max(0, x):size - max(0, -x)]
            channels.append(on_board & end)
    moves = torch.stack(channels, dim=3).reshape(n, -1)
    return torch.cat((moves, torch.zeros((n, 1), dtype=torch.bool, device=moves.device)), dim=1)
