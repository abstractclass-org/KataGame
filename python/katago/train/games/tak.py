"""Tak's module for the trainer (katago/train/games/__init__.py lists the members): its facts, as the C++ encoding part
gives them (game/tak/gameencoding.h), then the augmentation of its training rows and the policy's static mask.

MAX_BOARD_LEN is the one place of the network's board. CMake reads its number for a build of katago
(cpp/CMakeLists.txt, KATAGAME_NETWORK_BOARD_LEN), so a build and this module have the same board and channels.

A move is one action: the channel of its square says the placement's kind, or the spread's direction and drop pattern
(C++ gamerules.h). A spread's direction turns with the board: a flip of x swaps > and <, a flip of y swaps + and -, and
the transpose swaps + with > and - with <. The input planes of the road groups' edges turn alike. The other planes and
the value targets move as pixels. The image of a row does not depend on its position, so both policy targets map alike.

The rows of a 5 x 5 board are in the corner of the network's 6 x 6, and a flip moves them to another side, as KataGo's
rows of a board smaller than the network's move. So the policy's mask reads the board from the on-board plane.
"""

import functools

import torch

from . import (TARGET_FUTURE_PIECES, TARGET_INCOME, TARGET_OWNERSHIP, TARGET_SCORING, TARGET_SEKI,
               apply_symmetry_points)

# The members of the game's module (C++ game/tak/gameencoding.h).
NAME = "tak"
MAX_BOARD_LEN = 6        # the first network's board. A build of katago takes it from this line
PLACEMENTS, DIRECTIONS = 3, 4        # the channels of a flat, a wall and a capstone, then 4 of each drop pattern
NUM_PATTERNS = 2 ** MAX_BOARD_LEN - 1  # the drop patterns of a pickup of 1 to MAX_BOARD_LEN pieces
POLICY_CHANNELS = PLACEMENTS + DIRECTIONS * NUM_PATTERNS
NUM_SPATIAL_INPUTS = 35  # 36 with the on-board plane
NUM_GLOBAL_INPUTS = 14
HISTORY_PLANE = 30       # the squares that each of the last 5 moves changed, newest first
NUM_HISTORY_PLANES = 5
SYMMETRIES = 0xFF
# C0: the final flats, on flat endings. C1: the squares captured before the end. C2-3: the top owners 4 and 16 plies
# later. C4: the winning road's squares, on road endings. C5-6: the captures of the row's move and of the reply.
TARGETS = TARGET_OWNERSHIP | TARGET_SEKI | TARGET_FUTURE_PIECES | TARGET_SCORING | TARGET_INCOME
SCORE_SCALE = 0.2        # assumed until the pilot measures the flat margins (gameencoding.h)
GAME_LENGTH = 57.0       # assumed until the pilot measures it
DRAWS_AS_NO_RESULT = True
# The mover's road, the opponent's road, the mover's flat win, the opponent's flat win, a flat tie, the 50-ply rule.
ENDING_CLASSES = 6
LEGAL_MOVES_IN_ROWS = True

# The input planes that the augmentation and the income mask read. A plane's index in a batch is 1 + its index here:
# KataGo's on-board plane comes first.
MY_TOP, THEIR_TOP = 0, 3   # + the kind (flat, wall, capstone): the top piece
EDGE_PLANE = 20            # + 4 * side + direction: the square's road group touches the edge of that direction


def _direction_image(direction, symmetry):
    """The image of a direction, + - > < numbered 0 to 3 (C++ sym_direction)."""
    dx, dy = ((0, 1), (0, -1), (1, 0), (-1, 0))[direction]
    if symmetry & 2:
        dx = -dx
    if symmetry & 1:
        dy = -dy
    if symmetry & 4:
        dx, dy = dy, dx
    return {(0, 1): 0, (0, -1): 1, (1, 0): 2, (-1, 0): 3}[(dx, dy)]


def _sources(symmetry):
    """The direction whose image is each direction: the channel and the plane to read at each direction's place."""
    sources = [0] * DIRECTIONS
    for d in range(DIRECTIONS):
        sources[_direction_image(d, symmetry)] = d
    return sources


def apply_symmetry_inputs(binary_inputs, symmetry):
    """The input planes (N, C, y, x)."""
    if symmetry == 0:
        return binary_inputs
    order = list(range(binary_inputs.shape[1]))
    for side in range(2):
        first = 1 + EDGE_PLANE + DIRECTIONS * side
        order[first:first + DIRECTIONS] = [first + d for d in _sources(symmetry)]
    return apply_symmetry_points(binary_inputs, symmetry)[:, order]


def apply_symmetry_policy(policy, symmetry, binary_inputs, global_inputs):
    """Policy-shaped targets (N, C, MAX_BOARD_LEN^2 * POLICY_CHANNELS + 1): the squares move, each spread's channel goes
    to its direction's image, and pass stays last."""
    if symmetry == 0:
        return policy
    n, c, _ = policy.shape
    points = policy[:, :, :-1].reshape(n, c, MAX_BOARD_LEN, MAX_BOARD_LEN, POLICY_CHANNELS).movedim(-1, 2)
    points = apply_symmetry_points(points, symmetry).movedim(2, -1)
    order = list(range(PLACEMENTS))
    sources = _sources(symmetry)
    for pattern in range(NUM_PATTERNS):
        order += [PLACEMENTS + DIRECTIONS * pattern + d for d in sources]
    return torch.cat((points[..., order].reshape(n, c, -1), policy[:, :, -1:]), dim=2)


def apply_symmetry_policy_targets(policy, symmetry, binary_inputs, global_inputs, value_targets, global_targets):
    """The policy targets (N, 2, policy): a move's image does not depend on the position, so both channels move
    alike."""
    return apply_symmetry_policy(policy, symmetry, binary_inputs, global_inputs)


@functools.lru_cache(maxsize=None)
def _patterns():
    """Each drop pattern's pieces and steps (C++ pattern_pieces, pattern_steps): a pickup of k pieces has the patterns
    2^(k-1) - 1 to 2^k - 2, and a step ends at each 1 bit of the pattern's code, and at the last piece."""
    pieces, steps = [], []
    for pattern in range(NUM_PATTERNS):
        k = (pattern + 1).bit_length()
        pieces.append(k)
        steps.append(bin(pattern + 1 - 2 ** (k - 1)).count("1") + 1)
    return torch.tensor(pieces), torch.tensor(steps)


def policy_mask(on_board):
    """The moves that a position of the rows' board can have (N, policy), from the on-board plane (N, y, x): a placement
    on each square of the board, and a spread whose steps stay on the board with a pickup of at most its side. Pass is
    never legal."""
    on = on_board > 0.5
    n, device = on.shape[0], on.device
    side = on.any(dim=1).sum(dim=1)  # the files of the board: it is square
    # room[:, d, y, x]: the squares of the board beyond the square in the direction.
    room = torch.zeros((n, DIRECTIONS, MAX_BOARD_LEN, MAX_BOARD_LEN), dtype=torch.long, device=device)
    for d, (dim, sign) in enumerate(((1, 1), (1, -1), (2, 1), (2, -1))):  # + and - along y, > and < along x
        reach = on
        for _ in range(MAX_BOARD_LEN - 1):
            shifted = torch.zeros_like(reach)
            if sign > 0:
                shifted.narrow(dim, 0, MAX_BOARD_LEN - 1).copy_(reach.narrow(dim, 1, MAX_BOARD_LEN - 1))
            else:
                shifted.narrow(dim, 1, MAX_BOARD_LEN - 1).copy_(reach.narrow(dim, 0, MAX_BOARD_LEN - 1))
            reach = shifted & on
            room[:, d] += reach
    pieces, steps = (t.to(device) for t in _patterns())
    fits = steps[None, :, None, None, None] <= room[:, None]           # (N, pattern, direction, y, x)
    fits = fits & (pieces[None, :] <= side[:, None])[:, :, None, None, None] & on[:, None, None]
    spreads = fits.reshape(n, NUM_PATTERNS * DIRECTIONS, MAX_BOARD_LEN, MAX_BOARD_LEN)
    places = on[:, None].expand(n, PLACEMENTS, MAX_BOARD_LEN, MAX_BOARD_LEN)
    moves = torch.cat((places, spreads), dim=1).permute(0, 2, 3, 1).reshape(n, -1)
    return torch.cat((moves, torch.zeros((n, 1), dtype=torch.bool, device=device)), dim=1)


def income_mask(binary_inputs):
    """The squares where the loss of C5-6 reads the target (N, 2, y, x): C5 the opponent's top pieces, C6 the
    mover's."""
    planes = binary_inputs[:, 1:]
    return torch.stack((planes[:, THEIR_TOP:THEIR_TOP + 3].sum(dim=1), planes[:, MY_TOP:MY_TOP + 3].sum(dim=1)), dim=1)
