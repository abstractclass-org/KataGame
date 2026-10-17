"""Connect-4's module for the trainer (katago/train/games/__init__.py lists the members): its facts, as the C++
encoding part gives them (game/connect4/gameencoding.h), then the mirror of its training rows.

A move is its landing point, with one policy channel per point. So the input planes, the policy and the value targets
all move as pixels, as KataGo moves Go's. The mirror flips x on the network's board, 7 x 7. The rows of the standard
board (7 x 6) fill its width, so the flip is the C++ mirror: column x goes to column 6 - x. The rows of a narrower
board move to the other side of the network's board, as KataGo's rows of a Go board smaller than the network's do.
"""

import torch

from . import TARGET_OWNERSHIP, apply_symmetry_points

# The members of the game's module (C++ game/connect4/gameencoding.h).
NAME = "connect4"
MAX_BOARD_LEN = 7
POLICY_CHANNELS = 1   # a move is its landing point
NUM_SPATIAL_INPUTS = 3   # the mover's discs, the opponent's discs, the landing points: 4 with the on-board plane
NUM_GLOBAL_INPUTS = 1    # 1 if the mover is the first player
HISTORY_PLANE = 0
NUM_HISTORY_PLANES = 0   # a position holds the whole game state
SYMMETRIES = (1 << 0) | (1 << 2)   # the identity and the mirror (KataGo's symmetry 2, flip x)
TARGETS = TARGET_OWNERSHIP   # C0: the final board (+1 the mover's discs, -1 the opponent's)
SCORE_SCALE = 1.0   # the solver's score, in discs
GAME_LENGTH = 41.0  # a game of perfect play on 7 x 6


def apply_symmetry_inputs(binary_inputs, symmetry):
    """The input planes (N, C, y, x)."""
    return apply_symmetry_points(binary_inputs, symmetry)


def apply_symmetry_policy(policy, symmetry, binary_inputs, global_inputs):
    """Policy-shaped targets (N, C, MAX_BOARD_LEN^2 + 1): the points (y * MAX_BOARD_LEN + x) move, pass stays last."""
    if symmetry == 0:
        return policy
    n, c, _ = policy.shape
    points = policy[:, :, :-1].reshape(n, c, MAX_BOARD_LEN, MAX_BOARD_LEN)
    return torch.cat((apply_symmetry_points(points, symmetry).reshape(n, c, -1), policy[:, :, -1:]), dim=2)


def apply_symmetry_policy_targets(policy, symmetry, binary_inputs, global_inputs, value_targets, global_targets):
    """The policy targets (N, 2, policy): a move's image does not depend on the position, so both channels move
    alike."""
    return apply_symmetry_policy(policy, symmetry, binary_inputs, global_inputs)
