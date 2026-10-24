"""Blueprints' module for the trainer (katago/train/games/__init__.py lists the members): its facts, as the C++
encoding part gives them (game/blueprints/gameencoding.h).

The game allows the identity only: the site's frame of a blueprint (its lowest cell at 0,0) is not kept by a symmetry
of the hex grid, so a symmetry would move the hand's points and the placements' anchors in ways that are not pixels.
So the augmentation has nothing to do.
"""

from . import TARGET_OWNERSHIP

# The members of the game's module (C++ game/blueprints/gameencoding.h).
NAME = "blueprints"
MAX_BOARD_LEN = 41       # the axial square |q|, |r| <= 20
POLICY_CHANNELS = 9      # place blueprint 1-3, pick from blueprint 1-3, drop into blueprint 1-3
NUM_SPATIAL_INPUTS = 38  # 39 with the on-board plane
NUM_GLOBAL_INPUTS = 17
HISTORY_PLANE = 0
NUM_HISTORY_PLANES = 0   # a position holds the whole game state
SYMMETRIES = 1 << 0      # the identity only
TARGETS = TARGET_OWNERSHIP   # C0: the final board (+1 the mover's cells, -1 the opponent's)
SCORE_SCALE = 12.0
GAME_LENGTH = 120.0      # the site's moves of a game


def _identity_only(symmetry):
    if symmetry != 0:
        raise ValueError(f"Blueprints allows the identity only, not symmetry {symmetry}")


def apply_symmetry_inputs(binary_inputs, symmetry):
    """The input planes (N, C, y, x)."""
    _identity_only(symmetry)
    return binary_inputs


def apply_symmetry_policy(policy, symmetry, binary_inputs, global_inputs):
    """Policy-shaped targets (N, C, MAX_BOARD_LEN^2 * POLICY_CHANNELS + 1)."""
    _identity_only(symmetry)
    return policy


def apply_symmetry_policy_targets(policy, symmetry, binary_inputs, global_inputs, value_targets, global_targets):
    """The policy targets (N, 2, policy)."""
    _identity_only(symmetry)
    return policy
