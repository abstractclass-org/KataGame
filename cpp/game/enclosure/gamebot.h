#pragma once
// Enclosure's bot part of the game interface (game/botinterface.h), for the alpha-beta search (search/alphabeta.h):
// the heuristic of the greedy bot (games/enclosure/bots.py in the engine repository), and no move ordering.

#include <algorithm>

#include "../botinterface.h"
#include "engine.h"
#include "gamerules.h"

namespace enclosure {

struct GameBot {
  using Rules = GameRules;

  // The scoring events left, the end of the current turn included (bots.py, remaining_turn_ends).
  static int turn_ends_left(const Position& p) {
    if(p.terminal())
      return 0;
    return 1 + (std::max(0, MAX_MOVES - p.move_count - p.moves_left) + 1) / 2;
  }

  // The final margin of the player to move if every region stayed as it is until the end (bots.py, projected_scores).
  // All the moves of one position leave the same number of scoring events, and a move that ends the turn scores the
  // areas after it. So between the moves of one position, the value changes as the change in (own area - enemy area)
  // that the greedy bot maximises, times a positive factor: at depth 1, the search chooses among the greedy bot's
  // moves.
  static double evaluate(const Position& p) {
    const int mover = p.player;
    auto area = [&p](int player) { return p.area_ok[player] ? p.areas[player] : enclosed_area(p.edges[player]); };
    return p.scores[mover] - p.scores[1 - mover] + turn_ends_left(p) * (area(mover) - area(1 - mover));
  }
};

static_assert(GameInterface::checkBot<GameBot>(), "Enclosure's bot part");

}  // namespace enclosure
