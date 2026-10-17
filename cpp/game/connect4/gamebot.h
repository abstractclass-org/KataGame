#pragma once
// Connect-4's bot part of the game interface (game/botinterface.h), for the alpha-beta search (search/alphabeta.h):
// a heuristic of centre columns and open threes, and the centre columns first.

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "../botinterface.h"
#include "gamerules.h"

namespace connect4 {

// The empty points where a disc of `discs` makes four in a row (has_four), if the discs hold no four yet.
inline int open_threes(uint64_t discs, uint64_t empty) {
  int count = 0;
  for(uint64_t rest = empty; rest != 0; rest &= rest - 1)
    count += has_four(discs | (rest & (~rest + 1)));  // the lowest point left
  return count;
}

// A column's distance from the centre, in half columns: 0 for the centre column of an odd width.
inline int centre_distance(const Position& p, int x) { return std::abs(2 * x - (p.xSize - 1)); }

struct GameBot {
  using Rules = GameRules;

  static constexpr double OPEN_THREE = 8.0;

  // The mover's points minus the opponent's. A disc scores its column's closeness to the centre: on 7 columns, 6 in the
  // centre, then 4, 2 and 0 at the edges. Each empty point that would complete the player's four scores OPEN_THREE.
  static double evaluate(const Position& p) {
    const int mover = Rules::next_player(p);
    const uint64_t mine = p.discs[mover], theirs = p.discs[1 - mover];
    uint64_t board = 0;
    double value = 0.0;
    for(int x = 0; x < p.xSize; x++) {
      const uint64_t column = ((uint64_t(1) << p.ySize) - 1) << (COLUMN_BITS * x);
      board |= column;
      value += (p.xSize - 1 - centre_distance(p, x)) *
               (__builtin_popcountll(mine & column) - __builtin_popcountll(theirs & column));
    }
    const uint64_t empty = board & ~(mine | theirs);  // a position that is not terminal: neither player has four
    return value + OPEN_THREE * (open_threes(mine, empty) - open_threes(theirs, empty));
  }

  // The centre columns first, then outwards. The two columns of a pair keep their order: an insertion sort, stable and
  // without the allocation of std::stable_sort, for 7 moves at most.
  static void order_actions(const Position& p, std::vector<int>& actions) {
    for(size_t i = 1; i < actions.size(); i++) {
      const int action = actions[i], distance = centre_distance(p, action_x(action));
      size_t j = i;
      for(; j > 0 && centre_distance(p, action_x(actions[j - 1])) > distance; j--)
        actions[j] = actions[j - 1];
      actions[j] = action;
    }
  }
};

static_assert(GameInterface::checkBot<GameBot>(), "Connect-4's bot part");

}  // namespace connect4
