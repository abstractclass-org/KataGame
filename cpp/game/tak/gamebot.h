#pragma once
// Tak's bot part of the game interface (game/botinterface.h), for the alpha-beta search (search/alphabeta.h): the flat
// margin under the komi and the road groups' reach, and no move ordering. It is the gauntlet's start line (the engine
// repository's games/tak/docs/design/09-strength-play-and-analysis.md), not a strong engine.

#include <algorithm>

#include "../botinterface.h"
#include "gamerules.h"

namespace tak {

// The komi that the bot plays for, in half-flats, added to Black's flat count (as result_with_komi). A position does
// not hold its komi, so the match sets it, for the whole process.
inline int bot_half_komi = 0;

// The rules part as the search reads it: the ends of a game, under the bot's komi.
struct BotRules : GameRules {
  static float result(const Position& p) { return float(result_with_komi(p, bot_half_komi)); }

  // The search adds the mover's score minus the opponent's to an end's value, to choose among ends of one result. Tak
  // pays no margin, so the winner's score is minus the plies played: an earlier win is worth more, and so is a later
  // loss. A draw is worth 0.
  static double score(const Position& p, int player) {
    const int white = result_with_komi(p, bot_half_komi);
    return white != 0 && player == (white > 0 ? 0 : 1) ? -double(p.ply) : 0.0;
  }

  // The search's table keeps values from one search to the next, so a position under another komi is another key.
  static Hash128 hash(const Position& p) {
    Hash128 h = GameRules::hash(p);
    h.hash0 ^= mix64(uint64_t(int64_t(bot_half_komi)));
    return h;
  }
};

// The reach of the player's road groups (joined flats and capstones on top): the most ranks or files that one group
// spans. A road spans the board's side.
inline int road_reach(const Position& p, int player) {
  const uint64_t notWest = ~column_mask(0), notEast = ~column_mask(MAX_LEN - 1);
  const uint64_t row = (uint64_t(1) << p.size) - 1;
  uint64_t left = p.owner[player] & ~p.walls;
  int reach = 0;
  while(left) {
    uint64_t group = left & (~left + 1), last = 0;  // the lowest square left, then every square that joins it
    while(group != last) {
      last = group;
      group |= (((group & notEast) << 1) | ((group & notWest) >> 1) | (group << MAX_LEN) | (group >> MAX_LEN)) & left;
    }
    left &= ~group;
    // A group is connected, so the files that it has a square on are consecutive, and so are the ranks.
    uint64_t files = 0;
    int ranks = 0;
    for(int y = 0; y < p.size; y++) {
      const uint64_t squares = (group >> (y * MAX_LEN)) & row;
      files |= squares;
      ranks += squares != 0;
    }
    reach = std::max(reach, std::max(popcount(files), ranks));
  }
  return reach;
}

struct GameBot {
  using Rules = BotRules;

  // In flats, for the player to move:
  // - the flat margin under the komi, as the rules count it at a flat ending: the result if the game ended now;
  // - the reach of the longest road group, minus the opponent's, because the other way to win is a group that reaches
  //   across. One square of reach counts as one flat: a placed flat gives either of them, at the same price of one
  //   ply. So a flat that extends the group is worth more than one elsewhere, and a wall, which counts for neither, is
  //   placed only against what the search sees the opponent do.
  static double evaluate(const Position& p) {
    const double white = popcount(flats(p, 0)) - popcount(flats(p, 1)) - 0.5 * bot_half_komi +
                         road_reach(p, 0) - road_reach(p, 1);
    return GameRules::next_player(p) == 0 ? white : -white;
  }
};

static_assert(GameInterface::checkBot<GameBot>(), "Tak's bot part");

}  // namespace tak
