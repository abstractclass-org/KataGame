#pragma once
// The network encoding part of the test game (gamerules.h): each player's pieces, the mover's side and the komi, one
// policy channel a point, the identity alone (the score and the corner have no symmetry). Its draws are a class of
// their own, and its endings have classes and targets of their own (training_targets).

#include <algorithm>
#include <cstdint>

#include "../../../core/hash.h"
#include "../../../game/gameinterface.h"
#include "gamerules.h"

namespace endings {

struct GameEncoding {
  using Rules = GameRules;
  using Position = GameRules::Position;

  static constexpr int NUM_SPATIAL_INPUTS = 2;
  static constexpr int NUM_GLOBAL_INPUTS = 2;
  static constexpr int HISTORY_PLANE = 0, NUM_HISTORY_PLANES = 0;

  template<class Set>
  static void spatial_inputs(const Position& p, const GameInterface::InputOptions&, Set&& set) {
    for(int i = 0; i < POINTS; i++)
      if(p.cell[i] >= 0)
        set(i % LEN, i / LEN, p.cell[i] == GameRules::next_player(p) ? 0 : 1);
  }
  static void global_inputs(const Position& p, const GameInterface::InputOptions& options, float* out) {
    const bool first = GameRules::next_player(p) == 0;
    out[0] = first ? 1.0f : 0.0f;
    out[1] = (float)(first ? -options.komi : options.komi);
  }
  static Hash128 input_hash(const Position&) { return Hash128(); }

  static constexpr int POLICY_CHANNELS = 1;
  static void action_point(int action, int& x, int& y, int& channel) {
    x = action % LEN;
    y = action / LEN;
    channel = 0;
  }
  static int point_action(int x, int y, int) { return y * LEN + x; }

  static constexpr unsigned SYMMETRIES = 1u << 0;
  static Position sym_position(const Position& p, int) { return p; }
  struct SymActionMap {
    SymActionMap(const Position&, int) {}
    int operator()(int action) const { return action; }
  };

  // The targets of a row, each in the view of the row's mover (+1 the mover's, -1 the opponent's):
  // - C0, in a game that ended by the count: the final board.
  // - C1, in a game that its rules ended: the final owner of each point that is empty in the row, 2 on the others.
  // - C2-3, in every row: the board 2 and 4 moves later.
  // - C4, in a game that ended by a line: the points of the winner's lines.
  // - C5-6, in a row with a reply: the point of the row's move, and the point of the reply (1 there).
  static constexpr unsigned TARGETS =
    GameInterface::TARGET_OWNERSHIP | GameInterface::TARGET_SEKI | GameInterface::TARGET_FUTURE_PIECES |
    GameInterface::TARGET_SCORING | GameInterface::TARGET_INCOME;
  static constexpr unsigned OWN_TARGETS = TARGETS;
  template<class Positions>
  static unsigned training_targets(const Positions& positions, int numPositions, int row, int mover, float* values) {
    auto side = [mover](int8_t owner) { return owner < 0 ? 0.0f : owner == mover ? 1.0f : -1.0f; };
    const Position& now = positions(row);
    const Position& end = positions(numPositions - 1);
    unsigned held = GameInterface::TARGET_FUTURE_PIECES;
    for(int i = 0; i < POINTS; i++) {
      values[2 * POINTS + i] = side(positions(std::min(row + 2, numPositions - 1)).cell[i]);
      values[3 * POINTS + i] = side(positions(std::min(row + 4, numPositions - 1)).cell[i]);
    }
    if(GameRules::terminal(end)) {
      held |= GameInterface::TARGET_SEKI;
      for(int i = 0; i < POINTS; i++)
        values[1 * POINTS + i] = now.cell[i] >= 0 ? 2.0f : side(end.cell[i]);
      const int winner = line_winner(end);
      if(winner >= 0) {
        held |= GameInterface::TARGET_SCORING;
        for(int i = 0; i < POINTS; i++)
          if(on_line(end, i))
            values[4 * POINTS + i] = side((int8_t)winner);
      }
      else if(end.moves == POINTS) {
        held |= GameInterface::TARGET_OWNERSHIP;
        for(int i = 0; i < POINTS; i++)
          values[i] = side(end.cell[i]);
      }
    }
    if(row + 2 < numPositions) {
      held |= GameInterface::TARGET_INCOME;
      for(int i = 0; i < POINTS; i++) {
        values[5 * POINTS + i] = positions(row + 1).cell[i] != now.cell[i];
        values[6 * POINTS + i] = positions(row + 2).cell[i] != positions(row + 1).cell[i];
      }
    }
    return held;
  }

  static constexpr bool DRAWS_AS_NO_RESULT = true;

  static constexpr bool LEGAL_MOVES_IN_ROWS = true;

  // A line and a count, each for the mover and for the opponent, then the two draws.
  static constexpr int ENDING_CLASSES = 6;
  static int ending_class(const GameInterface::GameEnd& end, int mover) {
    const bool won = end.result == (mover == 0 ? 1 : -1);
    switch(end.kind) {
    case LINE: return won ? 0 : 1;
    case COUNT: return won ? 2 : 3;
    case COUNT_TIE: return 4;
    default: return 5;
    }
  }

  static constexpr double SCORE_SCALE = 1.0;
  static constexpr double SCORE_UNIT = 4.0;
  static constexpr double GAME_LENGTH = 14.0;
};

static_assert(GameInterface::checkEncoding<GameEncoding>(), "the test game's network encoding part");

}  // namespace endings
