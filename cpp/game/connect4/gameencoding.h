#pragma once
// Connect-4's network encoding part of the game interface (game/gameinterface.h). The engine repository's
// games/connect4/README.md describes the inputs, the targets and the score.
//
// - The inputs: the mover's discs, the opponent's discs and the landing points (where a disc of each column that is
//   not full lands), after the front's on-board plane, and one global input: whether the mover is the first player.
//   No history planes: a position holds the whole game state.
// - The policy: one channel per point. A move is its landing point (gamerules.h, action_at), and pass is never legal.
// - The symmetries: the identity and the left-right mirror (KataGo's symmetry 2, flip x), which keep gravity.
// - The training targets: the final board as ownership (C0). The value targets (a draw counts as half a win) and the
//   score targets are KataGo's, from the rules' result and score: the score is the solver's (gamerules.h, score).

#include <cstdint>

#include "../../core/hash.h"
#include "../gameinterface.h"
#include "gamerules.h"

namespace connect4 {

struct GameEncoding {
  using Rules = GameRules;
  using Position = GameRules::Position;

  static constexpr int MOVER_PLANE = 0, OPPONENT_PLANE = 1, LANDING_PLANE = 2;
  static constexpr int NUM_SPATIAL_INPUTS = 3;
  static constexpr int NUM_GLOBAL_INPUTS = 1;
  static constexpr int HISTORY_PLANE = 0, NUM_HISTORY_PLANES = 0;

  static void sym_point(int& x, int& y, int symmetry, const Position& p) {
    GameInterface::symPoint(x, y, symmetry, p.xSize, p.ySize);
  }

  template<class Set>
  static void spatial_inputs(const Position& p, const GameInterface::InputOptions& options, Set&& set) {
    const uint64_t mine = p.discs[GameRules::next_player(p)];
    auto image = [&](int x, int y, int plane) {
      sym_point(x, y, options.symmetry, p);
      set(x, y, plane);
    };
    for(int x = 0; x < p.xSize; x++) {
      for(int y = 0; y < p.height[x]; y++)
        image(x, y, (mine & point_bit(x, y)) ? MOVER_PLANE : OPPONENT_PLANE);
      if(p.height[x] < p.ySize)
        image(x, p.height[x], LANDING_PLANE);
    }
  }
  // Connect-4 has no komi: its runs play at komi 0.
  static void global_inputs(const Position& p, const GameInterface::InputOptions&, float* out) {
    out[0] = GameRules::next_player(p) == 0 ? 1.0f : 0.0f;
  }
  // The inputs read the position only.
  static Hash128 input_hash(const Position&) { return Hash128(); }

  static constexpr int POLICY_CHANNELS = 1;
  static void action_point(int action, int& x, int& y, int& channel) {
    x = action_x(action);
    y = action_y(action);
    channel = 0;
  }
  static int point_action(int x, int y, int) { return action_at(x, y); }

  // The identity and the mirror (symmetry 2, flip x). A flip of y or a transpose would break gravity.
  static constexpr unsigned SYMMETRIES = (1u << 0) | (1u << 2);

  // The image under an allowed symmetry: each disc at its point's image. An allowed symmetry keeps the rows, so a
  // column's height moves with the column.
  static Position sym_position(const Position& p, int symmetry) {
    Position s = p;
    s.discs[0] = s.discs[1] = 0;
    for(int x = 0; x < p.xSize; x++) {
      int sx = x, sy = 0;
      sym_point(sx, sy, symmetry, p);
      s.height[sx] = p.height[x];
      for(int y = 0; y < p.height[x]; y++) {
        sx = x;
        sy = y;
        sym_point(sx, sy, symmetry, p);
        s.discs[owner(p, x, y)] |= point_bit(sx, sy);
      }
    }
    return s;
  }
  // A move's image is the image of its landing point.
  struct SymActionMap {
    int symmetry;
    Position board;  // the board's size, for the images of points
    SymActionMap(const Position& p, int sym) : symmetry(sym), board(p) {}
    int operator()(int action) const {
      if(action == GameRules::PASS_ACTION || symmetry == 0)
        return action;
      int x = action_x(action), y = action_y(action);
      sym_point(x, y, symmetry, board);
      return action_at(x, y);
    }
  };

  // C0, the final board as ownership: +1 on the mover's discs, -1 on the opponent's, 0 on an empty point.
  static constexpr unsigned TARGETS = GameInterface::TARGET_OWNERSHIP;
  template<class Positions>
  static unsigned training_targets(const Positions& positions, int numPositions, int, int mover, float* values) {
    const Position& end = positions(numPositions - 1);
    int8_t owners[MAX_LEN * MAX_LEN];
    GameRules::area(end, owners);
    for(int i = 0; i < end.xSize * end.ySize; i++)
      values[i] = owners[i] < 0 ? 0.0f : owners[i] == mover ? 1.0f : -1.0f;
    return GameInterface::TARGET_OWNERSHIP;
  }

  // The score is the solver's, in discs (gamerules.h): KataGo's bins of 1 point fit it.
  static constexpr double SCORE_SCALE = 1.0;
  // KataGo's rule, the square root of the board's area, on the standard board: sqrt(7 * 6).
  static constexpr double SCORE_UNIT = 6.48074069840786;
  // The length of a game of perfect play on the standard board: the first player wins with its last disc, the 41st
  // (the solver's score of the start position is 1).
  static constexpr double GAME_LENGTH = 41.0;
};

static_assert(GameInterface::checkEncoding<GameEncoding>(), "Connect-4's network encoding part");

}  // namespace connect4
