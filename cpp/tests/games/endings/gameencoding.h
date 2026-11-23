#pragma once
// The network encoding part of the test game (gamerules.h): each player's pieces, the mover's side and the komi, one
// policy channel a point, the identity alone (the score and the corner have no symmetry), and the final board as the
// ownership. Its draws are a class of their own.

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

  static constexpr unsigned TARGETS = GameInterface::TARGET_OWNERSHIP;
  template<class Positions>
  static unsigned training_targets(const Positions& positions, int numPositions, int, int mover, float* values) {
    const Position& end = positions(numPositions - 1);
    for(int i = 0; i < POINTS; i++)
      values[i] = end.cell[i] < 0 ? 0.0f : end.cell[i] == mover ? 1.0f : -1.0f;
    return GameInterface::TARGET_OWNERSHIP;
  }

  static constexpr bool DRAWS_AS_NO_RESULT = true;

  static constexpr double SCORE_SCALE = 1.0;
  static constexpr double SCORE_UNIT = 4.0;
  static constexpr double GAME_LENGTH = 14.0;
};

static_assert(GameInterface::checkEncoding<GameEncoding>(), "the test game's network encoding part");

}  // namespace endings
