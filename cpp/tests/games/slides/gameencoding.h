#pragma once
// The network encoding part of the test game (gamerules.h): each player's pieces, the mover's side and the opening, the
// 13 channels of a point's moves, and the identity alone (a test of the symmetries is the trainer's,
// test_legal_moves.py, which has the same facts). Its rows hold their legal moves.

#include "../../../core/hash.h"
#include "../../../game/gameinterface.h"
#include "gamerules.h"

namespace slides {

struct GameEncoding {
  using Rules = GameRules;
  using Position = GameRules::Position;

  static constexpr int NUM_SPATIAL_INPUTS = 2;
  static constexpr int NUM_GLOBAL_INPUTS = 2;
  static constexpr int HISTORY_PLANE = 0, NUM_HISTORY_PLANES = 0;

  template<class Set>
  static void spatial_inputs(const Position& p, const GameInterface::InputOptions&, Set&& set) {
    for(int y = 0; y < p.size; y++)
      for(int x = 0; x < p.size; x++)
        if(p.cell[y * LEN + x] >= 0)
          set(x, y, p.cell[y * LEN + x] == GameRules::next_player(p) ? 0 : 1);
  }
  static void global_inputs(const Position& p, const GameInterface::InputOptions&, float* out) {
    out[0] = GameRules::next_player(p) == 0 ? 1.0f : 0.0f;
    out[1] = p.moves < OPENING_PLIES ? 1.0f : 0.0f;
  }
  static Hash128 input_hash(const Position&) { return Hash128(); }

  static constexpr int POLICY_CHANNELS = CHANNELS;
  static void action_point(int action, int& x, int& y, int& channel) {
    x = action / CHANNELS % LEN;
    y = action / CHANNELS / LEN;
    channel = action % CHANNELS;
  }
  static int point_action(int x, int y, int channel) { return (y * LEN + x) * CHANNELS + channel; }

  static constexpr unsigned SYMMETRIES = 1u << 0;
  static Position sym_position(const Position& p, int) { return p; }
  struct SymActionMap {
    SymActionMap(const Position&, int) {}
    int operator()(int action) const { return action; }
  };

  static constexpr unsigned TARGETS = 0;

  static constexpr bool LEGAL_MOVES_IN_ROWS = true;

  static constexpr double SCORE_SCALE = 1.0;
  static constexpr double SCORE_UNIT = 6.0;
  static constexpr double GAME_LENGTH = 22.0;
};

static_assert(GameInterface::checkEncoding<GameEncoding>(), "the test game's network encoding part");

}  // namespace slides
