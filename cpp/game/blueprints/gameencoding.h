#pragma once
// Blueprints' network encoding part of the game interface (game/gameinterface.h). The engine repository's
// games/blueprints/README.md describes the inputs, the targets and the score.
//
// - The inputs, after the front's on-board plane: on the board, each player's cells by type, each player's frontier
//   (the empty points next to its cells), each player's cells that a stamp next to them could take (not walls, next
//   to an empty point), and for each of the mover's blueprints, the anchors where it can go and where it takes 1 or
//   more and 3 or more enemy stars. Then each player's blueprints, drawn at the hand's points (gamerules.h, HAND_X and
//   HAND_Y), where the policy's picks and drops are, and the picked cell. No history planes: a position holds the
//   whole game state.
// - The global inputs: the mover's seat, its moves left in the turn, the game's progress, the stars on the board,
//   the scores, and the pick.
// - The policy: 9 channels a point, the actions of gamerules.h.
// - The symmetries: the identity only. The site's frame of a blueprint (its lowest cell at 0,0) is not kept by a
//   symmetry of the hex grid, so the images of the hand's points and of the placements' anchors are not pixels.
// - The training targets: the final board as ownership (C0). The value targets and the score targets are KataGo's,
//   from the rules' result and score.

#include <cmath>
#include <cstdint>

#include "../../core/hash.h"
#include "../gameinterface.h"
#include "gamerules.h"

namespace blueprints {

struct GameEncoding {
  using Rules = GameRules;
  using Position = GameRules::Position;

  // The planes: the mover's cells (plain, star, wall), the opponent's, the frontiers, the cells open to a stamp, the
  // mover's placements (each blueprint: legal, 1+ stars, 3+ stars), the mover's blueprints (cells, stars, walls), the
  // opponent's, and the picked cell.
  static constexpr int MINE = 0, THEIRS = 3, MY_FRONTIER = 6, THEIR_FRONTIER = 7, MY_OPEN = 8, THEIR_OPEN = 9;
  static constexpr int PLACES = 10, TAKES_ONE = 13, TAKES_THREE = 16;
  static constexpr int MY_HAND = 19, THEIR_HAND = 28, PICKED = 37;
  static constexpr int NUM_SPATIAL_INPUTS = 38;
  static constexpr int NUM_GLOBAL_INPUTS = 17;
  static constexpr int HISTORY_PLANE = 0, NUM_HISTORY_PLANES = 0;

  template<class Set>
  static void spatial_inputs(const Position& p, const GameInterface::InputOptions&, Set&& set) {
    const int mover = GameRules::next_player(p), opponent = 1 - mover;
    for(int y = 0; y < LEN; y++)
      for(int x = 0; x < LEN; x++) {
        const int8_t v = p.cell[y * LEN + x];
        if(v) {
          const int owner = cell_owner(v), type = cell_type(v);
          set(x, y, (owner == mover ? MINE : THEIRS) + type);
          if(type != WALL) {
            bool open = false;
            for(const auto& d : DIRS)
              open = open || (on_board(x + d[0], y + d[1]) && !p.cell[(y + d[1]) * LEN + x + d[0]]);
            if(open)
              set(x, y, owner == mover ? MY_OPEN : THEIR_OPEN);
          }
        }
        else {
          if(next_to_own(p, x, y, mover))
            set(x, y, MY_FRONTIER);
          if(next_to_own(p, x, y, opponent))
            set(x, y, THEIR_FRONTIER);
        }
      }
    // The mover's placements, when no pick waits for its drop.
    if(p.picked < 0 && !GameRules::terminal(p)) {
      for(int k = 0; k < HAND; k++) {
        const Blueprint& b = p.hand[mover][k];
        for(int y = 0; y < LEN; y++)
          for(int x = 0; x < LEN; x++) {
            if(!can_place(p, k, x, y))
              continue;
            set(x, y, PLACES + k);
            const int stars = stars_taken(p, mover, b, x, y);
            if(stars >= 1)
              set(x, y, TAKES_ONE + k);
            if(stars >= 3)
              set(x, y, TAKES_THREE + k);
          }
      }
    }
    for(int player = 0; player < 2; player++) {
      const int base = player == mover ? MY_HAND : THEIR_HAND;
      for(int k = 0; k < HAND; k++) {
        const Blueprint& b = p.hand[player][k];
        for(int i = 0; i < b.n; i++) {
          const int x = b.q[i] + HAND_X, y = b.r[i] + HAND_Y;
          set(x, y, base + k);
          if(b.t[i] == STAR)
            set(x, y, base + HAND + k);
          if(b.t[i] == WALL)
            set(x, y, base + 2 * HAND + k);
        }
      }
    }
    if(p.picked >= 0)
      set(p.pickQ + HAND_X, p.pickR + HAND_Y, PICKED);
  }

  // The enemy stars that blueprint b takes with its (0, 0) cell at (x, y), each once.
  static int stars_taken(const Position& p, int player, const Blueprint& b, int x, int y) {
    int taken[6 * MAX_CELLS], n = 0, stars = 0;
    for(int i = 0; i < b.n; i++)
      for(const auto& d : DIRS) {
        const int nx = x + b.q[i] + d[0], ny = y + b.r[i] + d[1];
        const int8_t v = at(p, nx, ny);
        if(cell_owner(v) != 1 - player || cell_type(v) != STAR)
          continue;
        const int index = ny * LEN + nx;
        bool counted = false;
        for(int j = 0; j < n && !counted; j++)
          counted = taken[j] == index;
        if(!counted) {
          taken[n++] = index;
          stars++;
        }
      }
    return stars;
  }

  // A score on a scale where 100 points is 1 and 10,000 is 2: the scores grow with the stars on the board.
  static float squash(double points) {
    const double a = std::log10(1.0 + std::fabs(points) / 10.0) / 2.0;
    return float(points < 0 ? -a : a);
  }
  static void global_inputs(const Position& p, const GameInterface::InputOptions&, float* out) {
    const int mover = GameRules::next_player(p), opponent = 1 - mover;
    for(int i = 0; i < NUM_GLOBAL_INPUTS; i++)
      out[i] = 0.0f;
    out[0] = mover == 0 ? 1.0f : 0.0f;
    out[1] = GameRules::turn_moves_left(p) >= 2 ? 1.0f : 0.0f;
    out[2] = float(p.ply) / PLIES;
    out[3] = float(PLIES - p.ply) / PLIES;
    out[4] = float(p.stars[mover]) / 50.0f;
    out[5] = float(p.stars[opponent]) / 50.0f;
    out[6] = squash(p.scores[mover]);
    out[7] = squash(p.scores[opponent]);
    out[8] = squash(double(p.scores[mover]) - p.scores[opponent]);
    if(p.picked >= 0) {
      out[9] = 1.0f;
      out[10 + p.picked] = 1.0f;
      out[13 + p.pickType] = 1.0f;
      out[16] = p.pickConnected ? 1.0f : 0.0f;
    }
  }
  // The inputs read the position only.
  static Hash128 input_hash(const Position&) { return Hash128(); }

  static constexpr int POLICY_CHANNELS = CHANNELS;
  static void action_point(int action, int& x, int& y, int& channel) {
    x = action_x(action);
    y = action_y(action);
    channel = action_channel(action);
  }
  static int point_action(int x, int y, int channel) { return action_at(x, y, channel); }

  // The identity only (above).
  static constexpr unsigned SYMMETRIES = 1u << 0;
  static Position sym_position(const Position& p, int) { return p; }
  struct SymActionMap {
    SymActionMap(const Position&, int) {}
    int operator()(int action) const { return action; }
  };

  // C0, the final board as ownership: +1 on the mover's cells, -1 on the opponent's, 0 on an empty point.
  static constexpr unsigned TARGETS = GameInterface::TARGET_OWNERSHIP;
  template<class Positions>
  static unsigned training_targets(const Positions& positions, int numPositions, int, int mover, float* values) {
    const Position& end = positions(numPositions - 1);
    int8_t owners[AREA];
    GameRules::area(end, owners);
    for(int i = 0; i < AREA; i++)
      values[i] = owners[i] < 0 ? 0.0f : owners[i] == mover ? 1.0f : -1.0f;
    return GameInterface::TARGET_OWNERSHIP;
  }

  // A capture late in a game scores tens of stars times tens of stars: margins of hundreds of points, against Go's
  // tens. So KataGo's constants in points scale by 12, as Enclosure's, and the search's unit is 200 points.
  static constexpr double SCORE_SCALE = 12.0;
  static constexpr double SCORE_UNIT = 200.0;
  // A game is 120 of the site's moves (moves_played counts them; a pick and its drop are one).
  static constexpr double GAME_LENGTH = 120.0;
};

static_assert(GameInterface::checkEncoding<GameEncoding>(), "Blueprints' network encoding part");

}  // namespace blueprints
