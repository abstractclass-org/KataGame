#pragma once
// Blueprints' network encoding part of the game interface (game/gameinterface.h). The engine repository's
// games/blueprints/README.md describes the inputs, the targets and the score.
//
// - The inputs, after the front's on-board plane, from the mover's side: on the board, each player's cells by type,
//   each player's frontier (the empty points next to its cells), each player's cells open to a stamp (not walls, next
//   to an empty point), and for each blueprint of each player the anchors where it fits and where it takes 1 or more
//   and 3 or more enemy stars, and each player's cells that some placement of the other player takes. Then each
//   player's blueprints, drawn at the hand's points (gamerules.h, HAND_X and HAND_Y), where the policy's picks and
//   drops are, and the picked cell. Then the history: the points that each of the last 4 actions stamped and took.
// - The global inputs: the mover's seat, its moves left in the turn, the game's progress, the stars on the board, the
//   scores, and the pick.
// - The policy: 9 channels a point, the actions of gamerules.h.
// - The symmetries: the 4 of KataGo's 8 that keep the hex neighbours (DIRS) on the axial square: the identity, the
//   turn by 180 degrees (3) and two reflections (4, 7). A board's point moves as a pixel. A blueprint moves cell by
//   cell, then goes back to the site's frame (its lowest cell at 0,0): so the hand's points, the anchors of the
//   placements and the picks and drops move as pixels and then by each blueprint's shift. The inputs at a symmetry are
//   the inputs of the position's image (sym_position).
// - The training targets: the final board as ownership (C0). The value targets and the score targets are KataGo's,
//   from the rules' result and score.

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <utility>

#include "../../core/hash.h"
#include "../gameinterface.h"
#include "gamerules.h"

namespace blueprints {

// The linear part of an allowed symmetry on axial offsets, as GameInterface::symPoint moves the board's points around
// its middle: flip x (q to -q), flip y (r to -r), then transpose (q and r swap).
inline void sym_offset(int& q, int& r, int symmetry) {
  if(symmetry & 0x2)
    q = -q;
  if(symmetry & 0x1)
    r = -r;
  if(symmetry & 0x4)
    std::swap(q, r);
}

// The shift that puts the lowest cell of the image of b's cells (with the cell (extraQ, extraR) when withExtra) at
// (0, 0): the site's frame of the image. (0, 0) for no cell.
inline void frame_shift(const Blueprint& b, int symmetry, bool withExtra, int extraQ, int extraR, int& mq, int& mr) {
  bool any = false;
  mq = mr = 0;
  auto consider = [&](int q, int r) {
    sym_offset(q, r, symmetry);
    if(!any || r < mr || (r == mr && q < mq)) {
      mq = q;
      mr = r;
      any = true;
    }
  };
  for(int i = 0; i < b.n; i++)
    consider(b.q[i], b.r[i]);
  if(withExtra)
    consider(extraQ, extraR);
}

inline int sym_index(int index, int symmetry) {
  int x = index % LEN, y = index / LEN;
  GameInterface::symPoint(x, y, symmetry, LEN, LEN);
  return y * LEN + x;
}

struct GameEncoding {
  using Rules = GameRules;
  using Position = GameRules::Position;

  // The planes. Each pair of bases names the mover's planes first, then the opponent's.
  static constexpr int MINE = 0, THEIRS = 3;                 // + type
  static constexpr int MY_FRONTIER = 6, THEIR_FRONTIER = 7;
  static constexpr int MY_OPEN = 8, THEIR_OPEN = 9;
  static constexpr int MY_PLACES = 10, THEIR_PLACES = 19;    // + k: fits; + 3 + k: takes 1+ stars; + 6 + k: 3+ stars
  static constexpr int MINE_TAKEN = 28, THEIRS_TAKEN = 29;   // cells that some placement of the other player takes
  static constexpr int MY_HAND = 30, THEIR_HAND = 39;        // + k: cells; + 3 + k: stars; + 6 + k: walls
  // + k: the picked cell of blueprint k. A plane for each blueprint, so that the planes alone give the blueprint's
  // frame (the trainer's augmentation reads no global input).
  static constexpr int PICKED = 48;
  static constexpr int HISTORY_PLANE = 51, NUM_HISTORY_PLANES = 2 * HISTORY;  // action h: + 2h stamped, + 2h + 1 taken
  static constexpr int NUM_SPATIAL_INPUTS = HISTORY_PLANE + NUM_HISTORY_PLANES;  // 59
  static constexpr int NUM_GLOBAL_INPUTS = 17;

  template<class Set>
  static void spatial_inputs(const Position& p, const GameInterface::InputOptions& options, Set&& set) {
    if(options.symmetry == 0)
      write_inputs(p, options.history, set);
    else
      write_inputs(sym_position(p, options.symmetry), options.history, set);
  }

  template<class Set>
  static void write_inputs(const Position& p, bool history, Set& set) {
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
    // The placements of each player, on the board as it is: the mover's when no pick waits for its drop, and the
    // opponent's as if it moved now (its threats).
    if(!GameRules::terminal(p)) {
      for(int player : {mover, opponent}) {
        if(player == mover && p.picked >= 0)
          continue;
        const int places = player == mover ? MY_PLACES : THEIR_PLACES;
        const int taken = player == mover ? THEIRS_TAKEN : MINE_TAKEN;
        for(int k = 0; k < HAND; k++) {
          const Blueprint& b = p.hand[player][k];
          for_each_placement(p, player, b, [&](int x, int y) {
            set(x, y, places + k);
            int16_t points[MAX_TAKEN];
            const int n = taken_by(p, player, b, x, y, points);
            int stars = 0;
            for(int i = 0; i < n; i++) {
              stars += cell_type(p.cell[points[i]]) == STAR;
              set(points[i] % LEN, points[i] / LEN, taken);
            }
            if(stars >= 1)
              set(x, y, places + HAND + k);
            if(stars >= 3)
              set(x, y, places + 2 * HAND + k);
          });
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
      set(p.pickQ + HAND_X, p.pickR + HAND_Y, PICKED + p.picked);
    if(history)
      for(int h = 0; h < HISTORY; h++) {
        const Footprint& f = p.history[h];
        for(int i = 0; i < f.nStamped; i++)
          set(f.stamped[i] % LEN, f.stamped[i] / LEN, HISTORY_PLANE + 2 * h);
        for(int i = 0; i < f.nTaken; i++)
          set(f.taken[i] % LEN, f.taken[i] / LEN, HISTORY_PLANE + 2 * h + 1);
      }
  }

  // A score on a scale where 100 points is 1 and 10,000 is 2: the scores grow with the stars on the board.
  static float squash(double points) {
    const double a = std::log10(1.0 + std::fabs(points) / 10.0) / 2.0;
    return float(points < 0 ? -a : a);
  }
  // The same at every symmetry: no input names a point.
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
  // What the inputs read beyond hash(): the history's points.
  static Hash128 input_hash(const Position& p) {
    Hash128 h;
    for(int t = 0; t < HISTORY; t++) {
      const Footprint& f = p.history[t];
      for(int i = 0; i < f.nStamped; i++)
        h ^= key(60000 + uint64_t(t) * 2 * AREA + uint64_t(f.stamped[i]), 0);
      for(int i = 0; i < f.nTaken; i++)
        h ^= key(60000 + uint64_t(t) * 2 * AREA + AREA + uint64_t(f.taken[i]), 0);
    }
    return h;
  }

  static constexpr int POLICY_CHANNELS = CHANNELS;
  static void action_point(int action, int& x, int& y, int& channel) {
    x = action_x(action);
    y = action_y(action);
    channel = action_channel(action);
  }
  static int point_action(int x, int y, int channel) { return action_at(x, y, channel); }

  static constexpr unsigned SYMMETRIES = (1u << 0) | (1u << 3) | (1u << 4) | (1u << 7);

  // The image: each cell of the board and of the history at its point's image, each blueprint's cells at their
  // images in the site's frame. The rest of a picked blueprint and the picked cell keep the frame of the whole
  // blueprint's image, as the rest keeps the whole blueprint's frame until the drop.
  static Position sym_position(const Position& p, int symmetry) {
    if(symmetry == 0)
      return p;
    Position s = p;
    for(int i = 0; i < AREA; i++)
      s.cell[i] = 0;
    for(int i = 0; i < AREA; i++)
      if(p.cell[i])
        s.cell[sym_index(i, symmetry)] = p.cell[i];
    const int mover = GameRules::next_player(p);
    for(int player = 0; player < 2; player++)
      for(int k = 0; k < HAND; k++) {
        const Blueprint& b = p.hand[player][k];
        const bool rest = player == mover && k == p.picked;
        int mq, mr;
        frame_shift(b, symmetry, rest, p.pickQ, p.pickR, mq, mr);
        Blueprint& image = s.hand[player][k];
        for(int i = 0; i < b.n; i++) {
          int q = b.q[i], r = b.r[i];
          sym_offset(q, r, symmetry);
          image.q[i] = int8_t(q - mq);
          image.r[i] = int8_t(r - mr);
        }
        if(rest) {
          int q = p.pickQ, r = p.pickR;
          sym_offset(q, r, symmetry);
          s.pickQ = int8_t(q - mq);
          s.pickR = int8_t(r - mr);
        }
      }
    for(int h = 0; h < HISTORY; h++) {
      Footprint& f = s.history[h];
      for(int i = 0; i < f.nStamped; i++)
        f.stamped[i] = int16_t(sym_index(f.stamped[i], symmetry));
      for(int i = 0; i < f.nTaken; i++)
        f.taken[i] = int16_t(sym_index(f.taken[i], symmetry));
    }
    return s;
  }

  // A legal action's image, an action of sym_position: a placement's anchor is the lowest point of its stamped cells'
  // images, and a pick's or a drop's hand point moves by its blueprint's frame (frame_shift).
  struct SymActionMap {
    int symmetry;
    Blueprint hand[HAND];  // the mover's
    int picked, pickQ, pickR;
    SymActionMap(const Position& p, int sym)
      : symmetry(sym), picked(p.picked), pickQ(p.pickQ), pickR(p.pickR) {
      for(int k = 0; k < HAND; k++)
        hand[k] = p.hand[GameRules::next_player(p)][k];
    }
    int operator()(int action) const {
      if(symmetry == 0 || action == GameRules::PASS_ACTION)
        return action;
      const int x = action_x(action), y = action_y(action), channel = action_channel(action);
      if(channel < PICK) {
        const Blueprint& b = hand[channel - PLACE];
        int ax = 0, ay = 0;
        for(int i = 0; i < b.n; i++) {
          int sx = x + b.q[i], sy = y + b.r[i];
          GameInterface::symPoint(sx, sy, symmetry, LEN, LEN);
          if(i == 0 || sy < ay || (sy == ay && sx < ax)) {
            ax = sx;
            ay = sy;
          }
        }
        return action_at(ax, ay, channel);
      }
      const int k = channel < DROP ? channel - PICK : channel - DROP;
      int mq, mr;
      frame_shift(hand[k], symmetry, k == picked, pickQ, pickR, mq, mr);
      int q = x - HAND_X, r = y - HAND_Y;
      sym_offset(q, r, symmetry);
      return action_at(q - mq + HAND_X, r - mr + HAND_Y, channel);
    }
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
