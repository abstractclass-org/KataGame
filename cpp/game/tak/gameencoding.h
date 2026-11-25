#pragma once
// Tak's network encoding part of the game interface (game/gameinterface.h). The engine repository's
// games/tak/docs/design/ gives the reasons: 03 (the policy), 05 (the inputs), 06 (the targets) and 08 (the constants).
//
// - The inputs: 35 planes, all from the side of the player to move (the mover): the top pieces, the pieces below them
//   to depth 7 and whether pieces lie deeper, each road group's edges, the empty squares that complete a road, and the
//   squares that each of the last 5 moves changed. 14 global inputs: the reserves, the board's side, the komi, the flat
//   margin, the seat, the opening and the draw clock. Inputs version 14 adds KataGo's 2 playout inputs and AZdb's
//   latent.
// - The policy: one action is one whole move, CHANNELS a square (gamerules.h). A smaller board is in the corner of the
//   network's board, and an action keeps its number there.
// - The symmetries: all 8. A square moves as a pixel of the board, and the directions of the spreads and of the edge
//   planes move with it (sym_direction).
// - The training targets: those of one ending (C0, C4), the captures (C1, C5-6) and the top owners ahead (C2-3). A draw
//   is a class of its own, and the endings have 6 classes.
//
// The trainer's module has the same values (python/katago/train/games/tak.py). Its MAX_BOARD_LEN is the largest board
// of this build of katago (gamerules.h, TAK_MAX_BOARD_LEN), so a larger board is the one number there. The planes of
// the pieces below the top then need the depth of its carry limit plus one (TOP_DEPTHS), which the check at the end of
// this file asks for.

#include <algorithm>
#include <cstdint>

#include "../../core/hash.h"
#include "../gameinterface.h"
#include "diversity.h"
#include "gamerules.h"
#include "roads.h"

namespace tak {

// The image of a move's direction under one of KataGo's symmetries (bit 2 flips x, bit 1 flips y, then bit 4
// transposes): a flip of x swaps > and <, a flip of y swaps + and -, and the transpose swaps + with > and - with <.
inline int sym_direction(int direction, int symmetry) {
  int dx = (symmetry & 2) ? -DX[direction] : DX[direction];
  int dy = (symmetry & 1) ? -DY[direction] : DY[direction];
  if(symmetry & 4)
    std::swap(dx, dy);
  return dy > 0 ? 0 : dy < 0 ? 1 : dx > 0 ? 2 : 3;
}
// The image of a square of a board of that side.
inline int sym_square(int i, int symmetry, int size) {
  int x = i % MAX_LEN, y = i / MAX_LEN;
  GameInterface::symPoint(x, y, symmetry, size, size);
  return point(x, y);
}
inline uint64_t sym_squares(uint64_t squares, int symmetry, int size) {
  uint64_t image = 0;
  for(; squares; squares &= squares - 1)
    image |= bit(sym_square(__builtin_ctzll(squares), symmetry, size));
  return image;
}

struct GameEncoding {
  using Rules = GameRules;
  using Position = GameRules::Position;

  // The planes. A side is 0 for the mover and 1 for the opponent, a kind is FLAT, WALL or CAP, and a direction is PTN's
  // (+, -, > and <, 0 to 3).
  static constexpr int TOP_DEPTHS = 7;        // the depths from the top that have planes, the top piece's included
  static constexpr int TOP_PLANE = 0;         // + 3 * side + kind: the top piece
  static constexpr int BELOW_PLANE = 6;       // + 2 * (depth - 2) + side: the piece at depth 2 to TOP_DEPTHS
  static constexpr int REST_PLANE = 18;       // + side: that player has a piece deeper than TOP_DEPTHS
  static constexpr int EDGE_PLANE = 20;       // + 4 * side + direction: the square's road group touches that edge
  static constexpr int COMPLETES_PLANE = 28;  // + side: a flat on this empty square completes that player's road
  static constexpr int HISTORY_PLANE = 30, NUM_HISTORY_PLANES = HISTORY;  // the squares of the last moves, newest first
  static constexpr int NUM_SPATIAL_INPUTS = 35;
  static_assert(BELOW_PLANE == TOP_PLANE + 6 && REST_PLANE == BELOW_PLANE + 2 * (TOP_DEPTHS - 1) &&
                  EDGE_PLANE == REST_PLANE + 2 && COMPLETES_PLANE == EDGE_PLANE + 8 &&
                  HISTORY_PLANE == COMPLETES_PLANE + 2 && NUM_SPATIAL_INPUTS == HISTORY_PLANE + NUM_HISTORY_PLANES,
                "the planes follow one another");

  template<class Set>
  static void spatial_inputs(const Position& p, const GameInterface::InputOptions& options, Set&& set) {
    const int me = GameRules::next_player(p), symmetry = options.symmetry;
    auto image = [&](int i, int plane) {
      const int j = sym_square(i, symmetry, p.size);
      set(j % MAX_LEN, j / MAX_LEN, plane);
    };
    auto all = [&](uint64_t squares, int plane) {
      for(; squares; squares &= squares - 1)
        image(__builtin_ctzll(squares), plane);
    };
    for(uint64_t on = occupied(p); on; on &= on - 1) {
      const int i = __builtin_ctzll(on), height = p.height[i];
      image(i, TOP_PLANE + 3 * (colour_at(p, i, 0) == me ? 0 : 1) + kind_at(p, i));
      for(int depth = 2; depth <= height && depth <= TOP_DEPTHS; depth++)
        image(i, BELOW_PLANE + 2 * (depth - 2) + (colour_at(p, i, depth - 1) == me ? 0 : 1));
      if(height > TOP_DEPTHS) {
        // The bits above a stack's height are 0.
        const int black = popcount(p.top[i] >> TOP_DEPTHS) + popcount(p.rest[i]);
        const int deep[2] = {height - TOP_DEPTHS - black, black};
        for(int side = 0; side < 2; side++)
          if(deep[side == 0 ? me : 1 - me] > 0)
            image(i, REST_PLANE + side);
      }
    }
    for(int side = 0; side < 2; side++) {
      const Roads r = roads(p, side == 0 ? me : 1 - me);
      for(int d = 0; d < 4; d++)
        all(r.edge[d], EDGE_PLANE + 4 * side + sym_direction(d, symmetry));
      all(r.completing, COMPLETES_PLANE + side);
    }
    if(options.history)
      for(int h = 0; h < HISTORY; h++)
        all(p.history[h], HISTORY_PLANE + h);
  }

  static constexpr int RESERVES_INPUT = 0;     // the mover's stones / 50 and capstones / 2, then the opponent's
  static constexpr int SIZE_INPUT = 4;         // + side - 5: one-hot over 5 to 8
  static constexpr int KOMI_INPUT = 8;         // the mover's komi / KOMI_UNIT
  static constexpr int MARGIN_INPUT = 9;       // the mover's flats on top less the opponent's / KOMI_UNIT, without komi
  static constexpr int FIRST_PLAYER_INPUT = 10;  // 1 when the mover is White, the first player
  static constexpr int OPENING_INPUT = 11;     // 1 when this ply places the opponent's flat
  static constexpr int DOUBLE_STACK_INPUT = 12;  // 1 while the double black stack's opening is in progress
  static constexpr int DRAW_CLOCK_INPUT = 13;  // the plies since the last placement or flattening / FIFTY_PLIES
  static constexpr int NUM_GLOBAL_INPUTS = 14;
  // Fixed scales on every board: the reserves read 8 x 8's counts, and komi and the margin one unit, playtak's largest
  // komi of 4 flats, which no change of SCORE_SCALE moves.
  static constexpr float STONES_UNIT = 50.0f, CAPSTONES_UNIT = 2.0f, KOMI_UNIT = 4.0f;

  static void global_inputs(const Position& p, const GameInterface::InputOptions& options, float* out) {
    const int me = GameRules::next_player(p);
    for(int side = 0; side < 2; side++) {
      const int who = side == 0 ? me : 1 - me;
      out[RESERVES_INPUT + 2 * side] = p.stones[who] / STONES_UNIT;
      out[RESERVES_INPUT + 2 * side + 1] = p.capstones[who] / CAPSTONES_UNIT;
    }
    for(int size = 5; size <= 8; size++)
      out[SIZE_INPUT + size - 5] = p.size == size ? 1.0f : 0.0f;
    out[KOMI_INPUT] = float(me == 0 ? 0.0 - options.komi : options.komi) / KOMI_UNIT;
    out[MARGIN_INPUT] = float(popcount(flats(p, me)) - popcount(flats(p, 1 - me))) / KOMI_UNIT;
    out[FIRST_PLAYER_INPUT] = me == 0 ? 1.0f : 0.0f;
    out[OPENING_INPUT] = p.ply < 2 ? 1.0f : 0.0f;
    out[DOUBLE_STACK_INPUT] = p.ply < 2 && opening == DOUBLE_BLACK_STACK ? 1.0f : 0.0f;
    out[DRAW_CLOCK_INPUT] = p.drawClock / float(FIFTY_PLIES);
  }

  // The inputs read the last moves' squares beyond hash(p), which holds the whole state of the rules.
  static Hash128 input_hash(const Position& p) {
    Hash128 h;
    for(int k = 0; k < HISTORY; k++)
      if(p.history[k])
        h ^= Hash128(mix64(mix64(0xA4093822299F31D0ULL + uint64_t(k)) ^ p.history[k]),
                     mix64(mix64(0x082EFA98EC4E6C89ULL + uint64_t(k)) ^ p.history[k]));
    return h;
  }

  static constexpr int POLICY_CHANNELS = CHANNELS;
  static void action_point(int action, int& x, int& y, int& channel) {
    const int i = tak::action_point(action);
    x = i % MAX_LEN;
    y = i / MAX_LEN;
    channel = action_channel(action);
  }
  static int point_action(int x, int y, int channel) { return point(x, y) * CHANNELS + channel; }

  // A road can join either pair of opposite edges, and nothing else in the rules names a direction.
  static constexpr unsigned SYMMETRIES = 0xFF;

  static Position sym_position(const Position& p, int symmetry) {
    if(symmetry == 0)
      return p;
    Position s = p;
    std::fill(s.top, s.top + NUM_POINTS, uint64_t(0));
    std::fill(s.rest, s.rest + NUM_POINTS, uint64_t(0));
    std::fill(s.height, s.height + NUM_POINTS, uint8_t(0));
    for(uint64_t on = occupied(p); on; on &= on - 1) {
      const int i = __builtin_ctzll(on), j = sym_square(i, symmetry, p.size);
      s.top[j] = p.top[i];
      s.rest[j] = p.rest[i];
      s.height[j] = p.height[i];
    }
    for(uint64_t* squares: {&s.owner[0], &s.owner[1], &s.walls, &s.caps})
      *squares = sym_squares(*squares, symmetry, p.size);
    for(uint64_t& squares: s.history)
      squares = sym_squares(squares, symmetry, p.size);
    return s;
  }
  // A placement's image is its square's. A spread's is its square's with its direction's, and the drop pattern stays.
  struct SymActionMap {
    int symmetry, size;
    SymActionMap(const Position& p, int sym) : symmetry(sym), size(p.size) {}
    int operator()(int action) const {
      if(action == GameRules::PASS_ACTION || symmetry == 0)
        return action;
      const int j = sym_square(tak::action_point(action), symmetry, size), channel = action_channel(action);
      if(channel < 3)
        return place_action(j, channel);
      return spread_action(j, sym_direction(channel_direction(channel), symmetry), channel_pattern(channel));
    }
  };

  // The training targets, in the view of the row's mover (+1 the mover's, -1 the opponent's). A move captures a square
  // when it drops pieces on it and the square's top passes from one player to the other. A spread also uncovers the
  // square that it leaves, whose top can pass to the other player: that is no capture.
  // - C0, in a game that the flat count ended: the flats on top at the end, 0 elsewhere. They sum to the flat margin
  //   without komi.
  // - C1, in a game that its rules ended, as classes on the squares with a top piece: +1 when a move captures the
  //   mover's square before the game ends, -1 when one captures the opponent's, 0 when none does. A square counts even
  //   if its owner takes it back, or comes back to it after leaving it. Class 2 on the empty squares, which the loss
  //   leaves out.
  // - C2-3, the owners of the top pieces FUTURE_PLIES[0] and [1] plies later, 0 on an empty square, or at the end of a
  //   game that ends before. A record that stops before the later ply, at the move cap, has neither.
  // - C4, in a game that a road ended: the squares of the winner's groups that join opposite edges, 0 elsewhere.
  // - C5, 1 on the opponent's top pieces of the row that the row's move captures. C6, 1 on the mover's that the reply
  //   captures. The loss reads only those players' top pieces (the trainer's income_mask). A game that the row's move
  //   ends has no reply, and no capture by it. A record that stops before the reply has neither.
  static constexpr unsigned TARGETS = GameInterface::TARGET_OWNERSHIP | GameInterface::TARGET_SEKI |
                                      GameInterface::TARGET_FUTURE_PIECES | GameInterface::TARGET_SCORING |
                                      GameInterface::TARGET_INCOME;
  static constexpr unsigned OWN_TARGETS = TARGETS;  // C2-3 too: the front's are 8 and 32 plies later
  static constexpr int FUTURE_PLIES[2] = {4, 16};

  // The squares of owner's that the move from before to after captures.
  static uint64_t captures(const Position& before, const Position& after, int owner) {
    uint64_t taken = 0;
    for(uint64_t m = before.owner[owner] & after.owner[1 - owner]; m; m &= m - 1) {
      const int i = __builtin_ctzll(m);
      if(after.height[i] > before.height[i])  // a drop: the square that a spread leaves gets lower
        taken |= bit(i);
    }
    return taken;
  }

  template<class Positions>
  static unsigned training_targets(const Positions& positions, int numPositions, int row, int mover, float* values) {
    const Position& now = positions(row);
    const Position& end = positions(numPositions - 1);
    const bool ended = GameRules::terminal(end);
    const int size = now.size, area = size * size, opponent = 1 - mover;
    auto fill = [&](int slot, uint64_t squares, float value) {
      for(; squares; squares &= squares - 1) {
        const int i = __builtin_ctzll(squares);
        values[slot * area + i / MAX_LEN * size + i % MAX_LEN] = value;
      }
    };
    unsigned held = 0;

    if(end.end == FLATS) {
      fill(0, flats(end, mover), 1.0f);
      fill(0, flats(end, opponent), -1.0f);
      held |= GameInterface::TARGET_OWNERSHIP;
    }
    if(ended) {
      uint64_t taken[2] = {0, 0};
      for(int j = row; j + 1 < numPositions; j++)
        for(int owner = 0; owner < 2; owner++)
          taken[owner] |= captures(positions(j), positions(j + 1), owner);
      fill(1, board_mask(size) & ~occupied(now), 2.0f);
      fill(1, now.owner[mover] & taken[mover], 1.0f);
      fill(1, now.owner[opponent] & taken[opponent], -1.0f);
      held |= GameInterface::TARGET_SEKI;
    }
    if(ended || row + FUTURE_PLIES[1] < numPositions) {
      for(int h = 0; h < 2; h++) {
        const Position& later = positions(std::min(row + FUTURE_PLIES[h], numPositions - 1));
        fill(2 + h, later.owner[mover], 1.0f);
        fill(2 + h, later.owner[opponent], -1.0f);
      }
      held |= GameInterface::TARGET_FUTURE_PIECES;
    }
    if(end.end == ROAD) {
      fill(4, roads(end, end.roadWinner).winning(), end.roadWinner == mover ? 1.0f : -1.0f);
      held |= GameInterface::TARGET_SCORING;
    }
    if(row + 2 < numPositions || (ended && row + 1 < numPositions)) {
      const Position& next = positions(row + 1);
      fill(5, captures(now, next, opponent), 1.0f);
      if(row + 2 < numPositions)
        fill(6, now.owner[mover] & captures(next, positions(row + 2), mover), 1.0f);
      held |= GameInterface::TARGET_INCOME;
    }
    return held;
  }

  // A flat tie after komi and the 50-ply rule are draws, KataGo's no result.
  static constexpr bool DRAWS_AS_NO_RESULT = true;

  // The classes of the endings: a road and a flat count, each won by the mover or by the opponent, then the two draws.
  static constexpr int ENDING_CLASSES = 6;
  static int ending_class(const GameInterface::GameEnd& end, int mover) {
    const bool won = end.result == (mover == 0 ? 1 : -1);
    switch(end.kind) {
    case ROAD_WIN: return won ? 0 : 1;
    case FLAT_WIN: return won ? 2 : 3;
    case FLAT_TIE: return 4;
    default: return 5;  // FIFTY_DRAW
    }
  }

  // 9,181 slots on 6 x 6, of which a position has about 100 legal: the soft policy's floor is on these alone.
  static constexpr bool LEGAL_MOVES_IN_ROWS = true;

  // AZdb's statistics of a position (diversity.h).
  using Diversity = tak::Diversity;

  // The score is the flat margin with komi, in flats. Its scale against Go's is assumed until the pilot measures the
  // margins of the games that end by the flat count (design 08, group 1: their median over KataGo's score of about 20).
  // The median of playtak's is 3 flats on 5 x 5 and on 6 x 6, so 0.15. 5 / 32 is the number next to it that ends the
  // range of komi and of the score, (36 + 60) * SCORE_SCALE = 15 flats, on the half-flat grid (the front clips a komi
  // there and rounds it to the grid).
  static constexpr double SCORE_SCALE = 0.15625;
  // KataGo's rule, the square root of the area, on the largest board. Unused: Tak's searches have no score utility.
  static constexpr double SCORE_UNIT = MAX_LEN;
  // Assumed until the pilot measures it: people's 45.7 plies on 5 x 5 and 62.1 on 6 x 6, at the mix of 3 to 7.
  static constexpr double GAME_LENGTH = 57.0;
  // The komi sampler draws the flats that the config gives, on every size.
  static constexpr double KOMI_AREA = 19 * 19;
  // No smoothing of the lead: flat counts give margins of both parities.
  static constexpr double KOMI_LEAD_STEP = 0.0;
};

static_assert(GameInterface::checkEncoding<GameEncoding>(), "Tak's network encoding part");
#ifdef KATAGAME_NETWORK_BOARD_LEN
static_assert(GameEncoding::TOP_DEPTHS > MAX_LEN,
              "a network's board of side N needs the planes of a stack to depth N + 1: a move picks up at most N "
              "pieces, so the piece below them can come to the top");
#endif

}  // namespace tak
