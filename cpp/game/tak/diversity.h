#pragma once
// Tak's statistics of a position for AZdb's diversity reward (the engine repository's
// games/tak/docs/design/10-katagame-additions.md, "AZdb for Tak"), and their running normalization. The reference is
// the engine repository's games/tak/tools/diversity_stats.py: the same statistics in the same order, the same
// transforms, and the same side. GameEncoding will give them to KataGame's front (GameInterface: Diversity, for Tak's
// encoding part), whose training rows read them for the members' rewards (program/azdb.h). The Python binding gives
// them to the tests (games/tak/binding/extras.h). No input or search reads them yet.
//
// The statistics are from the side of a player P (AZdb takes the player who made the move), against its opponent O:
// each exists for P and for O, so White and Black give the same features for the same play. None reads a frame or a
// board size: each is a share of pieces or squares, or a size divided by the board's side. A share is 0 when its count
// of pieces is 0, and a height or an extent is 0 when the player has no such piece.
// - Walls and capstones (4): for P, then O: the walls among the player's top pieces, as a share of its top pieces, and
//   its capstones on the board, as a share of the capstones it starts with.
// - Stacks (6): for P, then O: the other player's pieces under the player's top pieces, as a share of the other
//   player's pieces on the board (captives); the player's own pieces under its top pieces, as a share of its pieces on
//   the board (support); and the height of its tallest stack, over the board's side N.
// - Roads (6): for P, then O: log(1 + n) of the empty squares where a flat of the player completes its road (a square
//   that the player's road already spans would count nothing, as a complete road is the result: 0 then; the reserve
//   does not matter); the player's longest reach, the larger of the rows and the columns that one road group spans,
//   over N; and the road pieces of its largest road group, as a share of its road pieces. A road piece is a flat or a
//   capstone on top, and a road group is a set of them joined by orthogonal steps.
// - Contact (4): for P, then O: the player's top pieces with an orthogonal neighbour of the other player's, as a share
//   of its top pieces; and the player's walls next to a road piece of the other player's, as a share of its walls.
// - Edges (2): for P, then O: the player's top pieces on the edge squares, as a share of its top pieces, over the edge
//   squares' share of the board, 4 (N - 1) / N^2: 1 for play that does not favour the edges.
// - Tempo (2): for P, then O: the share of the player's moves that were spreads. A player's moves are its plies, and
//   its placements are its starting reserve less its reserve now. The swap opening keeps this true: each player places
//   one piece of the other's and has one piece placed for it. Before both players have moved (ply 2), and before a
//   player has moved, the share is 0. Under the double black stack, Black's first piece is two flats from Black's
//   reserve, so Black's placements are one fewer than its reserve shows.
//
// The normalization (DiversityNormalizer): z = (x - mean) / max(sd, SIGMA_FLOOR), clipped to [-CLIP, CLIP], times
// 1 / sqrt(the size of its group). The mean and the standard deviation are running estimates, exponential moving
// averages over the finished sides as the design's estimate of ψ, seeded from the statistics of a reference's
// positions (games/tak/runs/diversity_seed.json, from the pilot).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "gamerules.h"

namespace tak {

struct DiversityNormalizer;

namespace diversity_detail {
// Column x's squares and the edge squares of each board side, at compile time (a constexpr function of the class could
// not initialize the class's own constants).
constexpr uint64_t column_bits(int x) {
  uint64_t m = 0;
  for(int y = 0; y < MAX_LEN; y++)
    m |= uint64_t(1) << (y * MAX_LEN + x);
  return m;
}
constexpr uint64_t edge_bits(int size) {
  uint64_t m = 0;
  for(int y = 0; y < size; y++)
    for(int x = 0; x < size; x++)
      if(x == 0 || y == 0 || x == size - 1 || y == size - 1)
        m |= uint64_t(1) << (y * MAX_LEN + x);
  return m;
}
constexpr std::array<uint64_t, MAX_LEN + 1> edge_table() {
  std::array<uint64_t, MAX_LEN + 1> t{};
  for(int s = MIN_LEN; s <= MAX_LEN; s++)
    t[s] = edge_bits(s);
  return t;
}
constexpr uint64_t NOT_WEST = ~column_bits(0), NOT_EAST = ~column_bits(MAX_LEN - 1);
constexpr std::array<uint64_t, MAX_LEN + 1> EDGE = edge_table();
}  // namespace diversity_detail

struct Diversity {
  static constexpr int NUM_STATISTICS = 24;
  // Each player's moves in a game: the divisor of a member's reward (GameInterface: Diversity), so that the return of
  // a whole game is a mean reward for each move. Half of GAME_LENGTH's 57 plies, until the pilot measures the length.
  static constexpr int MOVES = 28;
  using Normalizer = DiversityNormalizer;
  // The groups in the order of the statistics: walls and capstones, stacks, roads, contact, edges, tempo. Each group's
  // statistics are P's, then O's.
  static constexpr int NUM_GROUPS = 6;
  static constexpr int GROUP_SIZE[NUM_GROUPS] = {4, 6, 6, 4, 2, 2};
  static constexpr int KINDS = 0, STACKS = 4, ROADS = 10, CONTACT = 16, EDGES = 20, TEMPO = 22;
  static_assert(STACKS == KINDS + 4 && ROADS == STACKS + 6 && CONTACT == ROADS + 6 && EDGES == CONTACT + 4 &&
                  TEMPO == EDGES + 2 && NUM_STATISTICS == TEMPO + 2,
                "the groups follow one another");
  // ALPHA: the running estimates' decay at each finished side, ψ's (Blueprints' design, "The estimate of ψ").
  static constexpr double SIGMA_FLOOR = 0.1, CLIP = 3.0, ALPHA = 0.9995;

  // a / b, 0 when b is 0.
  static double share(int a, int b) { return b > 0 ? double(a) / double(b) : 0.0; }
  static uint64_t low_mask(int n) { return n >= 64 ? ~uint64_t(0) : (uint64_t(1) << n) - 1; }
  // The squares that touch a square of m orthogonally.
  static uint64_t neighbours(uint64_t m) {
    return ((m & diversity_detail::NOT_EAST) << 1) | ((m & diversity_detail::NOT_WEST) >> 1) | (m << MAX_LEN) |
           (m >> MAX_LEN);
  }
  // The black pieces of the stack on square i (its colour bits, the top at bit 0).
  static int black_pieces(const Position& p, int i) {
    const int h = p.height[i];
    const int low = h < 64 ? h : 64;
    int n = popcount(p.top[i] & low_mask(low));
    if(h > 64)
      n += popcount(p.rest[i] & low_mask(h - 64));
    return n;
  }
  // The player's pieces on the board, from its reserve.
  static int on_board(const Position& p, int who) {
    return STONES[p.size] + CAPSTONES[p.size] - p.stones[who] - p.capstones[who];
  }
  // The flats and capstones on top, the pieces of a road.
  static uint64_t road_pieces(const Position& p, int who) { return p.owner[who] & ~p.walls; }

  // The stacks of a player's top pieces.
  struct Stacks {
    int captives = 0;  // the other player's pieces under the player's top pieces
    int support = 0;   // the player's own pieces under its top pieces
    int tallest = 0;   // the pieces of its tallest stack
  };
  static Stacks stacks(const Position& p, int who) {
    Stacks s;
    uint64_t on = p.owner[who];
    while(on) {
      const int i = __builtin_ctzll(on);
      on &= on - 1;
      const int h = p.height[i], black = black_pieces(p, i);
      const int own = who == 1 ? black : h - black;
      s.captives += h - own;
      s.support += own - 1;  // the top piece is the player's
      if(h > s.tallest)
        s.tallest = h;
    }
    return s;
  }

  // The roads of a player's flats and capstones.
  struct Roads {
    int pieces = 0, largest = 0, extent = 0, threats = 0;
  };
  static Roads roads(const Position& p, int who) {
    enum { WEST = 1, EAST = 2, SOUTH = 4, NORTH = 8 };
    const int n = p.size;
    const uint64_t road = road_pieces(p, who);
    const uint64_t westEdge = diversity_detail::column_bits(0), eastEdge = diversity_detail::column_bits(n - 1);
    const uint64_t southEdge = (uint64_t(1) << n) - 1, northEdge = southEdge << ((n - 1) * MAX_LEN);
    Roads r;
    r.pieces = popcount(road);
    int8_t label[NUM_POINTS];
    uint8_t flags[NUM_POINTS];
    int groups = 0;
    bool complete = false;
    uint64_t rest = road;
    while(rest) {
      uint64_t group = rest & (~rest + 1), last = 0;
      while(group != last) {
        last = group;
        group |= neighbours(group) & road;
      }
      rest &= ~group;
      const int f = ((group & westEdge) ? WEST : 0) | ((group & eastEdge) ? EAST : 0) |
                    ((group & southEdge) ? SOUTH : 0) | ((group & northEdge) ? NORTH : 0);
      flags[groups] = uint8_t(f);
      complete = complete || spans(f);
      uint64_t cols = 0;
      for(int y = 0; y < n; y++)
        cols |= (group >> (y * MAX_LEN)) & ((uint64_t(1) << MAX_LEN) - 1);
      const int rows = (63 - __builtin_clzll(group)) / MAX_LEN - __builtin_ctzll(group) / MAX_LEN + 1;
      const int columns = (63 - __builtin_clzll(cols)) - __builtin_ctzll(cols) + 1;
      r.extent = std::max(r.extent, std::max(rows, columns));
      r.largest = std::max(r.largest, popcount(group));
      for(uint64_t g = group; g; g &= g - 1)
        label[__builtin_ctzll(g)] = int8_t(groups);
      groups++;
    }
    // A complete road cannot be completed (it is the result). Else each empty square joins the groups that touch it,
    // and the square itself touches the edges that it is on.
    if(complete)
      return r;
    uint64_t empty = board_mask(n) & ~occupied(p);
    while(empty) {
      const int e = __builtin_ctzll(empty);
      empty &= empty - 1;
      const int x = e % MAX_LEN, y = e / MAX_LEN;
      int f = (x == 0 ? WEST : 0) | (x == n - 1 ? EAST : 0) | (y == 0 ? SOUTH : 0) | (y == n - 1 ? NORTH : 0);
      for(uint64_t nb = neighbours(uint64_t(1) << e) & road; nb; nb &= nb - 1)
        f |= flags[label[__builtin_ctzll(nb)]];
      if(spans(f))
        r.threats++;
    }
    return r;
  }
  // Whether the edges (the flags of roads) join opposite sides.
  static bool spans(int f) { return (f & 3) == 3 || (f & 12) == 12; }

  // The share of the player's moves that were spreads (the header comment's "Tempo").
  static double spread_share(const Position& p, int who) {
    const int moves = who == 0 ? (p.ply + 1) / 2 : p.ply / 2;
    if(p.ply < 2 || moves <= 0)
      return 0.0;
    const int extra = who == 1 && opening == DOUBLE_BLACK_STACK ? 1 : 0;
    const int spreads = moves - (on_board(p, who) - extra);
    return double(std::max(0, spreads)) / double(moves);
  }

  static void statistics(const Position& p, int player, double* out) {
    const double N = p.size;
    const double edgeShare = double(4 * (p.size - 1)) / (N * N);
    for(int side = 0; side < 2; side++) {
      const int who = side == 0 ? player : 1 - player, other = 1 - who;
      const uint64_t mine = p.owner[who], theirs = p.owner[other];
      const int tops = popcount(mine);
      out[KINDS + 2 * side] = share(popcount(mine & p.walls), tops);
      out[KINDS + 2 * side + 1] = share(popcount(mine & p.caps), CAPSTONES[p.size]);

      const Stacks s = stacks(p, who);
      out[STACKS + 3 * side] = share(s.captives, on_board(p, other));
      out[STACKS + 3 * side + 1] = share(s.support, on_board(p, who));
      out[STACKS + 3 * side + 2] = s.tallest / N;

      const Roads r = roads(p, who);
      out[ROADS + 3 * side] = std::log1p(double(r.threats));
      out[ROADS + 3 * side + 1] = r.extent / N;
      out[ROADS + 3 * side + 2] = share(r.largest, r.pieces);

      const uint64_t walls = mine & p.walls;
      out[CONTACT + 2 * side] = share(popcount(mine & neighbours(theirs)), tops);
      out[CONTACT + 2 * side + 1] = share(popcount(walls & neighbours(road_pieces(p, other))), popcount(walls));

      out[EDGES + side] = share(popcount(mine & diversity_detail::EDGE[p.size]), tops) / edgeShare;
      out[TEMPO + side] = spread_share(p, who);
    }
  }

  // Each statistic's weight, 1 / sqrt(the size of its group), so that each group adds the same variance.
  static double weight(int i) {
    int g = 0, end = GROUP_SIZE[0];
    while(i >= end)
      end += GROUP_SIZE[++g];
    return 1.0 / std::sqrt(double(GROUP_SIZE[g]));
  }
};

// The running standardization of the statistics (Diversity, "The normalization"): each statistic's mean and standard
// deviation over the positions of every member, as exponential moving averages of x and of x^2. At each finished side
// (one player's positions in one game), m1 <- alpha m1 + (1 - alpha) (the side's mean of x), and the same for m2 with
// x^2. Then sd = sqrt(max(0, m2 - m1^2)). The seed is each statistic's mean and standard deviation over a reference's
// positions, the population's as the reference's. It is Blueprints' (game/blueprints/diversity.h).
struct DiversityNormalizer {
  static constexpr int N = Diversity::NUM_STATISTICS;
  double alpha;
  double m1[N] = {}, m2[N] = {};  // the running averages of x and of x^2
  double weight[N] = {};

  DiversityNormalizer(const double* seedMean, const double* seedSd, double alpha = Diversity::ALPHA) : alpha(alpha) {
    for(int i = 0; i < N; i++) {
      m1[i] = seedMean[i];
      m2[i] = seedSd[i] * seedSd[i] + seedMean[i] * seedMean[i];
      weight[i] = Diversity::weight(i);
    }
  }
  // One finished side: the statistics of its positions, one position after another (positions x N numbers). A side
  // without positions changes nothing.
  void update(const double* x, int positions) {
    if(positions <= 0)
      return;
    double sum[N] = {}, squares[N] = {};
    for(int k = 0; k < positions; k++)
      for(int i = 0; i < N; i++) {
        const double v = x[size_t(k) * N + i];
        sum[i] += v;
        squares[i] += v * v;
      }
    for(int i = 0; i < N; i++) {
      m1[i] = alpha * m1[i] + (1.0 - alpha) * (sum[i] / positions);
      m2[i] = alpha * m2[i] + (1.0 - alpha) * (squares[i] / positions);
    }
  }
  double mean(int i) const { return m1[i]; }
  double sd(int i) const { return std::sqrt(std::max(0.0, m2[i] - m1[i] * m1[i])); }
  // The normalized statistics: (x - mean) / max(sd, SIGMA_FLOOR), clipped to [-CLIP, CLIP], times the group's weight.
  void z(const double* x, double* out) const {
    for(int i = 0; i < N; i++) {
      const double v = (x[i] - m1[i]) / std::max(sd(i), Diversity::SIGMA_FLOOR);
      out[i] = std::max(-Diversity::CLIP, std::min(Diversity::CLIP, v)) * weight[i];
    }
  }
};

}  // namespace tak
