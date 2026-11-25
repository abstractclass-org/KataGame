#pragma once
// The road groups of a position of Tak: the sets of one player's road pieces (its flats and capstones on top) that are
// joined by orthogonal steps, the edges that each touches, and the empty squares where a flat completes the player's
// road. AZdb's statistics (diversity.h), the network's inputs and the target of the winning road (gameencoding.h) read
// them.

#include <algorithm>
#include <cstdint>

#include "gamerules.h"

namespace tak {

namespace roads_detail {
// Column x's squares, at compile time.
constexpr uint64_t column_bits(int x) {
  uint64_t m = 0;
  for(int y = 0; y < MAX_LEN; y++)
    m |= uint64_t(1) << (y * MAX_LEN + x);
  return m;
}
constexpr uint64_t WEST = column_bits(0);
constexpr uint64_t NOT_WEST = ~WEST, NOT_EAST = ~column_bits(MAX_LEN - 1);
}  // namespace roads_detail

// The squares that touch a square of m orthogonally.
inline uint64_t neighbours(uint64_t m) {
  return ((m & roads_detail::NOT_EAST) << 1) | ((m & roads_detail::NOT_WEST) >> 1) | (m << MAX_LEN) | (m >> MAX_LEN);
}
// The flats and capstones on top, the pieces of a road.
inline uint64_t road_pieces(const Position& p, int who) { return p.owner[who] & ~p.walls; }

// The roads of a player's road pieces.
struct Roads {
  int pieces = 0;   // the road pieces
  int largest = 0;  // those of the largest group
  int extent = 0;   // the largest extent of one group, the larger of its rows and its columns
  bool complete = false;
  // The squares of the groups that touch each edge, by the edge's direction as the moves number it: + the top rank's
  // edge, - the first rank's, > the last file's, < the first file's.
  uint64_t edge[4] = {};
  // The empty squares where a flat completes the player's road, whatever its reserve: none when it is complete.
  uint64_t completing = 0;

  // The squares of the groups that join opposite edges, the player's complete roads.
  uint64_t winning() const { return (edge[0] & edge[1]) | (edge[2] & edge[3]); }
};

// Whether the edges (bit d for direction d) join opposite sides.
inline bool edges_join(int edges) { return (edges & 3) == 3 || (edges & 12) == 12; }

inline Roads roads(const Position& p, int who) {
  const int n = p.size;
  const uint64_t road = road_pieces(p, who);
  const uint64_t row = (uint64_t(1) << n) - 1;
  const uint64_t side[4] = {row << ((n - 1) * MAX_LEN), row, roads_detail::WEST << (n - 1), roads_detail::WEST};
  Roads r;
  r.pieces = popcount(road);
  int8_t label[NUM_POINTS];
  uint8_t edges[NUM_POINTS];
  int groups = 0;
  uint64_t left = road;
  while(left) {
    uint64_t group = left & (~left + 1), last = 0;
    while(group != last) {
      last = group;
      group |= neighbours(group) & road;
    }
    left &= ~group;
    int f = 0;
    for(int d = 0; d < 4; d++)
      if(group & side[d]) {
        f |= 1 << d;
        r.edge[d] |= group;
      }
    edges[groups] = uint8_t(f);
    r.complete = r.complete || edges_join(f);
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
  // A complete road cannot be completed (it is the result). Else an empty square joins the groups that touch it and
  // the edges that it is on. A square beside no road piece is on no two opposite edges (N >= 3): it completes none.
  if(r.complete)
    return r;
  for(uint64_t empty = board_mask(n) & ~occupied(p) & neighbours(road); empty; empty &= empty - 1) {
    const int e = __builtin_ctzll(empty);
    int f = 0;
    for(int d = 0; d < 4; d++)
      if(side[d] & bit(e))
        f |= 1 << d;
    for(uint64_t nb = neighbours(bit(e)) & road; nb; nb &= nb - 1)
      f |= edges[label[__builtin_ctzll(nb)]];
    if(edges_join(f))
      r.completing |= bit(e);
  }
  return r;
}

}  // namespace tak
