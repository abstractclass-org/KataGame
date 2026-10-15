#pragma once
// Extracted from the Enclosure repository's native engine (enclosure_native.cpp at a546e1a) and edited here since
// (V2's input options, the income areas, the CPU work of 2026-10-06). This is the only copy of Enclosure's rules:
// Python reaches it through the Enclosure repository's binding, games/enclosure/binding/rules.cpp
// (katago/ENCLOSURE.md, "Board and Rules").

#include <cstdlib>

#include "engine.h"

namespace enclosure {

// --- Feature planes (games/enclosure/planes.py) ----------------------------------
//
// Wall and territory structure per player, for the network's input and the
// heuristic bots. Python references: heuristic.hops, planes.solid_edges,
// heuristic.held_ground, planes.enclosed_points and planes.feature_planes;
// games/enclosure/tests/test_planes.py checks that both agree.

constexpr int UNREACHED = 99;
constexpr int PER_PLAYER = 9;
constexpr int FEATURE_PLANES = 2 * PER_PLAYER + HISTORY_TURNS;  // then the history planes
constexpr int N_POINTS = SIZE * SIZE;

// Crossing counts of every candidate segment with the given edges (engine.enemy_crossings).
inline void crossing_counts(const std::vector<Edge>& edges, std::vector<int16_t>& hits) {
  hits.assign(size_t(GRID) * GRID * NOFF, 0);
  for (const Edge& e : edges) {
    int oe = OFF_INDEX[e.bx - e.ax + 3][e.by - e.ay + 3];
    int base = ((e.ax + PAD) * GRID + e.ay + PAD) * NOFF;
    for (int rel : CROSSINGS[oe]) ++hits[base + rel];
  }
}

// Enclosure fork (owner, 2026-10-06; CPU trick 4): each point's on-board offsets, ascending, with the far end, built
// once: the hop search walks these instead of testing all NOFF offsets against the board's edges at every step.
struct HopNeighbours {
  int count[N_POINTS];
  int16_t k[N_POINTS][NOFF];
  int16_t b[N_POINTS][NOFF];
  HopNeighbours() {
    for (int q = 0; q < N_POINTS; ++q) {
      const int x = q / SIZE, y = q % SIZE;
      count[q] = 0;
      for (int o = 0; o < NOFF; ++o) {
        const int bx = x + OFFS[o].dx, by = y + OFFS[o].dy;
        if (bx < 0 || bx >= SIZE || by < 0 || by >= SIZE) continue;
        k[q][count[q]] = int16_t(o);
        b[q][count[q]] = int16_t(bx * SIZE + by);
        ++count[q];
      }
    }
  }
};
inline const HopNeighbours hop_neighbours;

// Hop distances from `player`'s nodes (heuristic.hops): a hop is a segment of
// Chebyshev length <= 3 crossing at most one enemy edge. dist[x * SIZE + y].
// With turn_reach (owner, 2026-10-03, the turn-safe plane): also the points `player` can start a cut from in its coming
// turn of turn_edges edges, as a by-product of the first layer: its nodes, and with two edges the points one hop
// reaches without crossing a guarded edge (guard_hits nonzero). See solid_flags and coming_turn.
inline void hop_distances(const Position& p, int player, int* dist, const std::vector<int16_t>* enemy_hits = nullptr,
                          uint8_t* turn_reach = nullptr, int turn_edges = 0, const std::vector<uint8_t>* guard_hits = nullptr) {
  std::vector<int16_t> own_counts;
  if (!enemy_hits) { crossing_counts(p.edges[1 - player], own_counts); enemy_hits = &own_counts; }
  const std::vector<int16_t>& hits = *enemy_hits;
  std::fill(dist, dist + N_POINTS, UNREACHED);
  if (turn_reach) std::fill(turn_reach, turn_reach + N_POINTS, 0);
  std::vector<int> frontier, next;
  for (const Edge& e : p.edges[player])
    for (int q : {e.ax * SIZE + e.ay, e.bx * SIZE + e.by})
      if (dist[q]) { dist[q] = 0; frontier.push_back(q); }
  if (turn_reach && turn_edges >= 1)
    for (int q : frontier) turn_reach[q] = 1;
  for (int d = 1; !frontier.empty(); ++d) {
    next.clear();
    for (int q : frontier) {
      const int x = q / SIZE, y = q % SIZE;
      const int base = ((x + PAD) * GRID + y + PAD) * NOFF;
      const int nn = hop_neighbours.count[q];
      for (int j = 0; j < nn; ++j) {
        const int idx = base + hop_neighbours.k[q][j];
        if (hits[idx] > 1) continue;
        const int b = hop_neighbours.b[q][j];
        if (d == 1 && turn_reach && turn_edges >= 2 && !(guard_hits && (*guard_hits)[idx])) turn_reach[b] = 1;
        if (dist[b] == UNREACHED) { dist[b] = d; next.push_back(b); }
      }
    }
    frontier.swap(next);
  }
}

// Per edge of `player` (in p.edges order): can the opponent not cut it?
// (planes.solid_edges) No on-board segment with an end the opponent reaches
// crosses that edge and no other edge of the player.
// With turn_safe (the turn-safe plane): also, per edge, whether the opponent cannot cut it in its coming turn: guarded
// (turn_guarded) or no cutting segment with an end in opp_turn_reach. That reach lies within the full reach, so one
// scan of the edge's cutters answers both questions; it stops once both are settled.
inline void solid_flags(const Position& p, int player, const int* opp_dist, std::vector<uint8_t>& solid,
                 const std::vector<int16_t>* own_hits = nullptr, const uint8_t* opp_turn_reach = nullptr,
                 const std::vector<Edge>* turn_guarded = nullptr, std::vector<uint8_t>* turn_safe = nullptr) {
  const auto& own = p.edges[player];
  std::vector<int16_t> own_counts;
  if (!own_hits) { crossing_counts(own, own_counts); own_hits = &own_counts; }
  const std::vector<int16_t>& hits = *own_hits;
  solid.assign(own.size(), 1);
  if (turn_safe) turn_safe->assign(own.size(), 1);
  for (size_t i = 0; i < own.size(); ++i) {
    const Edge& e = own[i];
    const bool settled_safe = !turn_safe || (turn_guarded && contains(*turn_guarded, e));
    int oe = OFF_INDEX[e.bx - e.ax + 3][e.by - e.ay + 3];
    int base = ((e.ax + PAD) * GRID + e.ay + PAD) * NOFF;
    for (int rel : CROSSINGS[oe]) {
      int idx = base + rel;
      if (hits[idx] != 1) continue;
      int cell = idx / NOFF, k = idx % NOFF;
      int sx = cell / GRID - PAD, sy = cell % GRID - PAD;
      int ex = sx + OFFS[k].dx, ey = sy + OFFS[k].dy;
      if (sx < 0 || sx >= SIZE || sy < 0 || sy >= SIZE || ex < 0 || ex >= SIZE || ey < 0 || ey >= SIZE)
        continue;
      if (opp_dist[sx * SIZE + sy] == UNREACHED && opp_dist[ex * SIZE + ey] == UNREACHED) continue;
      solid[i] = 0;
      if (!settled_safe && (opp_turn_reach[sx * SIZE + sy] || opp_turn_reach[ex * SIZE + ey])) (*turn_safe)[i] = 0;
      if (settled_safe || !(*turn_safe)[i]) break;
    }
  }
}

// The turn-safe plane's turn (owner, 2026-10-03; planes.turn_safe_edges is the Python reference): the opponent's coming
// turn, against `player`'s edges. It is the current turn when the opponent is to move, else its next, after `player`
// finishes this one (as many edges as remain, at most 2). Returns its edge count; `guarded` gets `player`'s edges the
// opponent may not cut in it: its protected edges, or, when it is to move, its edges placed this turn.
inline int coming_turn(const Position& p, int player, std::vector<Edge>& guarded) {
  guarded.clear();
  const std::vector<Edge>& marked = p.player == player ? p.placed : p.protected_;
  for (const Edge& e : marked)
    if (contains(p.edges[player], e)) guarded.push_back(e);
  return p.player == player ? std::min(2, MAX_MOVES - p.move_count - p.moves_left) : p.moves_left;
}

// Sets to `value` the entries of the candidate segments crossing any of `edges`, in a crossing-count-shaped array.
inline void mark_crossings(const std::vector<Edge>& edges, std::vector<uint8_t>& marks, uint8_t value) {
  for (const Edge& e : edges) {
    int oe = OFF_INDEX[e.bx - e.ax + 3][e.by - e.ay + 3];
    int base = ((e.ax + PAD) * GRID + e.ay + PAD) * NOFF;
    for (int rel : CROSSINGS[oe]) marks[base + rel] = value;
  }
}

// Feature planes (planes.feature_planes) as 0/1 bytes, channels last:
// out[(y * SIZE + x) * FEATURE_PLANES + f].
// Each player's plane 7 holds the points of its turn-safe edges: those the opponent cannot cut in its coming turn
// (coming_turn, planes.turn_safe_edges). Protection lasts one turn, which the steady-state planes (Solid, the hops,
// Unreached) ignore; this plane states it for that turn only. Plane 8 stays zero: it held the openings of held ground
// (a max flow, about 15% of self-play's CPU), which V2 dropped (games/enclosure/docs/decisions/inputs-and-targets.md). These were
// the input options ENCLOSURE_TURN_SAFE and ENCLOSURE_NO_HELD_GROUND until 2026-10-08, when they became the only way.
inline void feature_planes(const Position& p, uint8_t* out) {
  // Each player's crossing counts, computed once and shared by the hops and solidity.
  static thread_local std::vector<int16_t> counts[2];
  crossing_counts(p.edges[0], counts[0]);
  crossing_counts(p.edges[1], counts[1]);
  int dist[2][N_POINTS];
  // guarded[k] are k's edges the opponent may not cut in its coming turn (turn_edges[1 - k] edges); turn_reach[k]
  // the points k can start a cut from in its coming turn.
  static thread_local std::vector<uint8_t> guard_marks(size_t(GRID) * GRID * NOFF, 0);
  std::vector<Edge> guarded[2];
  int turn_edges[2] = {0, 0};
  uint8_t turn_reach[2][N_POINTS];
  for (int k = 0; k < 2; ++k) turn_edges[1 - k] = coming_turn(p, k, guarded[k]);
  for (int k = 0; k < 2; ++k) {
    mark_crossings(guarded[1 - k], guard_marks, 1);
    hop_distances(p, k, dist[k], &counts[1 - k], turn_reach[k], turn_edges[k], &guard_marks);
    mark_crossings(guarded[1 - k], guard_marks, 0);
  }
  std::fill(out, out + N_POINTS * FEATURE_PLANES, 0);
  auto set = [&](int q, int f) { out[((q % SIZE) * SIZE + q / SIZE) * FEATURE_PLANES + f] = 1; };
  uint8_t mask[N_POINTS];
  std::vector<uint8_t> solid;
  for (int k = 0; k < 2; ++k) {
    const int base = k * PER_PLAYER;
    const int* d = dist[k];
    for (int q = 0; q < N_POINTS; ++q) {
      if (d[q] == 0) set(q, base + 0);
      if (d[q] >= 1 && d[q] <= 3) set(q, base + 2 + d[q]);
      if (d[q] == UNREACHED) set(q, base + 6);
    }
    enclosed_area(p.edges[k], mask);
    for (int q = 0; q < N_POINTS; ++q)
      if (mask[q]) set(q, base + 1);
    std::vector<uint8_t> turn_safe;
    solid_flags(p, k, dist[1 - k], solid, &counts[k], turn_reach[1 - k], &guarded[k], &turn_safe);
    std::vector<Edge> solid_edges;
    for (size_t i = 0; i < solid.size(); ++i)
      if (solid[i]) solid_edges.push_back(p.edges[k][i]);
    edge_points(solid_edges, mask);
    for (int q = 0; q < N_POINTS; ++q)
      if (mask[q]) set(q, base + 2);
    std::vector<Edge> safe_edges;
    for (size_t i = 0; i < turn_safe.size(); ++i)
      if (turn_safe[i]) safe_edges.push_back(p.edges[k][i]);
    edge_points(safe_edges, mask);
    for (int q = 0; q < N_POINTS; ++q)
      if (mask[q]) set(q, base + 7);
  }
  for (int h = 0; h < HISTORY_TURNS; ++h) {
    if (!p.history_n[h]) continue;
    edge_points(p.history_turn(h), mask);
    for (int q = 0; q < N_POINTS; ++q)
      if (mask[q]) set(q, 2 * PER_PLAYER + h);
  }
}

// --- Mirror images ----------------------------------------------------------
//
// The rules are symmetric under y -> SIZE - 1 - y (so is the start position):
// a position's mirror image is as good for the same player.

inline Position mirror_position(const Position& p) {
  Position m = p;
  auto flip = [](std::vector<Edge>& v) {
    for (Edge& e : v) e = make_edge(e.ax, SIZE - 1 - e.ay, e.bx, SIZE - 1 - e.by);
  };
  flip(m.edges[0]);
  flip(m.edges[1]);
  flip(m.protected_);
  flip(m.placed);
  for (int h = 0; h < HISTORY_TURNS; ++h)
    for (int i = 0; i < m.history_n[h]; ++i) {
      const Edge& e = m.history[h][i];
      m.history[h][i] = make_edge(e.ax, SIZE - 1 - e.ay, e.bx, SIZE - 1 - e.by);
    }
  return m;  // Areas are unchanged by the flip, so the cached ones stay valid.
}

// The mirror of `action`, as listed in the mirrored position's sorted `legal`
// list (-1 if absent). Between two own nodes only one direction is listed,
// and the flip can change which, so the reverse direction is tried too.
inline int mirror_action(int action, const std::vector<int>& legal) {
  if (action == PASS) return std::binary_search(legal.begin(), legal.end(), PASS) ? PASS : -1;
  const int cell = action / NOFF, k = action % NOFF;
  const int ax = cell % SIZE, ay = cell / SIZE;
  const int bx = ax + OFFS[k].dx, by = ay + OFFS[k].dy;
  const int m_ax = ax, m_ay = SIZE - 1 - ay, m_bx = bx, m_by = SIZE - 1 - by;
  const int forward = (m_ay * SIZE + m_ax) * NOFF + OFF_INDEX[m_bx - m_ax + 3][m_by - m_ay + 3];
  if (std::binary_search(legal.begin(), legal.end(), forward)) return forward;
  const int reverse = (m_by * SIZE + m_bx) * NOFF + OFF_INDEX[m_ax - m_bx + 3][m_ay - m_by + 3];
  if (std::binary_search(legal.begin(), legal.end(), reverse)) return reverse;
  return -1;
}

}  // namespace enclosure
