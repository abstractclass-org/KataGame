// Enclosure fork: whether a position could occur in a game of the site's rules (learn-enclosure's request 16;
// docs/design/04-notation-and-protocol.md, "The position check"). The same checks as the Python reference,
// games/enclosure/position_check.py, in the same order and with the same words; its docstring gives why each holds in every
// game. Only what every game makes certain: a board that a game reaches in a rare way passes.
#pragma once

#include "engine.h"
#include "notation.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

namespace enclosure {
namespace position_check {

struct Reason {
  std::string code;
  std::string text;
  std::vector<std::string> edges;  // the edges it names, in the notation
};

inline const char* NAME[2] = {"Blue", "Red"};
inline constexpr int BOARD_AREA = (SIZE - 1) * (SIZE - 1);  // 324: no turn end adds more to a score
inline constexpr double TOLERANCE = 1e-6;  // scores and areas are exact to the rules' rounding (1e-9)

// The player of edge number m (from 0): Blue's first turn has one edge, then turns of two.
inline int turn_of_move(int m) { return m == 0 ? 0 : ((m + 1) / 2) % 2; }

// The mover and its edges left after n edges.
inline void expected_turn(int n, int& player, int& left) {
  if (n == 0) { player = 0; left = 1; return; }
  if (n >= MAX_MOVES) { player = 1; left = 0; return; }  // after Blue's last turn, of one edge
  const int k = (n + 1) / 2;
  player = k % 2;
  left = std::min(2 * k + 1, MAX_MOVES) - n;
}

inline int turns_ended(int n) { return n == 0 ? 0 : (n + 1) / 2 + (n >= MAX_MOVES ? 1 : 0); }

// The edges of the turn before the current one: 1 for Blue's first and last turns, else 2.
inline int previous_turn_size(int n) {
  if (n == 0) return 0;
  const int k = n >= MAX_MOVES ? 61 : (n + 1) / 2;
  return (k - 1 == 0 || k - 1 == 60) ? 1 : 2;
}

inline std::string plural(int k, const std::string& word) { return std::to_string(k) + " " + word + (k == 1 ? "" : "s"); }

inline bool edge_less(const Edge& a, const Edge& b) {
  return std::tie(a.ax, a.ay, a.bx, a.by) < std::tie(b.ax, b.ay, b.bx, b.by);
}

inline bool shares_end(const Edge& e, const Edge& f) {
  return (e.ax == f.ax && e.ay == f.ay) || (e.ax == f.bx && e.ay == f.by) || (e.bx == f.ax && e.by == f.ay) ||
         (e.bx == f.bx && e.by == f.by);
}

inline std::vector<Reason> check(const Position& p) {
  using notation::edge_name;
  using notation::point_name;
  using notation::score_text;
  std::vector<Reason> reasons;
  auto reason = [&](const std::string& code, const std::string& text, const std::vector<Edge>& edges) {
    Reason r{code, text, {}};
    for (const Edge& e : edges) r.edges.push_back(edge_name(e));
    reasons.push_back(r);
  };
  const int n = p.move_count;
  std::vector<Edge> edges[2] = {p.edges[0], p.edges[1]};
  for (auto& es : edges) std::sort(es.begin(), es.end(), edge_less);
  const int mover = p.player, opponent = 1 - p.player;

  int wantPlayer = 0, wantLeft = 0;
  expected_turn(n, wantPlayer, wantLeft);
  if (p.player != wantPlayer || p.moves_left != wantLeft)
    reason("turn", "After " + plural(n, "edge") + ", " + NAME[wantPlayer] + " moves with " + plural(wantLeft, "edge") +
                   " left, not " + NAME[p.player] + " with " + std::to_string(p.moves_left) + ".", {});

  if (n == 0) {
    const Edge start[2] = {make_edge(0, 9, 3, 9), make_edge(15, 9, 18, 9)};
    for (int side = 0; side < 2; ++side)
      if (edges[side].size() != 1 || !(edges[side][0] == start[side]))
        reason("start", std::string("Before the first edge, ") + NAME[side] + " has only its start edge, " +
                        edge_name(start[side]) + ".", edges[side]);
  }

  for (int side = 0; side < 2; ++side) {
    int most = 0;
    for (int m = 0; m < n; ++m) most += turn_of_move(m) == side ? 1 : 0;
    if ((int)edges[side].size() > 1 + most)
      reason("edge-count", std::string(NAME[side]) + " has " + plural((int)edges[side].size(), "edge") + ", but in " +
                           plural(n, "edge") + " " + NAME[side] + " can have placed at most " + std::to_string(most) +
                           ", and starts with 1.", {});
  }

  for (const Edge& e : edges[0])
    for (const Edge& f : edges[1])
      if (intersects(e.ax, e.ay, e.bx, e.by, f.ax, f.ay, f.bx, f.by)) {
        const long long d1 = cross(e.ax, e.ay, e.bx, e.by, f.ax, f.ay), d2 = cross(e.ax, e.ay, e.bx, e.by, f.bx, f.by);
        const long long d3 = cross(f.ax, f.ay, f.bx, f.by, e.ax, e.ay), d4 = cross(f.ax, f.ay, f.bx, f.by, e.bx, e.by);
        const bool proper = d1 * d2 < 0 && d3 * d4 < 0;
        reason("cross", "Blue's edge " + edge_name(e) + " and Red's edge " + edge_name(f) + " " +
                        (proper ? "cross" : "touch") + ": a move that does so removes the other edge.", {e, f});
      }

  for (int side = 0; side < 2; ++side) {
    std::vector<std::pair<int, int>> nodes;
    for (const Edge& e : edges[side]) {
      nodes.push_back({e.ax, e.ay});
      nodes.push_back({e.bx, e.by});
    }
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    for (const Edge& e : edges[side])
      for (const auto& q : nodes) {
        const bool end = (q.first == e.ax && q.second == e.ay) || (q.first == e.bx && q.second == e.by);
        if (!end && on_segment(q.first, q.second, e.ax, e.ay, e.bx, e.by))
          reason("node-inside", std::string(NAME[side]) + "'s node " + point_name(q.first, q.second) + " lies inside " +
                                NAME[side] + "'s edge " + edge_name(e) + ".", {e});
      }
  }

  const int wantPlaced = (n >= 2 && n < MAX_MOVES && n % 2 == 0) ? 1 : 0;
  std::vector<Edge> placed = p.placed;
  std::sort(placed.begin(), placed.end(), edge_less);
  if ((int)placed.size() != wantPlaced)
    reason("placed", "After " + plural(n, "edge") + ", the mover has placed " + plural(wantPlaced, "edge") +
                     " this turn, not " + std::to_string(placed.size()) + ".", placed);
  else
    for (const Edge& e : placed) {
      bool joined = false;
      for (const Edge& f : edges[mover])
        if (!contains(p.placed, f) && shares_end(e, f)) joined = true;
      if (!joined)
        reason("placed-joined", std::string(NAME[mover]) + "'s placed edge " + edge_name(e) + " joins no other " +
                                NAME[mover] + " edge: an edge starts at one of its player's nodes.", {e});
    }

  std::vector<Edge> prot = p.protected_;
  std::sort(prot.begin(), prot.end(), edge_less);
  const int mostProtected = previous_turn_size(n);
  if ((int)prot.size() > mostProtected)
    reason("protected", "After " + plural(n, "edge") + ", " + NAME[opponent] + "'s last turn placed at most " +
                        plural(mostProtected, "edge") + ", not " + std::to_string(prot.size()) + " protected.", prot);
  else if (!prot.empty() && (n % 2 == 1 || n >= MAX_MOVES)) {
    // No edge of the opponent's has been cut since its last turn: each of that turn's edges started at a node of an
    // older opponent edge, or of the turn's other edge.
    std::vector<bool> joined(prot.size(), false);
    for (size_t i = 0; i < prot.size(); ++i)
      for (const Edge& f : edges[opponent])
        if (!contains(p.protected_, f) && shares_end(prot[i], f)) joined[i] = true;
    if (prot.size() == 2 && joined[0] != joined[1] && shares_end(prot[0], prot[1])) joined[0] = joined[1] = true;
    for (size_t i = 0; i < prot.size(); ++i)
      if (!joined[i])
        reason("protected-joined", std::string(NAME[opponent]) + "'s protected edge " + edge_name(prot[i]) +
                                   " joins no other " + NAME[opponent] + " edge: an edge starts at one of its player's nodes.",
               {prot[i]});
  }

  const int ended = turns_ended(n);
  for (int side = 0; side < 2; ++side) {
    const double s = p.scores[side];
    if (s < -TOLERANCE)
      reason("score-negative", std::string(NAME[side]) + "'s score " + score_text(s) + " is less than 0.", {});
    else if (s > double(BOARD_AREA) * ended + TOLERANCE)
      reason("score-high", std::string(NAME[side]) + "'s score " + score_text(s) + " is more than " +
                           std::to_string(BOARD_AREA * ended) + ": " + plural(ended, "turn end") + " of at most " +
                           std::to_string(BOARD_AREA) + " each.", {});
    else if (ended > 0) {
      // The edges at the last turn end: the mover's without its placed edges; the opponent's, of which the mover's
      // placed edge may have cut one since, so its area then was at least its area now.
      std::vector<Edge> then;
      for (const Edge& f : edges[side])
        if (!(side == mover && contains(p.placed, f))) then.push_back(f);
      const double area = enclosed_area(then);
      if (s < area - TOLERANCE)
        reason("score-low", std::string(NAME[side]) + "'s score " + score_text(s) +
                            " is less than the area that its edges enclosed at the last turn end, " + score_text(area) + ".",
               {});
    }
  }
  return reasons;
}

}  // namespace position_check
}  // namespace enclosure
