// The Enclosure notation: points, edges, moves and positions as text (docs/design/04-notation-and-protocol.md).
// The C++ twin of enclosure/notation.py; both are checked against tests/data/enclosure_notation.txt.
//
//   D10                       a point: column letter A-T without I (x 0-18), row 1-19 (y 0-18, from the bottom)
//   D10-E11                   an edge: the lower point first (by column, then by row); either order is read
//   pass                      a timeout pass (the rest of the turn forfeited)
//   A10-D10 Q10-T10 b1 0/0 0  a position: Blue's edges, Red's edges, the mover and the edges left in its turn,
//                             the scores (Blue's/Red's), the edge count. `*` marks a protected edge (the opponent's
//                             last turn), `+` a placed edge (the mover's edges of this turn), `-` an empty list.
//
// A position holds the rules' state only: its history planes come back empty. Header only, so the game layer, the
// tests and the WebAssembly bot need no build change. No exceptions (the WebAssembly build has none): the parsers
// return false (or -1) and say why.
#pragma once

#include "engine.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace enclosure {
namespace notation {

inline constexpr const char* LETTERS = "ABCDEFGHJKLMNOPQRST";  // as GTP: no I
inline constexpr const char* PLAYERS = "br";  // the first player (Blue on both sites), the second (Red)

inline std::string point_name(int x, int y) { return std::string(1, LETTERS[x]) + std::to_string(y + 1); }

inline bool parse_point(const std::string& text, int& x, int& y) {
  if (text.size() < 2 || text.size() > 3) return false;
  const char c = char(std::toupper((unsigned char)text[0]));
  const char* at = std::strchr(LETTERS, c);
  if (c == 0 || at == nullptr) return false;
  for (size_t i = 1; i < text.size(); ++i)
    if (!std::isdigit((unsigned char)text[i])) return false;
  x = int(at - LETTERS);
  y = std::atoi(text.c_str() + 1) - 1;
  return y >= 0 && y < SIZE;
}

inline std::string edge_name(const Edge& e) {  // Edges are kept canonical (make_edge): the lower point first.
  return point_name(e.ax, e.ay) + "-" + point_name(e.bx, e.by);
}

// "D10-E11" or "E11-D10" -> the canonical edge. Checks the shape of an edge (two different points at most 3 apart
// in each direction), not its legality in a position.
inline bool parse_edge(const std::string& text, Edge& e, std::string* why = nullptr) {
  const size_t dash = text.find('-');
  int ax, ay, bx, by;
  if (dash == std::string::npos || text.find('-', dash + 1) != std::string::npos ||
      !parse_point(text.substr(0, dash), ax, ay) || !parse_point(text.substr(dash + 1), bx, by)) {
    if (why) *why = "not an edge: " + text;
    return false;
  }
  if ((ax == bx && ay == by) || std::abs(ax - bx) > 3 || std::abs(ay - by) > 3) {
    if (why) *why = "not an edge: " + text + " (its ends must differ and be at most 3 apart)";
    return false;
  }
  e = make_edge(ax, ay, bx, by);
  return true;
}

inline std::string move_name(int action) {
  if (action == PASS) return "pass";
  const int cell = action / NOFF, k = action % NOFF;
  const int ax = cell % SIZE, ay = cell / SIZE;
  return edge_name(make_edge(ax, ay, ax + OFFS[k].dx, ay + OFFS[k].dy));
}

// A move as text -> the action for `p`, or -1. The engine lists an edge in one direction only, so both are tried.
// "pass" is a timeout pass (PASS, which Position::play plays as forfeit_turn).
inline int parse_move(const std::string& text, const Position& p, std::string* why = nullptr) {
  std::string t = text;
  std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if (t == "pass") return PASS;
  Edge e;
  if (!parse_edge(text, e, why)) return -1;
  const int ends[2][4] = {{e.ax, e.ay, e.bx, e.by}, {e.bx, e.by, e.ax, e.ay}};
  for (const auto& d : ends) {
    const int dx = d[2] - d[0], dy = d[3] - d[1];
    const int k = OFF_INDEX[dx + 3][dy + 3];
    if (k < 0) continue;
    const int action = (d[1] * SIZE + d[0]) * NOFF + k;
    if (p.is_legal(action)) return action;
  }
  if (why) *why = "illegal move in this position: " + text;
  return -1;
}

// A score as written: exact to the rules' rounding (1e-9), with no trailing zeros.
inline std::string score_text(double x) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9f", snap(x));
  std::string s(buf);
  while (!s.empty() && s.back() == '0') s.pop_back();
  if (!s.empty() && s.back() == '.') s.pop_back();
  return (s == "-0" || s.empty()) ? "0" : s;
}

inline std::string position_string(const Position& p) {
  std::string out;
  for (int side = 0; side < 2; ++side) {
    std::vector<Edge> es = p.edges[side];
    std::sort(es.begin(), es.end(), [](const Edge& a, const Edge& b) {
      return std::tie(a.ax, a.ay, a.bx, a.by) < std::tie(b.ax, b.ay, b.bx, b.by);
    });
    std::string field;
    for (const Edge& e : es) {
      if (!field.empty()) field += ",";
      field += edge_name(e);
      if (contains(p.protected_, e)) field += "*";
      else if (contains(p.placed, e)) field += "+";
    }
    out += (field.empty() ? "-" : field) + " ";
  }
  out += std::string(1, PLAYERS[p.player]) + std::to_string(p.moves_left) + " ";
  out += score_text(p.scores[0]) + "/" + score_text(p.scores[1]) + " " + std::to_string(p.move_count);
  return out;
}

// A position as text -> `p` (its history empty). Returns false and says why.
inline bool parse_position(const std::string& text, Position& p, std::string& why) {
  std::istringstream in(text);
  std::vector<std::string> fields;
  for (std::string f; in >> f;) fields.push_back(f);
  if (fields.size() != 5) {
    why = "a position has 5 fields, not " + std::to_string(fields.size());
    return false;
  }
  const std::string& turn = fields[2];
  if (turn.size() != 2 || (turn[0] != 'b' && turn[0] != 'r') || turn[1] < '0' || turn[1] > '2') {
    why = "the mover field is b or r and the edges left (0-2): " + turn;
    return false;
  }
  Position q;
  q.edges[0].clear(); q.edges[1].clear(); q.protected_.clear(); q.placed.clear();
  q.player = turn[0] == 'b' ? 0 : 1;
  q.moves_left = turn[1] - '0';
  for (int side = 0; side < 2; ++side) {
    if (fields[side] == "-") continue;
    std::stringstream list(fields[side]);
    for (std::string name; std::getline(list, name, ',');) {
      const char mark = name.empty() ? 0 : name.back();
      if (mark == '*' || mark == '+') name.pop_back();
      Edge e;
      if (!parse_edge(name, e, &why)) return false;
      if (contains(q.edges[0], e) || contains(q.edges[1], e)) {
        why = "edge " + edge_name(e) + " is listed twice";
        return false;
      }
      q.edges[side].push_back(e);
      if (mark == '*') {
        if (side == q.player) { why = edge_name(e) + "* is the mover's: protected edges are the opponent's"; return false; }
        q.protected_.push_back(e);
      }
      else if (mark == '+') {
        if (side != q.player) { why = edge_name(e) + "+ is the opponent's: placed edges are the mover's"; return false; }
        q.placed.push_back(e);
      }
    }
  }
  if (q.protected_.size() > 2 || q.placed.size() > 2) {
    why = "a turn has at most 2 edges: at most 2 protected and 2 placed";
    return false;
  }
  const std::string& scores = fields[3];
  const size_t slash = scores.find('/');
  char* end0 = nullptr;
  char* end1 = nullptr;
  if (slash == std::string::npos) { why = "the scores field is Blue's/Red's: " + scores; return false; }
  const std::string s0 = scores.substr(0, slash), s1 = scores.substr(slash + 1);
  const double v0 = std::strtod(s0.c_str(), &end0), v1 = std::strtod(s1.c_str(), &end1);
  if (s0.empty() || s1.empty() || *end0 != 0 || *end1 != 0) { why = "the scores field is Blue's/Red's: " + scores; return false; }
  q.scores[0] = snap(v0);
  q.scores[1] = snap(v1);
  const std::string& count = fields[4];
  if (count.empty() || count.size() > 3 || !std::all_of(count.begin(), count.end(), ::isdigit) ||
      std::atoi(count.c_str()) > MAX_MOVES) {
    why = "the edge count is 0-" + std::to_string(MAX_MOVES) + ": " + count;
    return false;
  }
  q.move_count = std::atoi(count.c_str());
  q.area_ok[0] = q.area_ok[1] = false;
  std::fill(q.history_n, q.history_n + HISTORY_TURNS, 0);
  p = q;
  return true;
}

}  // namespace notation
}  // namespace enclosure
