// Blueprints' position check (the engine repository's games/blueprints/README.md, "The position check"; the sites
// repository's request 28): whether a game of the site's rules can reach a position, and if not, each reason in words
// that a player reads. Only what every game makes certain: a position that a game reaches only in a rare way passes.
// The check reads what the site's notation holds: the cells, both hands, the scores and the moves played. It does not
// read the engine's own fields (the offset, the recent placements' boxes, the view's orientation), which set the view
// and not the game. Each condition holds in every game for the reason that its text gives:
// - start: with no move played, the position is the standard start (setup).
// - start-wall: a capture does not remove a wall, and a placement puts cells only on empty hexes, so each wall of the
//   start stays on its hex.
// - touch: a placement captures each cell of the opponent that touches it, except walls, so no cell of Blue that is
//   not a wall touches a cell of Red that is not a wall (the start has none).
// - cell-count, wall-count: a player's cells that are not its cells of the start came from its placements. A placement
//   puts one blueprint: at most 7 cells (the start's blueprints) and 1 more for each edit before it, at most 21 (the
//   hand), with at most the hand's 3 walls.
// - hand: an edit moves a cell, and a placement keeps the hand, so a hand holds the start's kinds of cells.
// - hand-edits: an edit adds at most 1 cell to a blueprint, so a hand needs as many edits as its blueprints hold cells
//   more than at the start, and a hand that is not the start's needs one.
// - score-negative, score-unmoved: a move scores 0 or more points, and only a placement scores.
// The reasons come in this order, Blue's before Red's. With no move played, the start's reason is the only one.
#pragma once

#include "gamerules.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace blueprints {
namespace position_check {

struct Reason {
  std::string code;
  std::string text;
  std::vector<std::string> hexes;  // the hexes that it names, "q,r" in the site's coordinates, in reading order
};

inline const char* NAME[2] = {"Blue", "Red"};

// A hex of the site's grid as (r, q), so that the order of the pairs is the reading order: the least r, then the least
// q.
using Hex = std::pair<int, int>;

inline std::string hex_text(const Hex& h) { return cell_text(h.second, h.first); }

inline std::string plural(int k, const std::string& word, const std::string& words) {
  return std::to_string(k) + " " + (k == 1 ? word : words);
}

// The texts of some hexes: "a", "a and b", "a, b and c".
inline std::string hexes_text(const std::vector<Hex>& hexes) {
  std::string s;
  for(size_t i = 0; i < hexes.size(); i++)
    s += (i == 0 ? "" : i + 1 == hexes.size() ? " and " : ", ") + hex_text(hexes[i]);
  return s;
}

// The standard start (setup), made once.
inline const Position& standard() {
  static const Position* s = [] {
    Position* p = new Position();
    setup(*p);
    return p;
  }();
  return *s;
}

// Every cell of p by its hex in the site's coordinates: its cell_value.
inline std::map<Hex, int8_t> site_cells(const Position& p) {
  std::map<Hex, int8_t> out;
  const Box& used = p.cell.box();
  if(used.any)
    for(int y = used.y0; y <= used.y1; y++)
      for(int x = used.x0; x <= used.x1; x++)
        if(const int8_t v = p.cell[y * RULES_LEN + x])
          out[{y - RULES_RADIUS + p.offsetR, x - RULES_RADIUS + p.offsetQ}] = v;
  return out;
}

// A player's three blueprints, with a pick that waits for its drop put back: the rest of the picked blueprint keeps the
// whole blueprint's frame until the drop.
inline void hand_with_pick(const Position& p, int player, Blueprint (&out)[HAND]) {
  std::copy(p.hand[player], p.hand[player] + HAND, out);
  if(p.picked >= 0 && player == turn_of(p.ply))
    out[p.picked].add(p.pickQ, p.pickR, p.pickType);
}

inline bool same_blueprint(const Blueprint& a, const Blueprint& b) {
  const Blueprint x = sorted_cells(a), y = sorted_cells(b);
  if(x.n != y.n)
    return false;
  for(int i = 0; i < x.n; i++)
    if(x.q[i] != y.q[i] || x.r[i] != y.r[i] || x.t[i] != y.t[i])
      return false;
  return true;
}

// The cells of each kind (PLAIN, STAR, WALL) in a hand.
inline void kinds(const Blueprint (&hand)[HAND], int (&out)[3]) {
  out[0] = out[1] = out[2] = 0;
  for(const Blueprint& b : hand)
    for(int i = 0; i < b.n; i++)
      out[b.t[i]]++;
}

// The moves that a player made before the ply: Blue's first move, then turns of 2 (turn_of). A pass is a move.
inline int moves_made(int ply, int player) {
  int m = 0;
  for(int i = 0; i < ply; i++)
    m += turn_of(i) == player;
  return m;
}

// The most cells of the start's blueprints: 7.
inline int start_blueprint_cells() {
  int most = 0;
  for(const Blueprint& b : standard().hand[0])
    most = std::max(most, int(b.n));
  return most;
}

// The most cells that m moves of a player can place: with e of them edits, at most m - e placements of at most
// min(21, 7 + e) cells each.
inline int most_cells_placed(int m) {
  int most = 0;
  for(int e = 0; e <= m; e++)
    most = std::max(most, (m - e) * std::min(MAX_CELLS, start_blueprint_cells() + e));
  return most;
}

inline std::vector<Reason> check(const Position& p) {
  std::vector<Reason> reasons;
  auto reason = [&](const char* code, const std::string& text, std::vector<Hex> hexes) {
    std::sort(hexes.begin(), hexes.end());
    hexes.erase(std::unique(hexes.begin(), hexes.end()), hexes.end());
    Reason r{code, text, {}};
    for(const Hex& h : hexes)
      r.hexes.push_back(hex_text(h));
    reasons.push_back(r);
  };
  const Position& s = standard();
  const std::map<Hex, int8_t> cells = site_cells(p), start = site_cells(s);
  Blueprint hands[2][HAND];
  for(int player = 0; player < 2; player++)
    hand_with_pick(p, player, hands[player]);
  auto start_hand = [&](int player) {
    for(int k = 0; k < HAND; k++)
      if(!same_blueprint(hands[player][k], s.hand[player][k]))
        return false;
    return true;
  };
  // The cell of the start on a hex, or 0.
  auto start_cell = [&](const Hex& h) {
    const auto it = start.find(h);
    return it == start.end() ? int8_t(0) : it->second;
  };

  if(p.ply == 0) {
    std::vector<Hex> differ;
    for(const auto& [h, v] : cells)
      if(start_cell(h) != v)
        differ.push_back(h);
    for(const auto& [h, v] : start)
      if(!cells.count(h))
        differ.push_back(h);
    if(!differ.empty() || !start_hand(0) || !start_hand(1) || p.scores[0] != 0 || p.scores[1] != 0)
      reason("start",
             "No move has been played, so the position is the standard start: its cells, both hands and the scores "
             "0/0.",
             differ);
    return reasons;
  }

  int moves[2];
  for(int player = 0; player < 2; player++)
    moves[player] = moves_made(p.ply, player);

  // start-wall
  for(int player = 0; player < 2; player++) {
    std::vector<Hex> missing;
    for(const auto& [h, v] : start)
      if(cell_owner(v) == player && cell_type(v) == WALL) {
        const auto it = cells.find(h);
        if(it == cells.end() || it->second != v)
          missing.push_back(h);
      }
    if(missing.empty())
      continue;
    const bool one = missing.size() == 1;
    reason("start-wall",
           std::string(NAME[player]) + "'s " +
             (one ? "wall of the start is not on its hex " : "walls of the start are not on their hexes ") +
             hexes_text(missing) + ": a capture does not remove a wall, and a placement puts cells only on empty hexes.",
           missing);
  }

  // touch
  {
    std::vector<std::pair<Hex, Hex>> pairs;
    for(const auto& [h, v] : cells) {
      if(cell_owner(v) != 0 || cell_type(v) == WALL)
        continue;
      for(const auto& d : DIRS) {
        const Hex n{h.first + d[1], h.second + d[0]};
        const auto it = cells.find(n);
        if(it != cells.end() && cell_owner(it->second) == 1 && cell_type(it->second) != WALL)
          pairs.push_back({h, n});
      }
    }
    if(!pairs.empty()) {
      std::vector<Hex> hexes;
      for(const auto& [b, r] : pairs) {
        hexes.push_back(b);
        hexes.push_back(r);
      }
      const std::string first = "Blue's cell on " + hex_text(pairs[0].first) + " touches Red's cell on " +
                                hex_text(pairs[0].second);
      reason("touch",
             (pairs.size() == 1 ? first + ", and neither is a wall"
                                : std::to_string(pairs.size()) + " pairs of Blue's and Red's cells touch, and no cell of "
                                  "them is a wall (" + first + ")") +
               ": a placement captures each cell of the opponent that touches it, except walls.",
             hexes);
    }
  }

  // cell-count, wall-count: the cells of a player that are not its cells of the start
  int handWalls[2] = {0, 0};
  for(int player = 0; player < 2; player++) {
    int k[3];
    kinds(s.hand[player], k);
    handWalls[player] = k[WALL];
  }
  for(int player = 0; player < 2; player++) {
    int placed = 0, walls = 0;
    for(const auto& [h, v] : cells)
      if(cell_owner(v) == player && start_cell(h) != v) {
        placed++;
        walls += cell_type(v) == WALL;
      }
    const int m = moves[player];
    const std::string name = NAME[player];
    const int most = most_cells_placed(m);
    if(placed > most)
      reason("cell-count",
             m == 0 ? name + " has " + plural(placed, "cell", "cells") +
                        " that the start does not have, and " + name + " has made no move: only a placement puts cells."
                    : name + " has " + plural(placed, "cell", "cells") + " that the start does not have, more than its " +
                        plural(m, "move", "moves") + " can place (at most " + std::to_string(most) +
                        "): a placement puts one blueprint, of " + std::to_string(start_blueprint_cells()) +
                        " cells at the start and at most 1 more for each edit before it.",
             {});
    const int mostWalls = handWalls[player] * m;
    if(walls > mostWalls)
      reason("wall-count",
             m == 0 ? name + " has " + plural(walls, "wall", "walls") + " that the start does not have, and " + name +
                        " has made no move: only a placement puts walls."
                    : name + " has " + plural(walls, "wall", "walls") + " that the start does not have, more than its " +
                        plural(m, "move", "moves") + " can place (at most " + std::to_string(mostWalls) +
                        "): a placement puts at most " + std::to_string(handWalls[player]) + " walls, as a hand holds " +
                        std::to_string(handWalls[player]) + ".",
             {});
  }

  // hand, hand-edits
  for(int player = 0; player < 2; player++) {
    const std::string name = NAME[player];
    int have[3], want[3];
    kinds(hands[player], have);
    kinds(s.hand[player], want);
    auto kinds_text = [](const int (&k)[3]) {
      return plural(k[PLAIN], "Normal cell", "Normal cells") + ", " + plural(k[STAR], "star", "stars") + " and " +
             plural(k[WALL], "wall", "walls");
    };
    if(have[PLAIN] != want[PLAIN] || have[STAR] != want[STAR] || have[WALL] != want[WALL]) {
      reason("hand",
             name + "'s hand holds " + kinds_text(have) + ", not the start's " + std::to_string(want[PLAIN]) + ", " +
               std::to_string(want[STAR]) + " and " + std::to_string(want[WALL]) +
               ": an edit moves a cell from one blueprint to another, and a placement keeps the hand.",
             {});
      continue;
    }
    int more = 0;
    for(int k = 0; k < HAND; k++)
      more += std::max(0, hands[player][k].n - s.hand[player][k].n);
    const int m = moves[player];
    if(more > m)
      reason("hand-edits",
             name + "'s blueprints hold " + plural(more, "cell", "cells") + " more than at the start, and " + name +
               (m == 0 ? " has made no move" : " has made " + plural(m, "move", "moves")) +
               ": an edit adds at most 1 cell to a blueprint, so " + name + " needs at least " +
               plural(more, "edit", "edits") + ".",
             {});
    else if(more == 0 && m == 0 && !start_hand(player))
      reason("hand-edits",
             name + "'s hand is not the start's, and " + name + " has made no move: only an edit changes a hand.", {});
  }

  // score-negative, score-unmoved
  for(int player = 0; player < 2; player++) {
    const std::string name = NAME[player];
    if(p.scores[player] < 0)
      reason("score-negative",
             name + "'s score is " + std::to_string(p.scores[player]) + ": a move scores 0 or more points.", {});
    else if(p.scores[player] > 0 && moves[player] == 0)
      reason("score-unmoved",
             name + "'s score is " + std::to_string(p.scores[player]) + ", and " + name +
               " has made no move: only a placement scores.",
             {});
  }
  return reasons;
}

}  // namespace position_check
}  // namespace blueprints
