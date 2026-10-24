#pragma once
// Blueprints' rules part of the game interface (game/gameinterface.h): strategyjam.org's game of the Fall 2026 jam,
// version 3 of the site's rules module, in the site's move strings. The engine repository's games/blueprints/README.md
// describes the game, the board and the notation.
//
// - The board is a hex grid in axial coordinates (q, r), with the six neighbours of DIRS. The site's grid has no edge.
//   This port plays on the axial square |q| <= 20, |r| <= 20: the point (x, y) = (q + 20, r + 20) of a 41 x 41 board.
//   A placement must fit on it. That is the only rule of the port that the site does not have.
// - Each player has three blueprints: shapes of plain, star and wall cells, in the site's frame, where the lowest cell
//   (the least r, then the least q) is at (0, 0). A turn order of 1, 2, 2, ... moves ends after 120 moves (plies).
// - A move places a blueprint ("K@q,r": its (0, 0) cell lands on (q, r)) on empty cells, one of them next to a cell of
//   the mover. Each enemy cell next to the stamped cells that is not a wall goes. The mover scores the enemy stars that
//   go, times the mover's stars on the board before the move.
// - Or a move edits the hand ("I:q,r>J:q,r"): one cell of blueprint I goes to blueprint J, next to its cells (to
//   (0, 0) when J is empty). A move to another blueprint keeps blueprint I connected. A move within I keeps it
//   connected after the move, and puts the cell elsewhere. Both blueprints then go back to the site's frame.
// - This port plays an edit as two actions of the mover: the pick ("I:q,r") and the drop (">J:q,r"). So the site's
//   string of an edit is the pick's text, then the drop's. A pick is legal only when a drop follows it.
// - Pass is never legal. The site plays it when a clock runs out: a replay tolerates it, and it ends the move.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../core/hash.h"
#include "../gameinterface.h"

namespace blueprints {

constexpr int RADIUS = 20;              // the board: |q| <= RADIUS, |r| <= RADIUS
constexpr int LEN = 2 * RADIUS + 1;     // 41
constexpr int AREA = LEN * LEN;
constexpr int PLIES = 120;              // the moves of a game
constexpr int HAND = 3;                 // the blueprints of a player
constexpr int MAX_CELLS = 21;           // a player's cells in all its blueprints: 3 x 7
constexpr int PLAIN = 0, STAR = 1, WALL = 2;
constexpr const char* TYPE_LETTERS = "psw";  // plain, star, wall

// The six neighbours of (q, r), in the site's order.
constexpr int DIRS[6][2] = {{1, 0}, {1, -1}, {0, -1}, {-1, 0}, {-1, 1}, {0, 1}};

// The actions: a point and one of 9 channels, on the largest board, then pass.
// - Channels 0-2: place blueprint K at the board's point (q + RADIUS, r + RADIUS).
// - Channels 3-5: pick the cell (q, r) of blueprint I, at the hand's point (q + HAND_X, r + HAND_Y).
// - Channels 6-8: drop the picked cell into blueprint J at (q, r), at the hand's point.
// A blueprint's cells have 0 <= r <= 20 and |q| <= 20, and a drop's r is at least -1. So the hand's points fit.
constexpr int CHANNELS = 9;
constexpr int PLACE = 0, PICK = 3, DROP = 6;
constexpr int HAND_X = RADIUS, HAND_Y = 1;
inline int action_at(int x, int y, int channel) { return (y * LEN + x) * CHANNELS + channel; }
inline int action_x(int action) { return action / CHANNELS % LEN; }
inline int action_y(int action) { return action / CHANNELS / LEN; }
inline int action_channel(int action) { return action % CHANNELS; }

inline bool on_board(int x, int y) { return x >= 0 && x < LEN && y >= 0 && y < LEN; }

// A cell's value on the board: 0 when empty, else 1 + owner * 3 + type.
inline int8_t cell_value(int owner, int type) { return int8_t(1 + owner * 3 + type); }
inline int cell_owner(int8_t v) { return v ? (v - 1) / 3 : -1; }
inline int cell_type(int8_t v) { return (v - 1) % 3; }

// splitmix64's finalizer, as Connect-4's: a bijection of 64-bit words.
inline uint64_t mix64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
inline Hash128 key(uint64_t a, uint64_t b) {
  return Hash128(mix64(a * 0x9E3779B97F4A7C15ULL + b + 0x632BE59BD9B4E019ULL),
                 mix64(a * 0xC2B2AE3D27D4EB4FULL + b + 0x165667B19E3779F9ULL));
}

// A blueprint: its cells (q, r) and their types, in the site's frame (normalize), except the picked blueprint's rest.
struct Blueprint {
  int8_t n = 0;
  int8_t q[MAX_CELLS] = {}, r[MAX_CELLS] = {}, t[MAX_CELLS] = {};

  int find(int cq, int cr) const {
    for(int i = 0; i < n; i++)
      if(q[i] == cq && r[i] == cr)
        return i;
    return -1;
  }
  void add(int cq, int cr, int type) {
    q[n] = int8_t(cq);
    r[n] = int8_t(cr);
    t[n] = int8_t(type);
    n++;
  }
  void remove(int i) {
    for(int j = i + 1; j < n; j++) {
      q[j - 1] = q[j];
      r[j - 1] = r[j];
      t[j - 1] = t[j];
    }
    n--;
  }
  // The site's frame: the lowest cell (the least r, then the least q) moves to (0, 0) (the site's oe and ae).
  void normalize() {
    if(n == 0)
      return;
    int bq = q[0], br = r[0];
    for(int i = 1; i < n; i++)
      if(r[i] < br || (r[i] == br && q[i] < bq)) {
        bq = q[i];
        br = r[i];
      }
    for(int i = 0; i < n; i++) {
      q[i] = int8_t(q[i] - bq);
      r[i] = int8_t(r[i] - br);
    }
  }
  // Whether the blueprint is in the site's frame: empty, or its lowest cell at (0, 0).
  bool framed() const {
    if(n == 0)
      return true;
    if(find(0, 0) < 0)
      return false;
    for(int i = 0; i < n; i++)
      if(r[i] < 0 || (r[i] == 0 && q[i] < 0))
        return false;
    return true;
  }
  bool adjacent(int cq, int cr) const {
    for(int i = 0; i < n; i++)
      for(const auto& d : DIRS)
        if(q[i] + d[0] == cq && r[i] + d[1] == cr)
          return true;
    return false;
  }
  // Whether the cells, with (extraQ, extraR) when withExtra, are one connected group (the site's L). No cell: true.
  bool connected(bool withExtra = false, int extraQ = 0, int extraR = 0) const {
    int cq[MAX_CELLS + 1], cr[MAX_CELLS + 1];
    int m = 0;
    for(int i = 0; i < n; i++) {
      cq[m] = q[i];
      cr[m] = r[i];
      m++;
    }
    if(withExtra) {
      cq[m] = extraQ;
      cr[m] = extraR;
      m++;
    }
    if(m == 0)
      return true;
    bool seen[MAX_CELLS + 1] = {};
    int stack[MAX_CELLS + 1], top = 0, count = 1;
    seen[0] = true;
    stack[top++] = 0;
    while(top > 0) {
      const int a = stack[--top];
      for(int b = 0; b < m; b++) {
        if(seen[b])
          continue;
        const int dq = cq[b] - cq[a], dr = cr[b] - cr[a];
        for(const auto& d : DIRS)
          if(d[0] == dq && d[1] == dr) {
            seen[b] = true;
            stack[top++] = b;
            count++;
            break;
          }
      }
    }
    return count == m;
  }
};

struct Position {
  int8_t cell[AREA] = {};  // [y * LEN + x]: cell_value
  Blueprint hand[2][HAND];
  int32_t scores[2] = {0, 0};
  int16_t stars[2] = {0, 0};  // each player's stars on the board
  int16_t ply = 0;            // the site's moves played
  // The pick of an edit: the blueprint of the picked cell (-1: none), the cell and its type. The cell is out of its
  // blueprint, whose rest keeps its frame until the drop. pickConnected: whether that rest is connected.
  int8_t picked = -1, pickQ = 0, pickR = 0, pickType = 0;
  bool pickConnected = false;
};

// The player to move at a ply: the first plays ply 0, then each player two (the site's te).
inline int turn_of(int ply) { return ply == 0 || ((ply - 1) >> 1) % 2 == 1 ? 0 : 1; }
// The mover's moves left in its turn, this one included (the site's ne).
inline int left_of(int ply) { return ply == 0 ? 1 : std::min(2 - (ply - 1) % 2, PLIES - ply); }

inline int8_t at(const Position& p, int x, int y) { return on_board(x, y) ? p.cell[y * LEN + x] : int8_t(0); }

inline bool next_to_own(const Position& p, int x, int y, int player) {
  for(const auto& d : DIRS)
    if(cell_owner(at(p, x + d[0], y + d[1])) == player)
      return true;
  return false;
}

// Whether placing blueprint k of the mover with its (0, 0) cell on the board's point (x, y) is legal: every cell on
// the board and empty, and one of them next to a cell of the mover (the site's H, on this port's board).
inline bool can_place(const Position& p, int k, int x, int y) {
  const int mover = turn_of(p.ply);
  const Blueprint& b = p.hand[mover][k];
  if(b.n == 0)
    return false;
  bool touches = false;
  for(int i = 0; i < b.n; i++) {
    const int cx = x + b.q[i], cy = y + b.r[i];
    if(!on_board(cx, cy) || p.cell[cy * LEN + cx])
      return false;
    touches = touches || next_to_own(p, cx, cy, mover);
  }
  return touches;
}

// Whether the picked cell can go to blueprint j at (q, r) (the site's pe, after the pick), on the hand's points.
inline bool can_drop(const Position& p, int j, int q, int r) {
  if(p.picked < 0 || !on_board(q + HAND_X, r + HAND_Y))
    return false;
  const int mover = turn_of(p.ply);
  const Blueprint& target = p.hand[mover][j];
  if(target.find(q, r) >= 0)
    return false;
  if(target.n == 0 ? !(q == 0 && r == 0) : !target.adjacent(q, r))
    return false;
  if(j != p.picked)
    return p.pickConnected;
  if(q == p.pickQ && r == p.pickR)
    return false;
  return p.pickConnected || target.connected(true, q, r);
}

// The drops after the pick, in ascending order of action.
inline void drops(const Position& p, std::vector<int>& out) {
  const int mover = turn_of(p.ply);
  for(int j = 0; j < HAND; j++) {
    const Blueprint& target = p.hand[mover][j];
    auto consider = [&](int q, int r) {
      if(can_drop(p, j, q, r))
        out.push_back(action_at(q + HAND_X, r + HAND_Y, DROP + j));
    };
    if(target.n == 0)
      consider(0, 0);
    else
      for(int i = 0; i < target.n; i++)
        for(const auto& d : DIRS)
          consider(target.q[i] + d[0], target.r[i] + d[1]);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
}

// Takes the cell (q, r) out of the mover's blueprint k: the pick.
inline void take(Position& p, int k, int q, int r) {
  Blueprint& b = p.hand[turn_of(p.ply)][k];
  const int i = b.find(q, r);
  p.picked = int8_t(k);
  p.pickQ = int8_t(q);
  p.pickR = int8_t(r);
  p.pickType = b.t[i];
  b.remove(i);
  p.pickConnected = b.connected();
}

// Whether the pick of the cell (q, r) of the mover's blueprint k is legal: the cell exists and a drop follows.
inline bool can_pick(const Position& p, int k, int q, int r) {
  if(p.picked >= 0 || p.hand[turn_of(p.ply)][k].find(q, r) < 0)
    return false;
  Position after = p;
  take(after, k, q, r);
  std::vector<int> out;
  drops(after, out);
  return !out.empty();
}

// Stamps blueprint b of the player with its (0, 0) cell at (x, y): the stamped cells, the enemy cells that go and the
// score (the site's ce and le). The caller checked the placement.
inline void stamp(Position& p, int player, const Blueprint& b, int x, int y) {
  const int enemy = 1 - player;
  int removedStars = 0;
  // The cells that go: next to a stamped cell, the enemy's, not walls. Each counts once.
  std::vector<int> removed;
  for(int i = 0; i < b.n; i++)
    for(const auto& d : DIRS) {
      const int nx = x + b.q[i] + d[0], ny = y + b.r[i] + d[1];
      const int8_t v = at(p, nx, ny);
      if(cell_owner(v) == enemy && cell_type(v) != WALL) {
        const int index = ny * LEN + nx;
        if(std::find(removed.begin(), removed.end(), index) == removed.end())
          removed.push_back(index);
      }
    }
  for(int index : removed)
    removedStars += cell_type(p.cell[index]) == STAR;
  p.scores[player] += removedStars * p.stars[player];
  for(int index : removed)
    p.cell[index] = 0;
  p.stars[enemy] = int16_t(p.stars[enemy] - removedStars);
  for(int i = 0; i < b.n; i++) {
    p.cell[(y + b.r[i]) * LEN + x + b.q[i]] = cell_value(player, b.t[i]);
    p.stars[player] = int16_t(p.stars[player] + (b.t[i] == STAR));
  }
}

// The start's blueprints: a star with a ring of six cells (the site's re and ie), in the site's frame.
inline Blueprint flower(const int ring[6]) {
  Blueprint b;
  b.add(0, 0, STAR);
  for(int i = 0; i < 6; i++)
    b.add(DIRS[i][0], DIRS[i][1], ring[i]);
  b.normalize();
  return b;
}

inline void setup(Position& p) {
  p = Position();
  const int plain[6] = {PLAIN, PLAIN, PLAIN, PLAIN, PLAIN, PLAIN};
  const int firsts[6] = {WALL, STAR, WALL, STAR, WALL, STAR};
  const int seconds[6] = {STAR, WALL, STAR, WALL, STAR, WALL};
  const int starts[2][HAND] = {{-6, -10, -2}, {6, 10, 2}};  // the site's P: each blueprint's offset, at r = -1
  for(int player = 0; player < 2; player++) {
    p.hand[player][0] = flower(plain);
    p.hand[player][1] = flower(player == 0 ? firsts : seconds);
    p.hand[player][2] = flower(plain);
    for(int k = 0; k < HAND; k++)
      stamp(p, player, p.hand[player][k], starts[player][k] + RADIUS, -1 + RADIUS);
  }
}

// A cell's text: "q,r", integers.
inline bool parse_cell(const std::string& text, int& q, int& r) {
  const size_t comma = text.find(',');
  if(comma == std::string::npos || comma == 0 || comma + 1 >= text.size())
    return false;
  auto number = [](const std::string& s, int& out) {
    size_t i = 0;
    if(s[0] == '-')
      i = 1;
    if(i >= s.size() || s.size() - i > 3)
      return false;
    for(size_t j = i; j < s.size(); j++)
      if(!std::isdigit((unsigned char)s[j]))
        return false;
    out = std::atoi(s.c_str());
    return true;
  };
  return number(text.substr(0, comma), q) && number(text.substr(comma + 1), r);
}

inline std::string cell_text(int q, int r) { return std::to_string(q) + "," + std::to_string(r); }

// Splits text at each separator: "" gives one empty field.
inline std::vector<std::string> split(const std::string& text, char separator) {
  std::vector<std::string> out(1);
  for(char c : text) {
    if(c == separator)
      out.emplace_back();
    else
      out.back() += c;
  }
  return out;
}

struct GameRules {
  static constexpr const char* NAME = "blueprints";
  using Position = blueprints::Position;
  static constexpr int MAX_BOARD_LEN = LEN;
  static constexpr int NUM_ACTIONS = AREA * CHANNELS + 1;
  static constexpr int PASS_ACTION = AREA * CHANNELS;

  static bool new_game(int xSize, int ySize, Position& p) {
    if(xSize != LEN || ySize != LEN)
      return false;
    setup(p);
    return true;
  }
  static int x_size(const Position&) { return LEN; }
  static int y_size(const Position&) { return LEN; }
  static int next_player(const Position& p) { return turn_of(p.ply); }
  // The site's moves left in the turn: a pick and its drop are one of them.
  static int turn_moves_left(const Position& p) { return left_of(p.ply); }
  static int moves_played(const Position& p) { return p.ply; }
  static bool terminal(const Position& p) { return p.ply >= PLIES; }

  static void legal_actions(const Position& p, std::vector<int>& out) {
    out.clear();
    if(terminal(p))
      return;
    if(p.picked >= 0) {
      drops(p, out);
      return;
    }
    const int mover = turn_of(p.ply);
    // The placements: each anchor that puts a cell of the blueprint on an empty point next to the mover (the
    // frontier), checked once.
    std::vector<int8_t> frontier(AREA, 0);
    for(int y = 0; y < LEN; y++)
      for(int x = 0; x < LEN; x++)
        frontier[y * LEN + x] = !p.cell[y * LEN + x] && next_to_own(p, x, y, mover);
    std::vector<int8_t> seen(AREA);
    for(int k = 0; k < HAND; k++) {
      const Blueprint& b = p.hand[mover][k];
      std::fill(seen.begin(), seen.end(), 0);
      for(int f = 0; f < AREA; f++) {
        if(!frontier[f])
          continue;
        for(int i = 0; i < b.n; i++) {
          const int x = f % LEN - b.q[i], y = f / LEN - b.r[i];
          if(!on_board(x, y) || seen[y * LEN + x])
            continue;
          seen[y * LEN + x] = 1;
          if(can_place(p, k, x, y))
            out.push_back(action_at(x, y, PLACE + k));
        }
      }
    }
    // The picks.
    for(int k = 0; k < HAND; k++) {
      const Blueprint& b = p.hand[mover][k];
      for(int i = 0; i < b.n; i++)
        if(can_pick(p, k, b.q[i], b.r[i]))
          out.push_back(action_at(b.q[i] + HAND_X, b.r[i] + HAND_Y, PICK + k));
    }
    std::sort(out.begin(), out.end());
  }
  static bool is_legal(const Position& p, int action) {
    if(action < 0 || action >= PASS_ACTION || terminal(p))
      return false;
    const int x = action_x(action), y = action_y(action), channel = action_channel(action);
    if(channel >= DROP)
      return can_drop(p, channel - DROP, x - HAND_X, y - HAND_Y);
    if(p.picked >= 0)
      return false;
    if(channel >= PICK)
      return can_pick(p, channel - PICK, x - HAND_X, y - HAND_Y);
    return can_place(p, channel - PLACE, x, y);
  }
  static void play(Position& p, int action) {
    const int mover = turn_of(p.ply);
    if(action == PASS_ACTION) {
      // A timeout's pass ends the move. A pick without its drop goes back.
      if(p.picked >= 0) {
        Blueprint& b = p.hand[mover][p.picked];
        b.add(p.pickQ, p.pickR, p.pickType);
        b.normalize();
        p.picked = -1;
      }
      p.ply++;
      return;
    }
    const int x = action_x(action), y = action_y(action), channel = action_channel(action);
    if(channel >= DROP) {
      const int j = channel - DROP;
      p.hand[mover][j].add(x - HAND_X, y - HAND_Y, p.pickType);
      p.hand[mover][j].normalize();
      p.hand[mover][p.picked].normalize();
      p.picked = -1;
      p.ply++;
    }
    else if(channel >= PICK)
      take(p, channel - PICK, x - HAND_X, y - HAND_Y);
    else {
      stamp(p, mover, p.hand[mover][channel - PLACE], x, y);
      p.ply++;
    }
  }
  static bool replay_tolerates(const Position&, int action) { return action == PASS_ACTION; }

  // The higher score wins. Equal scores: a draw.
  static float result(const Position& p) {
    return p.scores[0] == p.scores[1] ? 0.0f : p.scores[0] > p.scores[1] ? 1.0f : -1.0f;
  }
  static double score(const Position& p, int player) { return p.scores[player]; }

  // Each cell, each blueprint's cells, the ply, the pick and the scores, hashed by key(): a hash of the whole position.
  static Hash128 hash(const Position& p) {
    Hash128 h = key(1, 0);
    for(int i = 0; i < AREA; i++)
      if(p.cell[i])
        h ^= key(2 + i, uint64_t(p.cell[i]));
    for(int player = 0; player < 2; player++)
      for(int k = 0; k < HAND; k++) {
        const Blueprint& b = p.hand[player][k];
        for(int i = 0; i < b.n; i++)
          h ^= key(10000 + (player * HAND + k) * 4096 + (b.q[i] + 32) * 64 + (b.r[i] + 32), uint64_t(b.t[i]));
      }
    h ^= key(40000 + uint64_t(p.ply), 0);
    if(p.picked >= 0)
      h ^= key(41000 + uint64_t(p.picked) * 4096 + (p.pickQ + 32) * 64 + (p.pickR + 32), uint64_t(p.pickType));
    h ^= key(50000, uint64_t(uint32_t(p.scores[0])));
    h ^= key(50001, uint64_t(uint32_t(p.scores[1])));
    return h;
  }

  static void pieces(const Position& p, int8_t* owners) {
    for(int i = 0; i < AREA; i++)
      owners[i] = int8_t(cell_owner(p.cell[i]));
  }
  static void area(const Position& p, int8_t* owners) { pieces(p, owners); }

  // The site's strings: "K@q,r" (a placement), "I:q,r" (a pick) and ">J:q,r" (a drop), with blueprints from 1.
  static std::string action_text(int action) {
    if(action == PASS_ACTION)
      return "pass";
    const int x = action_x(action), y = action_y(action), channel = action_channel(action);
    if(channel >= DROP)
      return ">" + std::to_string(channel - DROP + 1) + ":" + cell_text(x - HAND_X, y - HAND_Y);
    if(channel >= PICK)
      return std::to_string(channel - PICK + 1) + ":" + cell_text(x - HAND_X, y - HAND_Y);
    return std::to_string(channel - PLACE + 1) + "@" + cell_text(x - RADIUS, y - RADIUS);
  }
  static std::string protocol_text(int action) { return action_text(action); }
  // The action that a placement's, a pick's or a drop's string names, legal or not. False for any other text, and
  // for a point off this port's board or the hand's points.
  static bool parse_protocol_text(const std::string& text, int& action) {
    if(text.empty())
      return false;
    const bool drop = text[0] == '>';
    const std::string s = drop ? text.substr(1) : text;
    const size_t sep = s.find_first_of("@:");
    if(sep != 1 || s[0] < '1' || s[0] > '0' + HAND)
      return false;
    const int k = s[0] - '1';
    const bool place = s[sep] == '@';
    if(drop && place)
      return false;
    int q, r;
    if(!parse_cell(s.substr(sep + 1), q, r))
      return false;
    const int x = q + (place ? RADIUS : HAND_X), y = r + (place ? RADIUS : HAND_Y);
    if(!on_board(x, y))
      return false;
    action = action_at(x, y, (place ? PLACE : drop ? DROP : PICK) + k);
    return true;
  }
  static int parse_action(const Position& p, const std::string& text, std::string* why) {
    auto fail = [why](const std::string& reason) {
      if(why)
        *why = reason;
      return -1;
    };
    int action;
    if(!parse_protocol_text(text, action))
      return fail("not a move: \"" + text + "\" (K@q,r places blueprint K, I:q,r picks a cell, >J:q,r drops it; " +
                  "the board is |q|, |r| <= " + std::to_string(RADIUS) + ")");
    if(terminal(p))
      return fail("the game is over");
    if(!is_legal(p, action))
      return fail(text + " is not legal here");
    return action;
  }

  // The one-line position, seven fields: the ply, the two scores, the board ("q,r,T" with T of psw for the first
  // player and PSW for the second, joined by ;), the first player's three blueprints and the second's ("q,r,t" joined
  // by ;, the blueprints joined by |), and the pick ("I:q,r,t" of the mover, its cell out of blueprint I). "-" is
  // empty.
  static std::string position_text(const Position& p) {
    std::string s = std::to_string(p.ply) + " " + std::to_string(p.scores[0]) + " " + std::to_string(p.scores[1]) + " ";
    std::string board;
    for(int y = 0; y < LEN; y++)
      for(int x = 0; x < LEN; x++) {
        const int8_t v = p.cell[y * LEN + x];
        if(!v)
          continue;
        if(!board.empty())
          board += ';';
        const char letter = TYPE_LETTERS[cell_type(v)];
        board += cell_text(x - RADIUS, y - RADIUS) + "," + char(cell_owner(v) ? std::toupper(letter) : letter);
      }
    s += board.empty() ? "-" : board;
    for(int player = 0; player < 2; player++) {
      s += ' ';
      for(int k = 0; k < HAND; k++) {
        if(k > 0)
          s += '|';
        const Blueprint& b = p.hand[player][k];
        if(b.n == 0)
          s += '-';
        for(int i = 0; i < b.n; i++)
          s += (i > 0 ? ";" : "") + cell_text(b.q[i], b.r[i]) + "," + TYPE_LETTERS[b.t[i]];
      }
    }
    s += ' ';
    if(p.picked < 0)
      s += '-';
    else
      s += std::to_string(p.picked + 1) + ":" + cell_text(p.pickQ, p.pickR) + "," + TYPE_LETTERS[p.pickType];
    return s;
  }
  // Checks the fields, the board's points, each blueprint's frame and connection, and the cells of each player's hand.
  static bool parse_position(const std::string& text, Position& p, std::string& why) {
    const std::vector<std::string> fields = split(text, ' ');
    if(fields.size() != 7) {
      why = "a position is seven fields: the ply, the two scores, the board, the two hands and the pick";
      return false;
    }
    Position q;
    auto integer = [](const std::string& s, long& out) {
      if(s.empty() || s.size() > 10)
        return false;
      for(size_t i = (s[0] == '-'); i < s.size(); i++)
        if(!std::isdigit((unsigned char)s[i]))
          return false;
      out = std::atol(s.c_str());
      return true;
    };
    long ply, s0, s1;
    if(!integer(fields[0], ply) || ply < 0 || ply > PLIES || !integer(fields[1], s0) || !integer(fields[2], s1)) {
      why = "the ply is 0 to " + std::to_string(PLIES) + ", and the scores are integers";
      return false;
    }
    q.ply = int16_t(ply);
    q.scores[0] = int32_t(s0);
    q.scores[1] = int32_t(s1);
    // A cell's three parts: "q,r,T".
    auto cell3 = [](const std::string& s, int& cq, int& cr, char& letter) {
      const size_t last = s.rfind(',');
      if(last == std::string::npos || last + 2 != s.size())
        return false;
      letter = s[last + 1];
      return parse_cell(s.substr(0, last), cq, cr);
    };
    if(fields[3] != "-")
      for(const std::string& c : split(fields[3], ';')) {
        int cq, cr;
        char letter;
        if(!cell3(c, cq, cr, letter) || !std::strchr(TYPE_LETTERS, std::tolower(letter)) || letter == 0) {
          why = "a cell of the board is q,r,T with T one of psw (the first player) or PSW (the second): \"" + c + "\"";
          return false;
        }
        const int x = cq + RADIUS, y = cr + RADIUS;
        if(!on_board(x, y) || q.cell[y * LEN + x]) {
          why = "the cell " + cell_text(cq, cr) + " is off the board or given twice";
          return false;
        }
        const int owner = std::isupper((unsigned char)letter) ? 1 : 0;
        const int type = int(std::strchr(TYPE_LETTERS, std::tolower(letter)) - TYPE_LETTERS);
        q.cell[y * LEN + x] = cell_value(owner, type);
        q.stars[owner] = int16_t(q.stars[owner] + (type == STAR));
      }
    const int mover = turn_of(q.ply);
    if(fields[6] != "-") {
      int cq, cr;
      char letter;
      const std::string& f = fields[6];
      const char* type = f.size() > 2 ? std::strchr(TYPE_LETTERS, f.back()) : nullptr;
      if(f.size() < 7 || f[1] != ':' || f[0] < '1' || f[0] > '0' + HAND || !cell3(f.substr(2), cq, cr, letter) ||
         !type || q.ply >= PLIES) {
        why = "the pick is I:q,r,t (blueprint I from 1, t one of psw), before the end";
        return false;
      }
      q.picked = int8_t(f[0] - '1');
      q.pickQ = int8_t(cq);
      q.pickR = int8_t(cr);
      q.pickType = int8_t(type - TYPE_LETTERS);
    }
    for(int player = 0; player < 2; player++) {
      const std::vector<std::string> bps = split(fields[4 + player], '|');
      if(bps.size() != HAND) {
        why = "a hand is three blueprints, joined by |";
        return false;
      }
      int total = 0;
      for(int k = 0; k < HAND; k++) {
        Blueprint& b = q.hand[player][k];
        if(bps[k] != "-")
          for(const std::string& c : split(bps[k], ';')) {
            int cq, cr;
            char letter;
            const char* type = nullptr;
            if(!cell3(c, cq, cr, letter) || !(type = std::strchr(TYPE_LETTERS, letter)) || letter == 0 ||
               b.n == MAX_CELLS || b.find(cq, cr) >= 0 || !on_board(cq + HAND_X, cr + HAND_Y)) {
              why = "a blueprint's cell is q,r,t with t one of psw, each once, on the hand's points: \"" + c + "\"";
              return false;
            }
            b.add(cq, cr, int(type - TYPE_LETTERS));
          }
        total += b.n;
        const bool rest = player == mover && k == q.picked;
        if(!rest && (!b.framed() || !b.connected())) {
          why = "blueprint " + std::to_string(k + 1) + " of player " + std::to_string(player) +
                " is not connected or not in the site's frame (its lowest cell at 0,0)";
          return false;
        }
      }
      const int expected = MAX_CELLS - (player == mover && q.picked >= 0);
      if(total != expected) {
        why = "player " + std::to_string(player) + "'s blueprints hold " + std::to_string(total) + " cells, not " +
              std::to_string(expected);
        return false;
      }
    }
    if(q.picked >= 0) {
      if(q.hand[mover][q.picked].find(q.pickQ, q.pickR) >= 0) {
        why = "the picked cell is still in its blueprint";
        return false;
      }
      q.pickConnected = q.hand[mover][q.picked].connected();
    }
    p = q;
    return true;
  }
};

static_assert(GameInterface::checkRules<GameRules>(), "Blueprints' rules part");

}  // namespace blueprints
