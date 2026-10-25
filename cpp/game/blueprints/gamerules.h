#pragma once
// Blueprints' rules part of the game interface (game/gameinterface.h): strategyjam.org's game of the Fall 2026 jam,
// version 3 of the site's rules module, in the site's move strings. The engine repository's games/blueprints/README.md
// describes the game, the boards and the notation.
//
// - The board is a hex grid in axial coordinates (q, r), with the six neighbours of DIRS. The site's grid has no edge.
//   This port keeps the cells on a rules board of RULES_LEN x RULES_LEN points that follows the game: after each
//   placement the cells move so that their bounding box is in its middle (recenter), and the position keeps the
//   site's coordinates of that middle (offsetQ, offsetR). The rules do not depend on where the cells are. Every game of
//   the site's archive fits (games/blueprints/docs/decisions/port.md).
// - The front's board is the view: VIEW x VIEW points around the window's middle, which follows the mover's own play
//   (window_middle: the middle of the cells that the mover's last 2 placements stamped). The network sees the view, and
//   the engine places a blueprint only with all its cells in the view: a rule on the cells, which a symmetry maps as a
//   set (a rule on the (0, 0) cell would not hold under a symmetry, since the image's lowest cell is another cell). The
//   view and the rules board's size are the two rules of the port that the site does not have.
// - Each player has three blueprints: shapes of plain, star and wall cells, in the site's frame, where the lowest cell
//   (the least r, then the least q) is at (0, 0). A turn order of 1, 2, 2, ... moves ends after 120 moves (plies).
// - A move places a blueprint ("K@q,r": its (0, 0) cell lands on (q, r)) on empty cells, one of them next to a cell of
//   the mover. Each enemy cell next to the stamped cells that is not a wall goes. The mover scores the enemy stars that
//   go, times the mover's stars on the board before the move.
// - Or a move edits the hand ("I:q,r>J:q,r"): one cell of blueprint I goes to blueprint J, next to its cells (to
//   (0, 0) when J is empty). A move to another blueprint keeps blueprint I connected. A move within I keeps it
//   connected after the move, and puts the cell elsewhere. Both blueprints then go back to the site's frame.
// - This port plays an edit as two actions of the mover: the pick ("I:q,r") and the drop ("J:q,r"). So the site's
//   string of an edit is the pick's text, ">", then the drop's. A pick is legal only when a drop follows it.
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

constexpr int RULES_RADIUS = 127;                // the rules board: |q|, |r| <= RULES_RADIUS from its middle
constexpr int RULES_LEN = 2 * RULES_RADIUS + 1;  // 255
constexpr int RULES_AREA = RULES_LEN * RULES_LEN;
constexpr int VIEW_RADIUS = 33;                  // the view: |q|, |r| <= VIEW_RADIUS from the window's middle
constexpr int VIEW = 2 * VIEW_RADIUS + 1;        // 67, the front's board
constexpr int VIEW_AREA = VIEW * VIEW;
constexpr int PLIES = 120;              // the moves of a game
constexpr int HAND = 3;                 // the blueprints of a player
constexpr int MAX_CELLS = 21;           // a player's cells in all its blueprints: 3 x 7
constexpr int RECENT = 2;               // the mover's placements that set the window
constexpr int PLAIN = 0, STAR = 1, WALL = 2;
constexpr const char* TYPE_LETTERS = "psw";  // plain, star, wall

// The six neighbours of (q, r), in the site's order.
constexpr int DIRS[6][2] = {{1, 0}, {1, -1}, {0, -1}, {-1, 0}, {-1, 1}, {0, 1}};

// The actions: a point of the view and one of 6 channels, then pass.
// - Channels 0-2: place blueprint K with its (0, 0) cell at the view's point (q + VIEW_RADIUS, r + VIEW_RADIUS), (q, r)
//   from the window's middle.
// - Channels 3-5: the cell (q, r) of blueprint K, at the hand's point (q + HAND_X, r + HAND_Y): a pick of that cell
//   when no pick waits for its drop, else the drop of the picked cell there. A position never has both a legal pick
//   and a legal drop, so the two share these channels.
// A blueprint's cells have 0 <= r <= 20 and |q| <= 20, and a drop's cell is next to them. So the hand's points fit.
// The front keeps an action in a short (Board, "move Locs must fit in a short"): its points' array and the actions
// stay under 32,768, so 67 x 67 is the largest view of 6 channels.
constexpr int CHANNELS = 6;
constexpr int PLACE = 0, HAND_CHANNEL = 3;
constexpr int HAND_X = VIEW_RADIUS, HAND_Y = 1;
inline int action_at(int x, int y, int channel) { return (y * VIEW + x) * CHANNELS + channel; }
inline int action_x(int action) { return action / CHANNELS % VIEW; }
inline int action_y(int action) { return action / CHANNELS / VIEW; }
inline int action_channel(int action) { return action % CHANNELS; }

inline bool on_rules(int x, int y) { return x >= 0 && x < RULES_LEN && y >= 0 && y < RULES_LEN; }
inline bool on_view(int x, int y) { return x >= 0 && x < VIEW && y >= 0 && y < VIEW; }

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
// A point of the site's grid as one number, for the hash: a cell's key (by its site point) starts above 10^9.
inline uint64_t site_key(int q, int r) { return uint64_t(q + 32768) << 16 | uint64_t(r + 32768); }
// The first part of each other kind of the hash's keys, one range for each kind, so that no two keys meet. A
// blueprint's cell has |q| <= 20 and 0 <= r <= 20, so its number (q + 32) * 64 + r + 32 is below 4,096.
constexpr uint64_t KEY_HAND = 10000;     // + (player * HAND + k) * 4096 + a cell's number, and its type
constexpr uint64_t KEY_PLY = 40000;      // + the ply
constexpr uint64_t KEY_PICK = 41000;     // + k * 4096 + the picked cell's number, and its type
constexpr uint64_t KEY_SCORE = 55000;    // + the player, and the score
constexpr uint64_t KEY_BOX = 60000;      // + player * RECENT + i, and the recent placement's box
constexpr uint64_t KEY_OFFSET = 60100;   // the offset
constexpr uint64_t KEY_HISTORY = 70000;  // gameencoding.h's input_hash: + h * 2 * RULES_AREA + a point
static_assert(KEY_HAND + 6 * 4096 <= KEY_PLY && KEY_PLY + 121 <= KEY_PICK && KEY_PICK + 3 * 4096 <= KEY_SCORE &&
              KEY_SCORE + 2 <= KEY_BOX && KEY_BOX + 4 <= KEY_OFFSET && KEY_OFFSET < KEY_HISTORY,
              "the hash's ranges of keys do not meet");

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

// The last actions that the network sees (the history planes, gameencoding.h): for each, the points of the rules
// board that a placement stamped and whose cells it took. An edit's actions and a pass leave the board as it was: no
// points. A connected shape of n cells has at most 2n + 4 cells next to it, so a placement takes at most 46.
constexpr int HISTORY = 4;
constexpr int MAX_TAKEN = 2 * MAX_CELLS + 4;
struct Footprint {
  int8_t nStamped = 0, nTaken = 0;
  int32_t stamped[MAX_CELLS] = {}, taken[MAX_TAKEN] = {};  // points of the rules board, y * RULES_LEN + x
};

// A box on the rules board: the least and the most x and y of some points (any: whether it holds a point), such as the
// cells that a placement stamped.
struct Box {
  bool any = false;
  int16_t x0 = 0, x1 = 0, y0 = 0, y1 = 0;

  void add(int x, int y) {
    x0 = int16_t(any ? std::min<int>(x0, x) : x);
    x1 = int16_t(any ? std::max<int>(x1, x) : x);
    y0 = int16_t(any ? std::min<int>(y0, y) : y);
    y1 = int16_t(any ? std::max<int>(y1, y) : y);
    any = true;
  }
  void add(const Box& b) {
    if(b.any) {
      add(b.x0, b.y0);
      add(b.x1, b.y1);
    }
  }
};

// Half the distance of the middle of lo..hi from the rules board's middle, rounded toward zero (as C++ divides), so
// that it commutes with the symmetries (gameencoding.h): they negate and swap the coordinates around the middle.
inline int half_off_middle(int lo, int hi) { return (lo + hi - 2 * RULES_RADIUS) / 2; }

// The rules board's cells, [y * RULES_LEN + x]: cell_value. The box `used` holds every cell (it can be larger than the
// cells' box), and every point outside it is empty. So a copy, a clear and a scan of the cells read that box, not the
// RULES_AREA points (a copy of the whole board was the largest part of a position's copy: abscls-iwsg). Every write
// goes through set(), which grows the box.
class Cells {
  int8_t v[RULES_AREA] = {};

  void clear_box(const Box& b) {
    if(b.any)
      for(int y = b.y0; y <= b.y1; y++)
        std::memset(v + y * RULES_LEN + b.x0, 0, size_t(b.x1 - b.x0 + 1));
  }
  void copy_box(const Cells& o) {
    const Box& b = o.used;
    if(b.any)
      for(int y = b.y0; y <= b.y1; y++)
        std::memcpy(v + y * RULES_LEN + b.x0, o.v + y * RULES_LEN + b.x0, size_t(b.x1 - b.x0 + 1));
  }

 public:
  Box used;

  Cells() = default;
  Cells(const Cells& o) : used(o.used) { copy_box(o); }
  Cells& operator=(const Cells& o) {
    if(this != &o) {
      clear_box(used);
      copy_box(o);
      used = o.used;
    }
    return *this;
  }
  int8_t operator[](int index) const { return v[index]; }
  const int8_t* row(int y) const { return v + y * RULES_LEN; }
  void set(int index, int8_t value) {
    v[index] = value;
    if(value)
      used.add(index % RULES_LEN, index / RULES_LEN);
  }
  // Every point empty.
  void clear() {
    clear_box(used);
    used = Box();
  }
};

struct Position {
  Cells cell;
  Blueprint hand[2][HAND];
  int32_t scores[2] = {0, 0};
  int16_t stars[2] = {0, 0};  // each player's stars on the board
  int16_t cells[2] = {0, 0};  // each player's cells on the board
  int16_t ply = 0;            // the site's moves played
  // The pick of an edit: the blueprint of the picked cell (-1: none), the cell and its type. The cell is out of its
  // blueprint, whose rest keeps its frame until the drop. pickConnected: whether that rest is connected.
  int8_t picked = -1, pickQ = 0, pickR = 0, pickType = 0;
  bool pickConnected = false;
  // Each player's last RECENT placements, the newest first: where the window is when the player moves.
  Box recent[2][RECENT];
  // The site's coordinates (q, r) of the rules board's middle point (RULES_RADIUS, RULES_RADIUS): the board follows the
  // game (recenter).
  int16_t offsetQ = 0, offsetR = 0;
  // The hash of the cells, by their site coordinates, kept as cells come and go (so a recentering keeps it).
  Hash128 cellHash;
  // The last HISTORY actions, the newest first. Not part of the game's state: hash() leaves it out, and the one-line
  // position does not hold it.
  Footprint history[HISTORY];
};

// The player to move at a ply: the first plays ply 0, then each player two (the site's te).
inline int turn_of(int ply) { return ply == 0 || ((ply - 1) >> 1) % 2 == 1 ? 0 : 1; }
// The mover's moves left in its turn, this one included (the site's ne).
inline int left_of(int ply) { return ply == 0 ? 1 : std::min(2 - (ply - 1) % 2, PLIES - ply); }

inline int8_t at(const Position& p, int x, int y) { return on_rules(x, y) ? p.cell[y * RULES_LEN + x] : int8_t(0); }

inline void set_cell(Position& p, int index, int8_t v) {
  const int x = index % RULES_LEN - RULES_RADIUS + p.offsetQ, y = index / RULES_LEN - RULES_RADIUS + p.offsetR;
  const int8_t old = p.cell[index];
  if(old) {
    p.cellHash ^= key(site_key(x, y), uint64_t(old));
    p.cells[cell_owner(old)]--;
  }
  if(v) {
    p.cellHash ^= key(site_key(x, y), uint64_t(v));
    p.cells[cell_owner(v)]++;
  }
  p.cell.set(index, v);
}

// The window's middle, a point of the rules board: the middle of the box of the player's last placements, rounded
// toward zero from the board's middle (half_off_middle), or the board's middle when the player has placed nothing.
inline void window_middle(const Position& p, int player, int& wx, int& wy) {
  Box box;
  for(const Box& b : p.recent[player])
    box.add(b);
  wx = RULES_RADIUS + (box.any ? half_off_middle(box.x0, box.x1) : 0);
  wy = RULES_RADIUS + (box.any ? half_off_middle(box.y0, box.y1) : 0);
}
// The mover's window: the rules board's point of the view's point (0, 0).
inline void view_origin(const Position& p, int& ox, int& oy) {
  window_middle(p, turn_of(p.ply), ox, oy);
  ox -= VIEW_RADIUS;
  oy -= VIEW_RADIUS;
}

inline bool next_to_own(const Position& p, int x, int y, int player) {
  for(const auto& d : DIRS)
    if(cell_owner(at(p, x + d[0], y + d[1])) == player)
      return true;
  return false;
}

// Whether the player's blueprint b fits with its (0, 0) cell on the rules board's point (x, y): every cell on the
// board and empty, and one of them next to a cell of the player (the site's H, on this port's board).
inline bool fits(const Position& p, int player, const Blueprint& b, int x, int y) {
  if(b.n == 0)
    return false;
  bool touches = false;
  for(int i = 0; i < b.n; i++) {
    const int cx = x + b.q[i], cy = y + b.r[i];
    if(!on_rules(cx, cy) || p.cell[cy * RULES_LEN + cx])
      return false;
    touches = touches || next_to_own(p, cx, cy, player);
  }
  return touches;
}

// Whether every cell of blueprint b with its (0, 0) cell on the view's point (x, y) lies in the view.
inline bool in_view(const Blueprint& b, int x, int y) {
  for(int i = 0; i < b.n; i++)
    if(!on_view(x + b.q[i], y + b.r[i]))
      return false;
  return true;
}

// Whether the engine may place the player's blueprint b with its (0, 0) cell on the view's point (x, y), the view's
// origin on the rules point (ox, oy): every cell in the view, and the placement fits (fits).
inline bool fits_in_view(const Position& p, int player, const Blueprint& b, int ox, int oy, int x, int y) {
  return b.n > 0 && in_view(b, x, y) && fits(p, player, b, x + ox, y + oy);
}

// The player's frontier in the view whose origin is the rules point (ox, oy): the empty points next to the player's
// cells, as rules points. A placement within the view puts a cell on one of them.
inline void frontier_near(const Position& p, int player, int ox, int oy, std::vector<int32_t>& out) {
  out.clear();
  for(int fy = std::max(0, oy); fy < std::min(RULES_LEN, oy + VIEW); fy++)
    for(int fx = std::max(0, ox); fx < std::min(RULES_LEN, ox + VIEW); fx++)
      if(!p.cell[fy * RULES_LEN + fx] && next_to_own(p, fx, fy, player))
        out.push_back(fy * RULES_LEN + fx);
}

// Calls f(x, y) once for each point (x, y) of the view where the engine may place the player's blueprint b
// (fits_in_view), the view's origin on the rules point (ox, oy): each anchor that puts a cell of b on a point of the
// player's frontier (frontier_near).
template<class F>
inline void for_each_placement(const Position& p, int player, const Blueprint& b, int ox, int oy,
                               const std::vector<int32_t>& frontier, F&& f) {
  if(b.n == 0)
    return;
  int8_t seen[VIEW_AREA] = {};
  for(int32_t point : frontier) {
    const int fx = point % RULES_LEN, fy = point / RULES_LEN;
    for(int i = 0; i < b.n; i++) {
      const int vx = fx - b.q[i] - ox, vy = fy - b.r[i] - oy;
      if(!on_view(vx, vy) || seen[vy * VIEW + vx])
        continue;
      seen[vy * VIEW + vx] = 1;
      if(fits_in_view(p, player, b, ox, oy, vx, vy))
        f(vx, vy);
    }
  }
}

// The points of the rules board whose cells blueprint b of the player takes with its (0, 0) cell on (x, y): the
// enemy's cells next to a stamped cell, not walls, each once (the site's ce). Returns their number, at most MAX_TAKEN.
inline int taken_by(const Position& p, int player, const Blueprint& b, int x, int y, int32_t* out) {
  int n = 0;
  for(int i = 0; i < b.n; i++)
    for(const auto& d : DIRS) {
      const int nx = x + b.q[i] + d[0], ny = y + b.r[i] + d[1];
      const int8_t v = at(p, nx, ny);
      if(cell_owner(v) != 1 - player || cell_type(v) == WALL)
        continue;
      const int32_t index = ny * RULES_LEN + nx;
      if(std::find(out, out + n, index) == out + n)
        out[n++] = index;
    }
  return n;
}

// The edit's drop (the site's pe, after the pick), for a pick that took the cell (pickQ, pickR) out of blueprint k of
// the mover's blueprints hand (k's rest connected or not): whether the cell can go to blueprint j at (q, r), a point
// next to j's cells, or (0, 0) when j is empty (the caller checks this). The point must be one of the hand's points
// and not a cell of j. A drop into another blueprint needs k's rest connected. A drop within k needs a new place,
// and k connected after it.
inline bool drop_fits(const Blueprint* hand, int k, int pickQ, int pickR, bool restConnected, int j, int q, int r) {
  if(j != k && !restConnected)
    return false;
  if(!on_view(q + HAND_X, r + HAND_Y) || hand[j].find(q, r) >= 0)
    return false;
  return j != k || (!(q == pickQ && r == pickR) && (restConnected || hand[j].connected(true, q, r)));
}

// Calls f(j, q, r) for each point (q, r) where a drop into blueprint j of hand can go: (0, 0) when j is empty, else
// each point next to j's cells (some more than once). Stops, and returns true, when f returns true.
template<class F>
inline bool any_drop_point(const Blueprint* hand, F&& f) {
  for(int j = 0; j < HAND; j++) {
    const Blueprint& target = hand[j];
    if(target.n == 0) {
      if(f(j, 0, 0))
        return true;
      continue;
    }
    for(int i = 0; i < target.n; i++)
      for(const auto& d : DIRS)
        if(f(j, target.q[i] + d[0], target.r[i] + d[1]))
          return true;
  }
  return false;
}

// Whether the picked cell can go to blueprint j at (q, r): a drop after the pick (drop_fits).
inline bool can_drop(const Position& p, int j, int q, int r) {
  if(p.picked < 0)
    return false;
  const Blueprint* hand = p.hand[turn_of(p.ply)];
  if(hand[j].n == 0 ? !(q == 0 && r == 0) : !hand[j].adjacent(q, r))
    return false;
  return drop_fits(hand, p.picked, p.pickQ, p.pickR, p.pickConnected, j, q, r);
}

// The drops after the pick, in ascending order of action.
inline void drops(const Position& p, std::vector<int>& out) {
  const Blueprint* hand = p.hand[turn_of(p.ply)];
  any_drop_point(hand, [&](int j, int q, int r) {
    if(drop_fits(hand, p.picked, p.pickQ, p.pickR, p.pickConnected, j, q, r))
      out.push_back(action_at(q + HAND_X, r + HAND_Y, HAND_CHANNEL + j));
    return false;
  });
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

// Whether the pick of the cell (q, r) of the mover's blueprint k is legal: the cell exists and a drop follows. The
// drops depend on the hand alone, so a copy of the hand with the cell out stands in for the position.
inline bool can_pick(const Position& p, int k, int q, int r) {
  const int mover = turn_of(p.ply);
  if(p.picked >= 0 || p.hand[mover][k].find(q, r) < 0)
    return false;
  Blueprint hand[HAND];
  std::copy(p.hand[mover], p.hand[mover] + HAND, hand);
  hand[k].remove(hand[k].find(q, r));
  const bool restConnected = hand[k].connected();
  return any_drop_point(hand, [&](int j, int dq, int dr) {
    return drop_fits(hand, k, q, r, restConnected, j, dq, dr);
  });
}

// Moves every point of the rules board by (-dx, -dy) in the position's own records: the history and the recent
// placements' boxes. A history's point that leaves the board goes: the cells it took can lie beyond the box.
inline void shift_records(Position& p, int dx, int dy) {
  auto shift = [&](int32_t* points, int8_t& n) {
    int kept = 0;
    for(int i = 0; i < n; i++) {
      const int x = points[i] % RULES_LEN - dx, y = points[i] / RULES_LEN - dy;
      if(on_rules(x, y))
        points[kept++] = y * RULES_LEN + x;
    }
    n = int8_t(kept);
  };
  for(Footprint& f : p.history) {
    shift(f.stamped, f.nStamped);
    shift(f.taken, f.nTaken);
  }
  for(auto& boxes : p.recent)
    for(Box& b : boxes)
      if(b.any) {
        b.x0 = int16_t(b.x0 - dx);
        b.x1 = int16_t(b.x1 - dx);
        b.y0 = int16_t(b.y0 - dy);
        b.y1 = int16_t(b.y1 - dy);
      }
}

// The bounding box of the cells on the rules board (not any when the board is empty): a scan of the box that holds
// them (Cells::used).
inline Box cells_box(const Position& p) {
  Box box;
  const Box& used = p.cell.used;
  if(used.any)
    for(int y = used.y0; y <= used.y1; y++) {
      const int8_t* row = p.cell.row(y);
      for(int x = used.x0; x <= used.x1; x++)
        if(row[x])
          box.add(x, y);
    }
  return box;
}

// Moves the cells so that the box of the rules board is in its middle, and the offset with them: by half the box's
// distance from the middle (half_off_middle). The box holds every cell and is at most RULES_LEN wide, so the cells
// stay on the board. The cells' hash keys on site coordinates, so it does not change.
inline void center_box(Position& p, const Box& box) {
  const int dx = half_off_middle(box.x0, box.x1), dy = half_off_middle(box.y0, box.y1);
  if(dx == 0 && dy == 0)
    return;
  // The cells of the box, then the board cleared and each cell set at its new point.
  const int x0 = std::max(0, int(box.x0)), x1 = std::min(RULES_LEN - 1, int(box.x1));
  const int y0 = std::max(0, int(box.y0)), y1 = std::min(RULES_LEN - 1, int(box.y1));
  const int width = x1 - x0 + 1;
  static thread_local std::vector<int8_t> moved;
  moved.resize(size_t(width) * (y1 - y0 + 1));
  for(int y = y0; y <= y1; y++)
    std::memcpy(moved.data() + (y - y0) * width, p.cell.row(y) + x0, size_t(width));
  p.cell.clear();
  for(int y = y0; y <= y1; y++)
    for(int x = x0; x <= x1; x++)
      if(const int8_t v = moved[(y - y0) * width + x - x0])
        p.cell.set((y - dy) * RULES_LEN + x - dx, v);
  shift_records(p, dx, dy);
  p.offsetQ = int16_t(p.offsetQ + dx);
  p.offsetR = int16_t(p.offsetR + dy);
}

// Moves the cells so that their bounding box is in the rules board's middle (center_box): after each placement.
inline void recenter(Position& p) {
  const Box box = cells_box(p);
  if(box.any)
    center_box(p, box);
}

// Stamps blueprint b of the player with its (0, 0) cell at the rules point (x, y): the stamped cells, the enemy cells
// that go and the score (the site's ce and le). The caller checked the placement. Writes the stamped and the taken
// points to trace, and the stamped cells' box to the player's recent placements.
inline void stamp(Position& p, int player, const Blueprint& b, int x, int y, Footprint& trace) {
  const int enemy = 1 - player;
  trace.nTaken = int8_t(taken_by(p, player, b, x, y, trace.taken));
  int removedStars = 0;
  for(int i = 0; i < trace.nTaken; i++)
    removedStars += cell_type(p.cell[trace.taken[i]]) == STAR;
  p.scores[player] += removedStars * p.stars[player];
  for(int i = 0; i < trace.nTaken; i++)
    set_cell(p, trace.taken[i], 0);
  p.stars[enemy] = int16_t(p.stars[enemy] - removedStars);
  trace.nStamped = b.n;
  Box box;
  for(int i = 0; i < b.n; i++) {
    const int cx = x + b.q[i], cy = y + b.r[i];
    trace.stamped[i] = cy * RULES_LEN + cx;
    set_cell(p, trace.stamped[i], cell_value(player, b.t[i]));
    p.stars[player] = int16_t(p.stars[player] + (b.t[i] == STAR));
    box.add(cx, cy);
  }
  for(int i = RECENT - 1; i > 0; i--)
    p.recent[player][i] = p.recent[player][i - 1];
  p.recent[player][0] = box;
}

// Makes room for the newest action at the front of the history. An action that leaves no points leaves it empty.
inline void push_history(Position& p) {
  for(int h = HISTORY - 1; h > 0; h--)
    p.history[h] = p.history[h - 1];
  p.history[0] = Footprint();
}

// Plays the mover's placement of blueprint k with its (0, 0) cell at the rules point (x, y), which the caller checked,
// after push_history: the stamp, the board back to its middle (recenter), and the ply.
inline void place(Position& p, int k, int x, int y) {
  const int mover = turn_of(p.ply);
  stamp(p, mover, p.hand[mover][k], x, y, p.history[0]);
  recenter(p);
  p.ply++;
}

// Places the mover's blueprint k with its (0, 0) cell on the site's point (q, r), anywhere on the rules board: a move
// of the site that the engine's actions cannot name (a placement with a cell outside the view), as a replay of a site's
// game plays it. First the site's rule: every cell empty, and one of them next to a cell of the mover (a point off the
// board is empty, as at() reads it). Then a placement that sticks out of the board moves the cells, so that the box of
// the cells and the placement is in the middle (the board recenters after a placement, and a long blueprint can reach
// past its edge). False, and no change, when a pick waits, when the placement is not legal, or when the box of the
// cells and the placement is wider than the rules board.
inline bool place_at_site(Position& p, int k, int q, int r) {
  if(p.ply >= PLIES || p.picked >= 0 || k < 0 || k >= HAND)
    return false;
  const int mover = turn_of(p.ply);
  const Blueprint& b = p.hand[mover][k];
  int x = q - p.offsetQ + RULES_RADIUS, y = r - p.offsetR + RULES_RADIUS;
  bool touches = false, off = false;
  for(int i = 0; i < b.n; i++) {
    const int cx = x + b.q[i], cy = y + b.r[i];
    if(at(p, cx, cy))
      return false;
    touches = touches || next_to_own(p, cx, cy, mover);
    off = off || !on_rules(cx, cy);
  }
  if(!touches)
    return false;
  if(off) {
    Box box = cells_box(p);
    for(int i = 0; i < b.n; i++)
      box.add(x + b.q[i], y + b.r[i]);
    if(box.x1 - box.x0 >= RULES_LEN || box.y1 - box.y0 >= RULES_LEN)
      return false;
    center_box(p, box);
    x = q - p.offsetQ + RULES_RADIUS;
    y = r - p.offsetR + RULES_RADIUS;
  }
  push_history(p);
  place(p, k, x, y);
  return true;
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

// The start: each player's three blueprints stamped at the site's places. The start's stamps set no recent placement,
// so each player's first window is the board's middle.
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
    for(int k = 0; k < HAND; k++) {
      Footprint start;
      stamp(p, player, p.hand[player][k], starts[player][k] + RULES_RADIUS, -1 + RULES_RADIUS, start);
    }
    for(Box& b : p.recent[player])
      b = Box();
  }
}

// A number's text: an integer of at most `digits` digits (at most 9, so that it fits an int), with an optional minus.
inline bool parse_int(const std::string& s, int& out, size_t digits = 4) {
  const size_t sign = !s.empty() && s[0] == '-';
  if(s.size() <= sign || s.size() - sign > digits)
    return false;
  for(size_t i = sign; i < s.size(); i++)
    if(!std::isdigit((unsigned char)s[i]))
      return false;
  out = std::atoi(s.c_str());
  return true;
}
// A type's letter: PLAIN, STAR or WALL for p, s or w (TYPE_LETTERS), else -1.
inline int type_of(char letter) {
  const char* found = letter ? std::strchr(TYPE_LETTERS, letter) : nullptr;
  return found ? int(found - TYPE_LETTERS) : -1;
}
// A cell's text: "q,r", integers.
inline bool parse_cell(const std::string& text, int& q, int& r) {
  const size_t comma = text.find(',');
  if(comma == std::string::npos || comma == 0 || comma + 1 >= text.size())
    return false;
  return parse_int(text.substr(0, comma), q) && parse_int(text.substr(comma + 1), r);
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
  static constexpr int MAX_BOARD_LEN = VIEW;
  static constexpr int NUM_ACTIONS = VIEW_AREA * CHANNELS + 1;
  static constexpr int PASS_ACTION = VIEW_AREA * CHANNELS;

  static bool new_game(int xSize, int ySize, Position& p) {
    if(xSize != VIEW || ySize != VIEW)
      return false;
    setup(p);
    return true;
  }
  static int x_size(const Position&) { return VIEW; }
  static int y_size(const Position&) { return VIEW; }
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
    int ox, oy;
    view_origin(p, ox, oy);
    static thread_local std::vector<int32_t> frontier;
    frontier_near(p, mover, ox, oy, frontier);
    for(int k = 0; k < HAND; k++)
      for_each_placement(p, mover, p.hand[mover][k], ox, oy, frontier,
                         [&](int x, int y) { out.push_back(action_at(x, y, PLACE + k)); });
    // The picks.
    for(int k = 0; k < HAND; k++) {
      const Blueprint& b = p.hand[mover][k];
      for(int i = 0; i < b.n; i++)
        if(can_pick(p, k, b.q[i], b.r[i]))
          out.push_back(action_at(b.q[i] + HAND_X, b.r[i] + HAND_Y, HAND_CHANNEL + k));
    }
    std::sort(out.begin(), out.end());
  }
  static bool is_legal(const Position& p, int action) {
    if(action < 0 || action >= PASS_ACTION || terminal(p))
      return false;
    const int x = action_x(action), y = action_y(action), channel = action_channel(action);
    if(channel >= HAND_CHANNEL)
      return p.picked >= 0 ? can_drop(p, channel - HAND_CHANNEL, x - HAND_X, y - HAND_Y)
                           : can_pick(p, channel - HAND_CHANNEL, x - HAND_X, y - HAND_Y);
    if(p.picked >= 0)
      return false;
    int ox, oy;
    view_origin(p, ox, oy);
    const int mover = turn_of(p.ply);
    return fits_in_view(p, mover, p.hand[mover][channel - PLACE], ox, oy, x, y);
  }
  static void play(Position& p, int action) {
    const int mover = turn_of(p.ply);
    push_history(p);  // only a placement leaves points
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
    if(channel >= HAND_CHANNEL && p.picked >= 0) {  // the drop
      const int j = channel - HAND_CHANNEL;
      p.hand[mover][j].add(x - HAND_X, y - HAND_Y, p.pickType);
      p.hand[mover][j].normalize();
      p.hand[mover][p.picked].normalize();
      p.picked = -1;
      p.ply++;
    }
    else if(channel >= HAND_CHANNEL)  // the pick
      take(p, channel - HAND_CHANNEL, x - HAND_X, y - HAND_Y);
    else {
      int ox, oy;
      view_origin(p, ox, oy);
      place(p, channel - PLACE, x + ox, y + oy);
    }
  }
  static bool replay_tolerates(const Position&, int action) { return action == PASS_ACTION; }

  // The higher score wins. Equal scores: a draw.
  static float result(const Position& p) {
    return p.scores[0] == p.scores[1] ? 0.0f : p.scores[0] > p.scores[1] ? 1.0f : -1.0f;
  }
  static double score(const Position& p, int player) { return p.scores[player]; }

  // The cells (cellHash, by site coordinates), each blueprint's cells, the ply, the pick, the scores, the recent
  // placements' boxes by site coordinates and the offset: a hash of the whole position (each kind with its own keys,
  // KEY_HAND and the others). The boxes and the offset set the window, so the legal moves: the window's middle rounds
  // toward the rules board's middle, so with the same boxes two offsets can give two windows (half_off_middle).
  static Hash128 hash(const Position& p) {
    auto cell_number = [](int q, int r) { return uint64_t((q + 32) * 64 + r + 32); };
    Hash128 h = p.cellHash ^ key(1, 0);
    for(int player = 0; player < 2; player++)
      for(int k = 0; k < HAND; k++) {
        const Blueprint& b = p.hand[player][k];
        for(int i = 0; i < b.n; i++)
          h ^= key(KEY_HAND + uint64_t(player * HAND + k) * 4096 + cell_number(b.q[i], b.r[i]), uint64_t(b.t[i]));
      }
    h ^= key(KEY_PLY + uint64_t(p.ply), 0);
    if(p.picked >= 0)
      h ^= key(KEY_PICK + uint64_t(p.picked) * 4096 + cell_number(p.pickQ, p.pickR), uint64_t(p.pickType));
    for(int player = 0; player < 2; player++)
      h ^= key(KEY_SCORE + uint64_t(player), uint64_t(uint32_t(p.scores[player])));
    const int dq = p.offsetQ - RULES_RADIUS, dr = p.offsetR - RULES_RADIUS;
    for(int player = 0; player < 2; player++)
      for(int i = 0; i < RECENT; i++) {
        const Box& b = p.recent[player][i];
        if(b.any)
          h ^= key(KEY_BOX + uint64_t(player * RECENT + i),
                   site_key(b.x0 + dq, b.y0 + dr) << 32 | site_key(b.x1 + dq, b.y1 + dr));
      }
    h ^= key(KEY_OFFSET, site_key(p.offsetQ, p.offsetR));
    return h;
  }

  // The view's points: the mover's window.
  static void pieces(const Position& p, int8_t* owners) {
    int ox, oy;
    view_origin(p, ox, oy);
    for(int y = 0; y < VIEW; y++)
      for(int x = 0; x < VIEW; x++)
        owners[y * VIEW + x] = int8_t(cell_owner(at(p, x + ox, y + oy)));
  }
  static void area(const Position& p, int8_t* owners) { pieces(p, owners); }

  // The strings: "K@q,r" (a placement, its cell from the window's middle) and "K:q,r" (a cell of blueprint K: a pick,
  // or a drop after a pick), with blueprints from 1. The site's string of an edit joins its pick's and its drop's with
  // ">": "I:q,r>J:q,r". A string of a cell may start with ">", as the drop's part of the site's string does.
  static std::string action_text(int action) {
    if(action == PASS_ACTION)
      return "pass";
    const int x = action_x(action), y = action_y(action), channel = action_channel(action);
    if(channel >= HAND_CHANNEL)
      return std::to_string(channel - HAND_CHANNEL + 1) + ":" + cell_text(x - HAND_X, y - HAND_Y);
    return std::to_string(channel - PLACE + 1) + "@" + cell_text(x - VIEW_RADIUS, y - VIEW_RADIUS);
  }
  static std::string protocol_text(int action) { return action_text(action); }
  // The action that a placement's or a cell's string names, legal or not. False for any other text, and for a point
  // off the view or the hand's points.
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
    // The hand is drawn at the view's middle column, so a hand's point and a view's point add the same x.
    static_assert(HAND_X == VIEW_RADIUS, "the hand's x is the view's middle column");
    const int x = q + VIEW_RADIUS, y = r + (place ? VIEW_RADIUS : HAND_Y);
    if(!on_view(x, y))
      return false;
    action = action_at(x, y, (place ? PLACE : HAND_CHANNEL) + k);
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
      return fail("not a move: \"" + text + "\" (K@q,r places blueprint K from the window's middle, K:q,r picks a " +
                  "cell or drops the picked one; the view is |q|, |r| <= " + std::to_string(VIEW_RADIUS) + ")");
    if(terminal(p))
      return fail("the game is over");
    if(!is_legal(p, action))
      return fail(text + " is not legal here");
    return action;
  }

  // The one-line position, nine fields: the ply, the two scores, the board ("q,r,T" from the rules board's middle,
  // with T of psw for the first player and PSW for the second, joined by ;), the first player's three blueprints and
  // the second's ("q,r,t" joined by ;, the blueprints joined by |), the pick ("I:q,r,t" of the mover, its cell out of
  // blueprint I), the site's coordinates of the rules board's middle ("q,r"), and each player's last placements'
  // boxes ("q0,q1,r0,r1" from the rules board's middle, the newest first, joined by ;, the players joined by |). "-"
  // is empty.
  static std::string position_text(const Position& p) {
    std::string s = std::to_string(p.ply) + " " + std::to_string(p.scores[0]) + " " + std::to_string(p.scores[1]) + " ";
    std::string board;
    for(int y = 0; y < RULES_LEN; y++)
      for(int x = 0; x < RULES_LEN; x++) {
        const int8_t v = p.cell[y * RULES_LEN + x];
        if(!v)
          continue;
        if(!board.empty())
          board += ';';
        const char letter = TYPE_LETTERS[cell_type(v)];
        board += cell_text(x - RULES_RADIUS, y - RULES_RADIUS) + "," +
                 char(cell_owner(v) ? std::toupper(letter) : letter);
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
    s += " " + cell_text(p.offsetQ, p.offsetR) + " ";
    for(int player = 0; player < 2; player++) {
      if(player > 0)
        s += '|';
      std::string boxes;
      for(const Box& b : p.recent[player])
        if(b.any)
          boxes += (boxes.empty() ? "" : ";") + std::to_string(b.x0 - RULES_RADIUS) + "," +
                   std::to_string(b.x1 - RULES_RADIUS) + "," + std::to_string(b.y0 - RULES_RADIUS) + "," +
                   std::to_string(b.y1 - RULES_RADIUS);
      s += boxes.empty() ? "-" : boxes;
    }
    return s;
  }
  // Checks the fields, the board's points, each blueprint's frame and connection, and the cells of each player's hand.
  // Then the board goes to its middle (recenter), as play() keeps it.
  static bool parse_position(const std::string& text, Position& p, std::string& why) {
    const std::vector<std::string> fields = split(text, ' ');
    if(fields.size() != 9) {
      why = "a position is nine fields: the ply, the two scores, the board, the two hands, the pick, the offset and "
            "the recent placements";
      return false;
    }
    static thread_local Position q;
    q = Position();
    int ply, s0, s1;
    if(!parse_int(fields[0], ply) || ply < 0 || ply > PLIES || !parse_int(fields[1], s0, 9) ||
       !parse_int(fields[2], s1, 9)) {
      why = "the ply is 0 to " + std::to_string(PLIES) + ", and the scores are integers of at most 9 digits";
      return false;
    }
    q.ply = int16_t(ply);
    q.scores[0] = s0;
    q.scores[1] = s1;
    int oq, orr;
    if(!parse_cell(fields[7], oq, orr)) {
      why = "the offset is q,r: the site's coordinates of the rules board's middle";
      return false;
    }
    q.offsetQ = int16_t(oq);
    q.offsetR = int16_t(orr);
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
        const int type = cell3(c, cq, cr, letter) ? type_of(char(std::tolower((unsigned char)letter))) : -1;
        if(type < 0) {
          why = "a cell of the board is q,r,T with T one of psw (the first player) or PSW (the second): \"" + c + "\"";
          return false;
        }
        const int x = cq + RULES_RADIUS, y = cr + RULES_RADIUS;
        if(!on_rules(x, y) || q.cell[y * RULES_LEN + x]) {
          why = "the cell " + cell_text(cq, cr) + " is off the board or given twice";
          return false;
        }
        const int owner = std::isupper((unsigned char)letter) ? 1 : 0;
        set_cell(q, y * RULES_LEN + x, cell_value(owner, type));
        q.stars[owner] = int16_t(q.stars[owner] + (type == STAR));
      }
    const int mover = turn_of(q.ply);
    if(fields[6] != "-") {
      int cq, cr;
      char letter;
      const std::string& f = fields[6];
      const bool form =
          f.size() >= 7 && f[1] == ':' && f[0] >= '1' && f[0] <= '0' + HAND && cell3(f.substr(2), cq, cr, letter);
      const int type = form ? type_of(letter) : -1;
      if(type < 0 || !on_view(cq + HAND_X, cr + HAND_Y) || q.ply >= PLIES) {
        why = "the pick is I:q,r,t (blueprint I from 1, t one of psw, the cell on the hand's points), before the end";
        return false;
      }
      q.picked = int8_t(f[0] - '1');
      q.pickQ = int8_t(cq);
      q.pickR = int8_t(cr);
      q.pickType = int8_t(type);
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
            const int type = cell3(c, cq, cr, letter) ? type_of(letter) : -1;
            if(type < 0 || b.n == MAX_CELLS || b.find(cq, cr) >= 0 || !on_view(cq + HAND_X, cr + HAND_Y)) {
              why = "a blueprint's cell is q,r,t with t one of psw, each once, on the hand's points: \"" + c + "\"";
              return false;
            }
            b.add(cq, cr, type);
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
      // The rest keeps the whole blueprint's frame until the drop (take): with the picked cell, it is the blueprint.
      Blueprint whole = q.hand[mover][q.picked];
      if(whole.find(q.pickQ, q.pickR) >= 0) {
        why = "the picked cell is still in its blueprint";
        return false;
      }
      whole.add(q.pickQ, q.pickR, q.pickType);
      if(!whole.framed() || !whole.connected()) {
        why = "the picked cell and the rest of its blueprint are not connected or not in the site's frame";
        return false;
      }
      q.pickConnected = q.hand[mover][q.picked].connected();
    }
    const std::vector<std::string> players = split(fields[8], '|');
    if(players.size() != 2) {
      why = "the recent placements are the two players' boxes, joined by |";
      return false;
    }
    for(int player = 0; player < 2; player++) {
      if(players[player] == "-")
        continue;
      const std::vector<std::string> boxes = split(players[player], ';');
      if(boxes.size() > RECENT) {
        why = "a player has at most " + std::to_string(RECENT) + " recent placements";
        return false;
      }
      for(size_t i = 0; i < boxes.size(); i++) {
        const std::vector<std::string> v = split(boxes[i], ',');
        int c[4];
        if(v.size() != 4 || !parse_int(v[0], c[0]) || !parse_int(v[1], c[1]) || !parse_int(v[2], c[2]) ||
           !parse_int(v[3], c[3]) || c[0] > c[1] || c[2] > c[3] ||
           !on_rules(c[0] + RULES_RADIUS, c[2] + RULES_RADIUS) || !on_rules(c[1] + RULES_RADIUS, c[3] + RULES_RADIUS)) {
          why = "a recent placement's box is q0,q1,r0,r1 on the rules board: \"" + boxes[i] + "\"";
          return false;
        }
        Box& b = q.recent[player][i];
        b.any = true;
        b.x0 = int16_t(c[0] + RULES_RADIUS);
        b.x1 = int16_t(c[1] + RULES_RADIUS);
        b.y0 = int16_t(c[2] + RULES_RADIUS);
        b.y1 = int16_t(c[3] + RULES_RADIUS);
      }
    }
    recenter(q);
    p = q;
    return true;
  }
};

static_assert(GameInterface::checkRules<GameRules>(), "Blueprints' rules part");

}  // namespace blueprints
