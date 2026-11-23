#pragma once
// Tak's rules part of the game interface (game/gameinterface.h): playtak.com's rules on square boards of 3 x 3 to
// 8 x 8 (the engine repository's games/tak/docs/design/02-rules.md, and its research, rules-theory.md, sections 1 and
// 2.1). Tak is James Ernest's and Patrick Rothfuss's game: this code plays its rules for AI and analysis.
//
// - The pieces a player: 3x3 10 stones, 4x4 15, 5x5 21 and 1 capstone, 6x6 30 and 1, 7x7 40 and 2, 8x8 50 and 2.
// - A turn places a flat, a wall or a capstone on an empty square, or spreads a stack that the player tops. No pass.
// - The opening (the setting opening, SETTINGS): plies 1 and 2 each place one flat of the opponent's colour (swap).
//   Under "double black stack" (playtak's Opening.java, commit 5384347b), ply 1 places two of Black's flats as one
//   stack, both from Black's reserve.
// - A spread carries at most the board's side. A capstone that moves alone as the last step of a spread flattens a
//   wall of either colour. Nothing else covers a wall or a capstone.
// - A road (flats and capstones on top that join opposite edges) wins, also a road for the opponent only. A double
//   road wins for the mover. Then a full board, or a player with no stones and no capstones left, ends the game by
//   the flat count: only flats on top count. Then playtak's 50-ply rule: a draw after 50 plies with no placement and
//   no flattening (Game.java, checkStaleGame).
//
// The players: Tak's White moves first, so White is player 0 (KataGo's Black in the code's names) and Black is player
// 1. Komi is the front's (KataGo's rules.komi, in flats, added to the second player's score as playtak adds it to
// Black's): score(p, player) is the player's flats on top, and result(p) is the result without komi. result_with_komi
// gives the result under a komi in half-flats, as playtak and TEI give it.
//
// The notation is Tak's own: moves in PTN (c3, Sb3, Cc4, 3c3>12), positions in TPS with one more field, the plies
// since the last placement or flattening (TPS does not hold them). The reserves come from the board.
//
// An action is one whole move, numbered on the largest board (MAX_LEN): (y * MAX_LEN + x) * CHANNELS + channel
// (games/tak/docs/design/03-moves-and-policy.md). Channels 0, 1 and 2 place a flat, a wall and a capstone on the
// square (channel 0 also places the opponent's flat in the opening). Channel 3 + 4 * pattern + direction spreads the
// square's stack: the directions are PTN's +, -, > and < (0 to 3), and a pattern of a pickup of k pieces is
// 2^(k-1) - 1 + code, where bit i - 1 of the (k-1)-bit code is 1 when a step ends after the i-th piece dropped (the
// first piece dropped is the lowest piece carried). A flattening is a spread whose last step is the capstone alone.
//
// TAK_MAX_BOARD_LEN sets the largest board, 8 by default: the rules module plays every size. A build whose network's
// largest board is smaller sets it (the first network's is 6: 255 channels and 9,181 actions).

#include <array>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

#include "../../core/hash.h"
#include "../gameinterface.h"

#ifndef TAK_MAX_BOARD_LEN
#define TAK_MAX_BOARD_LEN 8
#endif

namespace tak {

constexpr int MIN_LEN = 3;
constexpr int MAX_LEN = TAK_MAX_BOARD_LEN;
static_assert(MAX_LEN >= MIN_LEN && MAX_LEN <= 8, "Tak's boards are 3 x 3 to 8 x 8");
constexpr int STANDARD_LEN = MAX_LEN < 6 ? MAX_LEN : 6;  // the tournaments' board
constexpr int NUM_POINTS = MAX_LEN * MAX_LEN;
constexpr int NUM_PATTERNS = (1 << MAX_LEN) - 1;  // the drop patterns of a pickup of 1 to MAX_LEN pieces
constexpr int CHANNELS = 3 + 4 * NUM_PATTERNS;
constexpr int PASS = NUM_POINTS * CHANNELS;
constexpr int FIFTY_PLIES = 50;  // playtak's draw: plies with no placement and no flattening

// The pieces a player on each board side (stones, then capstones).
constexpr int STONES[9] = {0, 0, 0, 10, 15, 21, 30, 40, 50};
constexpr int CAPSTONES[9] = {0, 0, 0, 0, 0, 1, 1, 2, 2};

enum Kind { FLAT = 0, WALL = 1, CAP = 2 };  // also the placements' channels
enum End : int8_t { ONGOING = 0, ROAD, FLATS, FIFTY };
enum Opening { SWAP = 0, DOUBLE_BLACK_STACK = 1 };

constexpr int DX[4] = {0, 0, 1, -1};
constexpr int DY[4] = {1, -1, 0, 0};
constexpr const char* DIRECTIONS = "+-><";
constexpr const char* OPENING_NAMES[2] = {"swap", "double black stack"};  // PTN's Opening tag, as Opening.java

// The setting opening, for the whole process (GameRules::SETTINGS). The rules read it on ply 1.
inline int opening = SWAP;

inline int point(int x, int y) { return y * MAX_LEN + x; }
inline uint64_t bit(int i) { return uint64_t(1) << i; }
inline int popcount(uint64_t b) { return __builtin_popcountll(b); }

inline int action_point(int action) { return action / CHANNELS; }
inline int action_channel(int action) { return action % CHANNELS; }
inline int place_action(int i, int kind) { return i * CHANNELS + kind; }
inline int spread_action(int i, int direction, int pattern) { return i * CHANNELS + 3 + 4 * pattern + direction; }

// A pattern's pickup: k for patterns 2^(k-1) - 1 to 2^k - 2.
inline int pattern_pieces(int pattern) { return 64 - __builtin_clzll(uint64_t(pattern) + 1); }
inline int pattern_code(int pattern) { return pattern + 1 - (1 << (pattern_pieces(pattern) - 1)); }
inline int pattern_steps(int pattern) { return popcount(uint64_t(pattern_code(pattern))) + 1; }
// Whether the last step drops one piece: the capstone alone, as a flattening needs.
inline bool pattern_last_one(int pattern) {
  const int k = pattern_pieces(pattern);
  return k == 1 || ((pattern_code(pattern) >> (k - 2)) & 1);
}
// The pieces that each step drops, in order.
inline std::vector<int> pattern_drops(int pattern) {
  const int k = pattern_pieces(pattern), code = pattern_code(pattern);
  std::vector<int> drops(1, 0);
  for(int i = 1; i <= k; i++) {
    drops.back()++;
    if(i < k && ((code >> (i - 1)) & 1))
      drops.push_back(0);
  }
  return drops;
}
// The pattern of drops that sum to k (1 <= k <= MAX_LEN, each drop at least 1).
inline int drops_pattern(const std::vector<int>& drops) {
  int k = 0, code = 0;
  for(size_t s = 0; s < drops.size(); s++) {
    k += drops[s];
    if(s + 1 < drops.size())
      code |= 1 << (k - 1);
  }
  return (1 << (k - 1)) - 1 + code;
}

// splitmix64's finalizer, a bijection of 64-bit words (as Connect-4's rules part).
inline uint64_t mix64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

struct Position {
  // Each square's stack, its top piece at bit 0: bit h of top[i], or of rest[i] for h >= 64, is the colour of the
  // piece h below the top (0 White, 1 Black). A stack of 8 x 8 can hold 104 pieces.
  uint64_t top[NUM_POINTS] = {};
  uint64_t rest[NUM_POINTS] = {};
  uint8_t height[NUM_POINTS] = {};
  // The squares that each player tops, and the walls and capstones on top: bit point(x, y).
  uint64_t owner[2] = {0, 0};
  uint64_t walls = 0, caps = 0;
  int8_t size = STANDARD_LEN;
  uint8_t stones[2] = {STONES[STANDARD_LEN], STONES[STANDARD_LEN]};
  uint8_t capstones[2] = {CAPSTONES[STANDARD_LEN], CAPSTONES[STANDARD_LEN]};
  int32_t ply = 0;
  int32_t drawClock = 0;  // the plies since the last placement or flattening
  int8_t end = ONGOING;
  int8_t roadWinner = -1;  // the player whose road ended the game
};

inline uint64_t board_mask(int size) {
  uint64_t m = 0;
  for(int y = 0; y < size; y++)
    m |= ((uint64_t(1) << size) - 1) << (y * MAX_LEN);
  return m;
}
inline uint64_t column_mask(int x) {
  uint64_t m = 0;
  for(int y = 0; y < MAX_LEN; y++)
    m |= bit(point(x, y));
  return m;
}
inline uint64_t occupied(const Position& p) { return p.owner[0] | p.owner[1]; }
inline int kind_at(const Position& p, int i) { return (p.walls & bit(i)) ? WALL : (p.caps & bit(i)) ? CAP : FLAT; }
inline uint64_t flats(const Position& p, int player) { return p.owner[player] & ~p.walls & ~p.caps; }
inline int colour_at(const Position& p, int i, int depth) {  // depth below the top
  return int(((depth < 64 ? p.top[i] >> depth : p.rest[i] >> (depth - 64))) & 1);
}

// Takes the k (1 to 8) top pieces of a stack: their colours, the top one at bit 0.
inline uint64_t take(Position& p, int i, int k) {
  const uint64_t pieces = p.top[i] & ((uint64_t(1) << k) - 1);
  p.top[i] = (p.top[i] >> k) | (p.rest[i] << (64 - k));
  p.rest[i] >>= k;
  p.height[i] = uint8_t(p.height[i] - k);
  return pieces;
}
// Puts c (1 to 8) pieces on a stack, their top one at bit 0.
inline void put(Position& p, int i, uint64_t pieces, int c) {
  p.rest[i] = (p.rest[i] << c) | (p.top[i] >> (64 - c));
  p.top[i] = (p.top[i] << c) | pieces;
  p.height[i] = uint8_t(p.height[i] + c);
}
// Sets the square's top piece: its owner, and whether it is a wall or a capstone.
inline void set_top(Position& p, int i, int kind) {
  const uint64_t b = bit(i);
  p.owner[0] &= ~b;
  p.owner[1] &= ~b;
  p.walls &= ~b;
  p.caps &= ~b;
  if(p.height[i] == 0)
    return;
  p.owner[p.top[i] & 1] |= b;
  if(kind == WALL)
    p.walls |= b;
  else if(kind == CAP)
    p.caps |= b;
}

// Whether the player's flats and capstones on top join opposite edges.
inline bool has_road(const Position& p, int player) {
  const uint64_t board = board_mask(p.size), road = p.owner[player] & ~p.walls;
  const uint64_t west = column_mask(0), east = column_mask(MAX_LEN - 1);
  const uint64_t notWest = ~west, notEast = ~east;
  const uint64_t row = (uint64_t(1) << p.size) - 1;  // row 1's squares
  const uint64_t starts[2] = {road & column_mask(0), road & row};
  const uint64_t goals[2] = {column_mask(p.size - 1) & board, row << ((p.size - 1) * MAX_LEN)};
  for(int axis = 0; axis < 2; axis++) {
    uint64_t group = starts[axis], last = 0;
    while(group != last) {
      last = group;
      group |= (((group & notEast) << 1) | ((group & notWest) >> 1) | (group << MAX_LEN) | (group >> MAX_LEN)) & road;
    }
    if(group & goals[axis])
      return true;
  }
  return false;
}

// Sets the end of the game after a move, or in a parsed position: a road first (both: the player who moved last
// wins), then a full board or a player with no pieces left (the flat count), then the 50-ply rule.
inline void update_end(Position& p) {
  const bool roads[2] = {has_road(p, 0), has_road(p, 1)};
  p.end = ONGOING;
  p.roadWinner = -1;
  if(roads[0] || roads[1]) {
    p.end = ROAD;
    p.roadWinner = int8_t(roads[0] && roads[1] ? (p.ply + 1) % 2 : roads[0] ? 0 : 1);
  } else if(occupied(p) == board_mask(p.size) || p.stones[0] + p.capstones[0] == 0 ||
            p.stones[1] + p.capstones[1] == 0) {
    p.end = FLATS;
  } else if(p.drawClock >= FIFTY_PLIES) {
    p.end = FIFTY;
  }
}

// White's return (1, 0 or -1) at the end, with a komi of halfKomi half-flats added to Black's flat count.
inline int result_with_komi(const Position& p, int halfKomi) {
  if(p.end == ROAD)
    return p.roadWinner == 0 ? 1 : -1;
  if(p.end != FLATS)
    return 0;
  const int margin = 2 * (popcount(flats(p, 0)) - popcount(flats(p, 1))) - halfKomi;
  return margin > 0 ? 1 : margin < 0 ? -1 : 0;
}

// The open run of a spread from square i in the direction: the squares that the spread can drop on (empty or topped
// by a flat) before the edge, a wall or a capstone, and whether a wall ends it.
inline int open_run(const Position& p, int i, int direction, bool& wallAfter) {
  int x = i % MAX_LEN + DX[direction], y = i / MAX_LEN + DY[direction], run = 0;
  wallAfter = false;
  while(x >= 0 && x < p.size && y >= 0 && y < p.size) {
    const int j = point(x, y);
    if(p.walls & bit(j)) {
      wallAfter = true;
      break;
    }
    if(p.caps & bit(j))
      break;
    run++;
    x += DX[direction];
    y += DY[direction];
  }
  return run;
}

// Whether a spread of the pattern fits the run: each step on the open run, or the last step a lone capstone onto the
// wall that ends it.
inline bool spread_fits(int pattern, int run, bool wallAfter, bool capstone) {
  const int steps = pattern_steps(pattern);
  return steps <= run || (capstone && wallAfter && steps == run + 1 && pattern_last_one(pattern));
}

inline std::string square_name(int i) {
  return std::string(1, char('a' + i % MAX_LEN)) + std::to_string(i / MAX_LEN + 1);
}

// Reads a move in PTN, case-insensitively, with no position: its action on the largest board, or false and the
// reason. PTN's marks may end it: a flattening's *, and ' " ! ? (tak and the annotations), which change nothing.
// flattens tells whether the text has the *.
inline bool parse_ptn(const std::string& text, int& action, bool& flattens, std::string& why) {
  std::string s = text;
  while(!s.empty() && std::isspace((unsigned char)s.back()))
    s.pop_back();
  size_t start = 0;
  while(start < s.size() && std::isspace((unsigned char)s[start]))
    start++;
  s = s.substr(start);
  while(!s.empty() && (s.back() == '\'' || s.back() == '"' || s.back() == '!' || s.back() == '?'))
    s.pop_back();
  flattens = !s.empty() && s.back() == '*';
  if(flattens)
    s.pop_back();
  auto fail = [&why, &text](const std::string& reason) {
    why = "not a move: \"" + text + "\" (" + reason + ")";
    return false;
  };
  size_t at = 0;
  int count = 0, kind = -1;
  if(at < s.size() && std::isdigit((unsigned char)s[at])) {
    count = s[at++] - '0';
    if(count == 0)
      return fail("a spread carries at least one piece");
  } else if(s.size() >= 2 && std::isalpha((unsigned char)s[1])) {
    const char k = char(std::toupper((unsigned char)s[0]));
    if(k != 'F' && k != 'S' && k != 'C')
      return fail("a placement's stone is F, S or C");
    kind = k == 'F' ? FLAT : k == 'S' ? WALL : CAP;
    at = 1;
  }
  if(at + 2 > s.size())
    return fail("a square is a file letter and a rank, as c3");
  const int x = std::tolower((unsigned char)s[at]) - 'a', rank = s[at + 1] - '0';
  if(x < 0 || x >= MAX_LEN || rank < 1 || rank > MAX_LEN)
    return fail("no square " + s.substr(at, 2) + " on a board of up to " + std::to_string(MAX_LEN) + " x " +
                std::to_string(MAX_LEN));
  const int i = point(x, rank - 1);
  at += 2;
  if(at == s.size()) {
    if(count > 0)
      return fail("a spread has a direction: +, -, > or <");
    if(flattens)
      return fail("only a spread flattens a wall");
    action = place_action(i, kind < 0 ? FLAT : kind);
    return true;
  }
  if(kind >= 0)
    return fail("a placement ends with its square");
  const char* d = std::char_traits<char>::find(DIRECTIONS, 4, s[at]);
  if(!d)
    return fail("a spread's direction is +, -, > or <");
  const int direction = int(d - DIRECTIONS);
  at++;
  if(count == 0)
    count = 1;
  if(count > MAX_LEN)
    return fail("a spread carries at most " + std::to_string(MAX_LEN) + " pieces");
  std::vector<int> drops;
  for(; at < s.size(); at++) {
    if(!std::isdigit((unsigned char)s[at]) || s[at] == '0')
      return fail("each drop is a count of 1 to 8");
    drops.push_back(s[at] - '0');
  }
  if(drops.empty())
    drops.push_back(count);
  int sum = 0;
  for(int c : drops)
    sum += c;
  if(sum != count)
    return fail("the drops sum to " + std::to_string(sum) + ", not the " + std::to_string(count) + " carried");
  action = spread_action(i, direction, drops_pattern(drops));
  return true;
}

struct GameRules {
  static constexpr const char* NAME = "tak";
  using Position = tak::Position;
  static constexpr int MAX_BOARD_LEN = MAX_LEN;
  // Each square's channels (CHANNELS), then pass, which is never legal.
  static constexpr int NUM_ACTIONS = PASS + 1;
  static constexpr int PASS_ACTION = PASS;
  static constexpr int STANDARD_X_SIZE = STANDARD_LEN, STANDARD_Y_SIZE = STANDARD_LEN;

  static bool new_game(int xSize, int ySize, Position& p) {
    if(xSize != ySize || xSize < MIN_LEN || xSize > MAX_LEN)
      return false;
    p = Position();
    p.size = int8_t(xSize);
    p.stones[0] = p.stones[1] = uint8_t(STONES[xSize]);
    p.capstones[0] = p.capstones[1] = uint8_t(CAPSTONES[xSize]);
    return true;
  }
  static int x_size(const Position& p) { return p.size; }
  static int y_size(const Position& p) { return p.size; }
  static int next_player(const Position& p) { return p.ply % 2; }
  static int turn_moves_left(const Position&) { return 1; }
  static int moves_played(const Position& p) { return p.ply; }
  static bool terminal(const Position& p) { return p.end != ONGOING; }

  static void legal_actions(const Position& p, std::vector<int>& out) {
    out.clear();
    if(terminal(p))
      return;
    const int me = next_player(p);
    const uint64_t empty = board_mask(p.size) & ~occupied(p);
    for(int i = 0; i < NUM_POINTS; i++) {
      if(empty & bit(i)) {
        if(p.ply < 2) {
          out.push_back(place_action(i, FLAT));
          continue;
        }
        if(p.stones[me] > 0) {
          out.push_back(place_action(i, FLAT));
          out.push_back(place_action(i, WALL));
        }
        if(p.capstones[me] > 0)
          out.push_back(place_action(i, CAP));
        continue;
      }
      if(p.ply < 2 || !(p.owner[me] & bit(i)))
        continue;
      const int carry = p.height[i] < p.size ? p.height[i] : p.size;
      const bool capstone = (p.caps & bit(i)) != 0;
      int run[4];
      bool wallAfter[4];
      for(int d = 0; d < 4; d++)
        run[d] = open_run(p, i, d, wallAfter[d]);
      for(int pattern = 0; pattern < (1 << carry) - 1; pattern++)
        for(int d = 0; d < 4; d++)
          if(spread_fits(pattern, run[d], wallAfter[d], capstone))
            out.push_back(spread_action(i, d, pattern));
    }
  }

  static bool is_legal(const Position& p, int action) {
    if(action < 0 || action >= PASS || terminal(p))
      return false;
    const int i = action_point(action), channel = action_channel(action);
    if(i % MAX_LEN >= p.size || i / MAX_LEN >= p.size)
      return false;
    const int me = next_player(p);
    if(channel < 3) {
      if(occupied(p) & bit(i))
        return false;
      if(p.ply < 2)
        return channel == FLAT;
      return channel == CAP ? p.capstones[me] > 0 : p.stones[me] > 0;
    }
    if(p.ply < 2 || !(p.owner[me] & bit(i)))
      return false;
    const int pattern = (channel - 3) / 4, direction = (channel - 3) % 4;
    const int k = pattern_pieces(pattern);
    if(k > p.height[i] || k > p.size)
      return false;
    bool wallAfter;
    const int run = open_run(p, i, direction, wallAfter);
    return spread_fits(pattern, run, wallAfter, (p.caps & bit(i)) != 0);
  }

  static void play(Position& p, int action) {
    const int me = next_player(p);
    const int i = action_point(action), channel = action_channel(action);
    if(channel < 3) {
      const int colour = p.ply < 2 ? 1 - me : me;
      if(channel == CAP)
        p.capstones[colour]--;
      else
        p.stones[colour]--;
      put(p, i, uint64_t(colour), 1);
      if(p.ply == 0 && opening == DOUBLE_BLACK_STACK) {  // White's first placement: two of Black's flats
        p.stones[colour]--;
        put(p, i, uint64_t(colour), 1);
      }
      set_top(p, i, channel);
      p.drawClock = 0;
    } else {
      const int pattern = (channel - 3) / 4, direction = (channel - 3) % 4;
      const int k = pattern_pieces(pattern), code = pattern_code(pattern);
      const int kind = kind_at(p, i);
      uint64_t carried = take(p, i, k);
      set_top(p, i, FLAT);
      int x = i % MAX_LEN, y = i / MAX_LEN, left = k, piece = 0;
      bool flattened = false;
      while(left > 0) {
        int c = 0;
        do {
          c++;
          piece++;
        } while(piece < k && !((code >> (piece - 1)) & 1));
        x += DX[direction];
        y += DY[direction];
        const int j = point(x, y);
        flattened = flattened || (p.walls & bit(j)) != 0;
        put(p, j, carried >> (left - c), c);
        left -= c;
        carried &= (uint64_t(1) << left) - 1;
        set_top(p, j, left == 0 ? kind : FLAT);
      }
      p.drawClock = flattened ? 0 : p.drawClock + 1;
    }
    p.ply++;
    update_end(p);
  }

  // The result without komi (the front adds its komi to the score): White's return.
  static float result(const Position& p) { return float(result_with_komi(p, 0)); }
  // The player's flats on top, so that KataGo's margin, score(p, 1) - score(p, 0) + komi, is Black's flat count
  // minus White's, plus komi.
  static double score(const Position& p, int player) { return popcount(flats(p, player)); }

  // Each occupied square's stack (its pieces' colours, its height and its top's kind), mixed with its point, and the
  // board's side, the player to move, the opening's plies and the plies since the last placement or flattening. The
  // move number changes no rule after the opening, so it is left out: positions that differ only in it are the same
  // state. 128 bits from two independent mixes.
  static Hash128 hash(const Position& p) {
    uint64_t h1 = mix64(0x9E3779B97F4A7C15ULL ^ (uint64_t(p.size) | uint64_t(p.ply % 2) << 4 |
                                                  uint64_t(p.ply < 2 ? p.ply : 2) << 5 | uint64_t(p.drawClock) << 8));
    uint64_t h2 = mix64(h1 ^ 0xD1B54A32D192ED03ULL);
    uint64_t on = occupied(p);
    while(on) {
      const int i = __builtin_ctzll(on);
      on &= on - 1;
      const uint64_t key = uint64_t(i) | uint64_t(kind_at(p, i)) << 8 | uint64_t(p.height[i]) << 16;
      h1 ^= mix64(mix64(mix64(0x243F6A8885A308D3ULL ^ key) ^ p.top[i]) ^ p.rest[i]);
      h2 ^= mix64(mix64(mix64(0x13198A2E03707344ULL ^ key) ^ p.top[i]) ^ p.rest[i]);
    }
    return Hash128(h1, h2);
  }

  // The player whose piece tops the point.
  static void pieces(const Position& p, int8_t* owners) {
    for(int y = 0; y < p.size; y++)
      for(int x = 0; x < p.size; x++) {
        const uint64_t b = bit(point(x, y));
        owners[y * p.size + x] = int8_t((p.owner[0] & b) ? 0 : (p.owner[1] & b) ? 1 : -1);
      }
  }
  // The player whose flat tops the point: the flat count's squares, which sum to the flat margin.
  static void area(const Position& p, int8_t* owners) {
    for(int y = 0; y < p.size; y++)
      for(int x = 0; x < p.size; x++) {
        const uint64_t b = bit(point(x, y));
        owners[y * p.size + x] = int8_t((flats(p, 0) & b) ? 0 : (flats(p, 1) & b) ? 1 : -1);
      }
  }

  // A move in PTN: c3, Sb3, Cc4, then a spread's count (left out when 1), square, direction and drops (left out when
  // one step drops them all), as 3c3>12. A flattening has no mark (the action does not know the position).
  static std::string action_text(int action) {
    if(action == PASS_ACTION)
      return "pass";
    const int i = action_point(action), channel = action_channel(action);
    if(channel < 3)
      return std::string(channel == WALL ? "S" : channel == CAP ? "C" : "") + square_name(i);
    const int pattern = (channel - 3) / 4, k = pattern_pieces(pattern);
    std::string s = (k > 1 ? std::to_string(k) : "") + square_name(i) + DIRECTIONS[(channel - 3) % 4];
    const std::vector<int> drops = pattern_drops(pattern);
    if(drops.size() > 1)
      for(int c : drops)
        s += char('0' + c);
    return s;
  }
  // The legal move that the PTN names, or -1 and the reason. Under the double black stack, ply 1 may also be written
  // 2a1, as ptn.ninja writes it.
  static int parse_action(const Position& p, const std::string& text, std::string* why) {
    std::string reason;
    auto fail = [why, &reason](const std::string& r) {
      if(why)
        *why = r;
      return -1;
    };
    if(terminal(p))
      return fail("the game is over");
    int action;
    bool flattens;
    std::string body = text;
    if(opening == DOUBLE_BLACK_STACK && p.ply == 0 && body.size() > 1 && body[0] == '2' &&
       std::isalpha((unsigned char)body[1]))
      body = body.substr(1);
    if(!parse_ptn(body, action, flattens, reason))
      return fail(reason);
    const int i = action_point(action), channel = action_channel(action), me = next_player(p);
    if(i % MAX_LEN >= p.size || i / MAX_LEN >= p.size)
      return fail(square_name(i) + " is off the " + std::to_string(p.size) + " x " + std::to_string(p.size) +
                  " board");
    if(channel < 3) {
      if(occupied(p) & bit(i))
        return fail(square_name(i) + " is not empty");
      if(p.ply < 2 && channel != FLAT)
        return fail("the first two plies each place a flat of the opponent's colour");
      if(p.ply >= 2 && channel == CAP && p.capstones[me] == 0)
        return fail("no capstone is left");
      if(p.ply >= 2 && channel != CAP && p.stones[me] == 0)
        return fail("no stone is left");
      return action;
    }
    if(p.ply < 2)
      return fail("the first two plies are placements");
    if(!(occupied(p) & bit(i)))
      return fail(square_name(i) + " is empty");
    if(!(p.owner[me] & bit(i)))
      return fail("the stack on " + square_name(i) + " is not the mover's");
    const int pattern = (channel - 3) / 4, k = pattern_pieces(pattern);
    if(k > p.size)
      return fail("the carry limit is " + std::to_string(p.size));
    if(k > p.height[i])
      return fail("the stack on " + square_name(i) + " has " + std::to_string(p.height[i]) + " pieces");
    if(!is_legal(p, action))
      return fail(text + " meets the edge, a wall or a capstone (only a capstone alone flattens a wall)");
    bool wallAfter;
    const bool flattening = pattern_steps(pattern) > open_run(p, i, (channel - 3) % 4, wallAfter);
    if(flattens && !flattening)
      return fail(text + " flattens no wall (*)");
    return action;
  }

  // The engine protocol's text: PTN, read in any case and with no position (GameInterface's protocol_text).
  static std::string protocol_text(int action) { return action_text(action); }
  static bool parse_protocol_text(const std::string& text, int& action) {
    bool flattens;
    std::string why;
    return parse_ptn(text, action, flattens, why);
  }

  // TPS (the rows from the top rank down, each square's pieces from the bottom up: 1 White, 2 Black, S or C for a
  // top wall or capstone, x or xN for empty squares; the player to move, 1 or 2; the move number), then the plies
  // since the last placement or flattening.
  static std::string position_text(const Position& p) {
    std::string s;
    for(int y = p.size - 1; y >= 0; y--) {
      int empties = 0;
      bool first = true;
      auto flush = [&]() {
        if(empties == 0)
          return;
        s += (first ? "" : ",") + std::string("x") + (empties > 1 ? std::to_string(empties) : "");
        first = false;
        empties = 0;
      };
      for(int x = 0; x < p.size; x++) {
        const int i = point(x, y);
        if(p.height[i] == 0) {
          empties++;
          continue;
        }
        flush();
        s += first ? "" : ",";
        first = false;
        for(int depth = p.height[i] - 1; depth >= 0; depth--)
          s += char('1' + colour_at(p, i, depth));
        const int kind = kind_at(p, i);
        if(kind != FLAT)
          s += kind == WALL ? 'S' : 'C';
      }
      flush();
      if(y > 0)
        s += '/';
    }
    return s + " " + std::to_string(p.ply % 2 + 1) + " " + std::to_string(p.ply / 2 + 1) + " " +
           std::to_string(p.drawClock);
  }

  // Reads TPS, with the plies since the last placement or flattening as a fourth field (0 when it is left out).
  // Checks the format, the pieces against each player's (the reserves come from the board), the opening's plies
  // against the board, and the count against the plies. The end of the game comes from the board.
  static bool parse_position(const std::string& text, Position& p, std::string& why) {
    std::vector<std::string> fields;
    for(size_t at = 0; at < text.size();) {
      while(at < text.size() && text[at] == ' ')
        at++;
      if(at == text.size())
        break;
      const size_t end = text.find(' ', at);
      fields.push_back(text.substr(at, end == std::string::npos ? std::string::npos : end - at));
      at = end == std::string::npos ? text.size() : end;
    }
    if(fields.size() != 3 && fields.size() != 4) {
      why = "a position is TPS (the board, the player to move, the move number), then the plies since the last "
            "placement or flattening";
      return false;
    }
    std::vector<std::string> rows(1);
    for(char c : fields[0]) {
      if(c == '/')
        rows.emplace_back();
      else
        rows.back() += c;
    }
    const int size = int(rows.size());
    Position q;
    if(!new_game(size, size, q)) {
      why = "the board has " + std::to_string(size) + " rows: Tak's boards are 3 x 3 to " + std::to_string(MAX_LEN) +
            " x " + std::to_string(MAX_LEN);
      return false;
    }
    int used[2] = {0, 0}, capsUsed[2] = {0, 0};
    for(int r = 0; r < size; r++) {
      const int y = size - 1 - r;
      int x = 0;
      std::vector<std::string> squares(1);
      for(char c : rows[r]) {
        if(c == ',')
          squares.emplace_back();
        else
          squares.back() += c;
      }
      for(const std::string& sq : squares) {
        if(sq.empty()) {
          why = "row " + std::to_string(r + 1) + " has an empty square's text";
          return false;
        }
        if(sq[0] == 'x' || sq[0] == 'X') {
          int n = 1;
          if(sq.size() > 1) {
            n = 0;
            for(size_t k = 1; k < sq.size(); k++) {
              if(!std::isdigit((unsigned char)sq[k]) || n > MAX_LEN) {
                why = "\"" + sq + "\" is not x or xN";
                return false;
              }
              n = n * 10 + (sq[k] - '0');
            }
          }
          if(n < 1) {
            why = "\"" + sq + "\" is not x or xN";
            return false;
          }
          x += n;
          continue;
        }
        if(x >= size) {
          why = "row " + std::to_string(r + 1) + " has more than " + std::to_string(size) + " squares";
          return false;
        }
        int kind = FLAT;
        size_t n = sq.size();
        const char last = char(std::toupper((unsigned char)sq.back()));
        if(last == 'S' || last == 'C') {
          kind = last == 'S' ? WALL : CAP;
          n--;
        }
        if(n == 0 || n > 255) {
          why = "\"" + sq + "\" is not a stack";
          return false;
        }
        const int i = point(x, y);
        for(size_t k = 0; k < n; k++) {
          if(sq[k] != '1' && sq[k] != '2') {
            why = "\"" + sq + "\" is not a stack: its pieces are 1 (White) and 2 (Black), from the bottom up";
            return false;
          }
          const int colour = sq[k] - '1';
          if(k + 1 == n && kind == CAP)
            capsUsed[colour]++;
          else
            used[colour]++;
          put(q, i, uint64_t(colour), 1);
        }
        set_top(q, i, kind);
        x++;
      }
      if(x != size) {
        why = "row " + std::to_string(r + 1) + " does not have " + std::to_string(size) + " squares";
        return false;
      }
    }
    const char* names[2] = {"White", "Black"};
    for(int c = 0; c < 2; c++) {
      if(used[c] > STONES[size] || capsUsed[c] > CAPSTONES[size]) {
        why = std::string(names[c]) + " has more pieces than " + std::to_string(STONES[size]) + " stones and " +
              std::to_string(CAPSTONES[size]) + " capstones";
        return false;
      }
      q.stones[c] = uint8_t(STONES[size] - used[c]);
      q.capstones[c] = uint8_t(CAPSTONES[size] - capsUsed[c]);
    }
    auto number = [](const std::string& s, int& value) {
      if(s.empty() || s.size() > 6)
        return false;
      value = 0;
      for(char c : s) {
        if(!std::isdigit((unsigned char)c))
          return false;
        value = value * 10 + (c - '0');
      }
      return true;
    };
    int turn, move, since = 0;
    if(!number(fields[1], turn) || (turn != 1 && turn != 2)) {
      why = "the player to move is 1 (White) or 2 (Black), not \"" + fields[1] + "\"";
      return false;
    }
    if(!number(fields[2], move) || move < 1) {
      why = "the move number is 1 or more, not \"" + fields[2] + "\"";
      return false;
    }
    if(fields.size() == 4 && !number(fields[3], since)) {
      why = "the plies since the last placement or flattening are a count, not \"" + fields[3] + "\"";
      return false;
    }
    q.ply = (move - 1) * 2 + (turn - 1);
    const int onBoard = used[0] + used[1] + capsUsed[0] + capsUsed[1];
    if(q.ply == 0 && onBoard != 0) {
      why = "the first ply comes before any piece";
      return false;
    }
    if(q.ply == 1) {
      // One stack of Black's flats: one under the swap, two under the double black stack.
      const int pieces = opening == DOUBLE_BLACK_STACK ? 2 : 1;
      const uint64_t on = occupied(q);
      const int i = on ? __builtin_ctzll(on) : 0;
      if(popcount(on) != 1 || kind_at(q, i) != FLAT || q.height[i] != pieces || q.top[i] != (1u << pieces) - 1) {
        why = "after the first ply, the board holds " +
              std::string(pieces == 2 ? "one stack of two black flats (the double black stack)" : "one black flat");
        return false;
      }
    }
    if(q.ply >= 2 && onBoard < (opening == DOUBLE_BLACK_STACK ? 3 : 2)) {
      why = "after the first two plies, the board holds their pieces";
      return false;
    }
    if(since > FIFTY_PLIES || (q.ply < 2 ? since != 0 : since > q.ply - 2)) {
      why = "the plies since the last placement or flattening are at most 50, and fewer than the plies after the "
            "opening";
      return false;
    }
    q.drawClock = since;
    update_end(q);
    p = q;
    return true;
  }

  // The game's settings: opening, swap (the default) or double black stack (PTN's Opening tag).
  static constexpr std::array<const char*, 1> SETTINGS = {"opening"};
  static bool set_setting(const std::string& name, const std::string& value, std::string& why) {
    if(name != SETTINGS[0]) {
      why = "Tak has no setting " + name + " (its settings: opening)";
      return false;
    }
    for(int o = SWAP; o <= DOUBLE_BLACK_STACK; o++) {
      const std::string known = OPENING_NAMES[o];
      bool same = known.size() == value.size();
      for(size_t k = 0; same && k < known.size(); k++)
        same = std::tolower((unsigned char)value[k]) == known[k];
      if(same) {
        opening = o;
        return true;
      }
    }
    why = "opening is swap or double black stack, not \"" + value + "\"";
    return false;
  }
};

static_assert(GameInterface::checkRules<GameRules>(), "Tak's rules part");

}  // namespace tak
