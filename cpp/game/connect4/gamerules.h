#pragma once
// Connect-4's rules part of the game interface (game/gameinterface.h): the standard rules (gravity, four in a row in
// any direction, a draw on a full board) on a board of any width and height from 1 to 7. The standard board is 7 x 6.
// The engine repository's games/connect4/README.md describes the game, its notation and its score.
//
// A position holds each player's discs as a bitboard, as John Tromp's Fhourstones does: bit 8x + y is the point
// (x, y). A column has 8 bits and a board is at most 7 high, so the bit above each column stays empty. A shift by 1,
// 7, 8 or 9 then pairs each disc with its neighbour in one direction, and no line wraps from one column to the next.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include "../../core/hash.h"
#include "../gameinterface.h"

namespace connect4 {

constexpr int MAX_LEN = 7;                       // the longest side
constexpr int STANDARD_X_SIZE = 7, STANDARD_Y_SIZE = 6;
constexpr int COLUMN_BITS = MAX_LEN + 1;         // a column's bits in a bitboard: the top one stays empty
constexpr const char* PLAYER_LETTERS = "xo";      // a player's disc in the notation: x moves first

// An action is the point where the disc lands, numbered on the largest board, so it keeps its number on every size.
inline int action_at(int x, int y) { return y * MAX_LEN + x; }
inline int action_x(int action) { return action % MAX_LEN; }
inline int action_y(int action) { return action / MAX_LEN; }

inline char column_letter(int x) { return char('A' + x); }
inline uint64_t point_bit(int x, int y) { return uint64_t(1) << (COLUMN_BITS * x + y); }

// Whether the discs hold four in a row: up (shift 1), down-right (7), right (8) or up-right (9).
inline bool has_four(uint64_t discs) {
  for(int shift : {1, COLUMN_BITS - 1, COLUMN_BITS, COLUMN_BITS + 1}) {
    const uint64_t pairs = discs & (discs >> shift);
    if(pairs & (pairs >> (2 * shift)))
      return true;
  }
  return false;
}

// splitmix64's finalizer, a bijection of 64-bit words. KataGo's Hash::splitMix64 is the same, but it is defined in
// core/hash.cpp, and the rules part needs KataGo's headers only.
inline uint64_t mix64(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// A point's text: a column letter (either case) and a row number from 1, with no leading zero.
inline bool parse_point(const std::string& text, int& x, int& y) {
  if(text.size() < 2 || text.size() > 3 || text[1] == '0')
    return false;
  const int letter = std::toupper((unsigned char)text[0]);
  if(letter < 'A' || letter >= 'A' + MAX_LEN)
    return false;
  int row = 0;
  for(size_t i = 1; i < text.size(); i++) {
    if(!std::isdigit((unsigned char)text[i]))
      return false;
    row = row * 10 + (text[i] - '0');
  }
  x = letter - 'A';
  y = row - 1;
  return true;
}

struct Position {
  uint64_t discs[2] = {0, 0};  // each player's discs, bit 8x + y
  int8_t xSize = STANDARD_X_SIZE, ySize = STANDARD_Y_SIZE;
  int8_t height[MAX_LEN] = {};  // the discs in each column
  int8_t moves = 0;             // the discs on the board
  int8_t winner = -1;           // the player with four in a row, or -1
};

// Puts a disc of the player on the point (x, y), the landing point of column x.
inline void place(Position& p, int player, int x, int y) {
  p.discs[player] |= point_bit(x, y);
  p.height[x]++;
  p.moves++;
}

// The player whose disc is on the point, or -1.
inline int owner(const Position& p, int x, int y) {
  const uint64_t b = point_bit(x, y);
  return (p.discs[0] & b) ? 0 : (p.discs[1] & b) ? 1 : -1;
}

struct GameRules {
  static constexpr const char* NAME = "connect4";
  using Position = connect4::Position;
  static constexpr int MAX_BOARD_LEN = MAX_LEN;
  // The landing points (action_at), then pass, which is never legal.
  static constexpr int NUM_ACTIONS = MAX_LEN * MAX_LEN + 1;
  static constexpr int PASS_ACTION = MAX_LEN * MAX_LEN;
  // The standard board, 7 x 6: the front's default board and the binding's Position().
  static constexpr int STANDARD_X_SIZE = connect4::STANDARD_X_SIZE, STANDARD_Y_SIZE = connect4::STANDARD_Y_SIZE;

  static bool new_game(int xSize, int ySize, Position& p) {
    if(xSize < 1 || xSize > MAX_LEN || ySize < 1 || ySize > MAX_LEN)
      return false;
    p = Position();
    p.xSize = int8_t(xSize);
    p.ySize = int8_t(ySize);
    return true;
  }
  static int x_size(const Position& p) { return p.xSize; }
  static int y_size(const Position& p) { return p.ySize; }
  static int next_player(const Position& p) { return p.moves % 2; }
  static int turn_moves_left(const Position&) { return 1; }
  static int moves_played(const Position& p) { return p.moves; }
  static bool terminal(const Position& p) { return p.winner >= 0 || p.moves == p.xSize * p.ySize; }

  static void legal_actions(const Position& p, std::vector<int>& out) {
    out.clear();
    if(terminal(p))
      return;
    for(int x = 0; x < p.xSize; x++)
      if(p.height[x] < p.ySize)
        out.push_back(action_at(x, p.height[x]));
    std::sort(out.begin(), out.end());
  }
  static bool is_legal(const Position& p, int action) {
    if(action < 0 || action >= PASS_ACTION || terminal(p))
      return false;
    const int x = action_x(action), y = action_y(action);
    return x < p.xSize && y < p.ySize && y == p.height[x];
  }
  static void play(Position& p, int action) {
    const int player = next_player(p);
    place(p, player, action_x(action), action_y(action));
    if(has_four(p.discs[player]))
      p.winner = int8_t(player);
  }

  static float result(const Position& p) { return p.winner < 0 ? 0.0f : p.winner == 0 ? 1.0f : -1.0f; }
  // The solver's convention (Pascal Pons's): a win scores the winner's discs left before the winning disc, that disc
  // included, so a faster win scores more (18 at most on 7 x 6, 1 for a win with the last disc). The loser and a draw
  // score 0, so KataGo's margin, score(p, 1) - score(p, 0), is the solver's score from the second player's view.
  static double score(const Position& p, int player) {
    return p.winner == player ? (p.xSize * p.ySize + 2 - p.moves) / 2 : 0;
  }

  // The two bitboards (56 bits each) and the board's size fit in 128 bits, and mix64 is a bijection, so equal hashes
  // mean equal positions. The discs fix the player to move. The constant (splitmix64's increment) keeps the second
  // word of an empty board from 0, which mix64 keeps as 0.
  static Hash128 hash(const Position& p) {
    const uint64_t second = mix64(p.discs[1] ^ 0x9E3779B97F4A7C15ULL);
    const uint64_t first = p.discs[0] | uint64_t(p.xSize) << 56 | uint64_t(p.ySize) << 60;
    return Hash128(mix64(first) ^ second, second);
  }

  static void pieces(const Position& p, int8_t* owners) {
    for(int y = 0; y < p.ySize; y++)
      for(int x = 0; x < p.xSize; x++)
        owners[y * p.xSize + x] = int8_t(owner(p, x, y));
  }
  // A disc stays where it lands, so the final ownership is the final board.
  static void area(const Position& p, int8_t* owners) { pieces(p, owners); }

  // A point as GTP writes it: the column letter (A for x = 0) and the row number (1 at the bottom).
  static std::string action_text(int action) {
    if(action == PASS_ACTION)
      return "pass";
    return column_letter(action_x(action)) + std::to_string(action_y(action) + 1);
  }
  static int parse_action(const Position& p, const std::string& text, std::string* why) {
    auto fail = [why](const std::string& reason) {
      if(why)
        *why = reason;
      return -1;
    };
    int x, y;
    if(!parse_point(text, x, y))
      return fail(text == "pass" ? "Connect-4 has no pass"
                                 : "not a point: \"" + text + "\" (a column letter and a row number, as D1)");
    if(x >= p.xSize || y >= p.ySize)
      return fail(text + " is off the " + std::to_string(p.xSize) + " x " + std::to_string(p.ySize) + " board");
    if(terminal(p))
      return fail("the game is over");
    const std::string column(1, column_letter(x));
    if(p.height[x] == p.ySize)
      return fail("column " + column + " is full");
    if(y != p.height[x])
      return fail(text + " is not where column " + column + "'s next disc lands: " +
                  action_text(action_at(x, p.height[x])));
    return action_at(x, y);
  }

  // The one-line position: the rows from the bottom up, separated by /, each from column A, with . for an empty
  // point, x for the first player's disc and o for the second's. Then a space and the player to move, x or o.
  static std::string position_text(const Position& p) {
    std::string s;
    for(int y = 0; y < p.ySize; y++) {
      if(y > 0)
        s += '/';
      for(int x = 0; x < p.xSize; x++) {
        const int player = owner(p, x, y);
        s += player < 0 ? '.' : PLAYER_LETTERS[player];
      }
    }
    s += ' ';
    s += PLAYER_LETTERS[next_player(p)];
    return s;
  }
  // Checks the format, gravity, the number of each player's discs, the player to move, and that four in a row is the
  // end of the game: one player has it, that player moved last, and one of the player's top discs completes it.
  static bool parse_position(const std::string& text, Position& p, std::string& why) {
    const size_t space = text.find(' ');
    if(space == std::string::npos || text.find(' ', space + 1) != std::string::npos) {
      why = "a position is two fields: the rows from the bottom up, separated by /, then the player to move";
      return false;
    }
    std::vector<std::string> rows(1);
    for(char c : text.substr(0, space)) {
      if(c == '/')
        rows.emplace_back();
      else
        rows.back() += c;
    }
    const int xSize = int(rows[0].size()), ySize = int(rows.size());
    for(const std::string& row : rows)
      if(int(row.size()) != xSize) {
        why = "the rows have different lengths";
        return false;
      }
    Position q;
    if(!new_game(xSize, ySize, q)) {
      why = "the board is " + std::to_string(xSize) + " x " + std::to_string(ySize) + ": each side is 1 to 7";
      return false;
    }
    int xDiscs = 0;
    for(int y = 0; y < ySize; y++)
      for(int x = 0; x < xSize; x++) {
        const char c = rows[y][x];
        if(c == '.')
          continue;
        const char* letter = c ? std::strchr(PLAYER_LETTERS, c) : nullptr;
        if(!letter) {
          why = std::string("unknown point '") + c + "': a point is ., x or o";
          return false;
        }
        if(q.height[x] != y) {
          why = "the disc at " + action_text(action_at(x, y)) + " is above an empty point";
          return false;
        }
        const int player = int(letter - PLAYER_LETTERS);
        place(q, player, x, y);
        xDiscs += player == 0;
      }
    const int oDiscs = q.moves - xDiscs;
    if(xDiscs != oDiscs && xDiscs != oDiscs + 1) {
      why = "x moves first, so x has as many discs as o or one more";
      return false;
    }
    const std::string mover = text.substr(space + 1), expected(1, PLAYER_LETTERS[next_player(q)]);
    if(mover != expected) {
      why = "the player to move is " + expected + ", not \"" + mover + "\"";
      return false;
    }
    const bool four[2] = {has_four(q.discs[0]), has_four(q.discs[1])};
    if(four[0] && four[1]) {
      why = "both players have four in a row";
      return false;
    }
    if(four[0] || four[1]) {
      const int winner = four[0] ? 0 : 1;
      const std::string name(1, PLAYER_LETTERS[winner]);
      if(next_player(q) == winner) {
        why = name + " has four in a row, but " + name + " did not move last";
        return false;
      }
      bool completed = false;
      for(int x = 0; x < xSize && !completed; x++) {
        const uint64_t top = q.height[x] > 0 ? point_bit(x, q.height[x] - 1) : 0;
        completed = (q.discs[winner] & top) && !has_four(q.discs[winner] & ~top);
      }
      if(!completed) {
        why = name + "'s fours in a row need more than one last disc";
        return false;
      }
      q.winner = int8_t(winner);
    }
    p = q;
    return true;
  }
};

static_assert(GameInterface::checkRules<GameRules>(), "Connect-4's rules part");

}  // namespace connect4
