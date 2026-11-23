#pragma once
// A game of the tests, not of the game layer (GAMES.md, "A game of the tests"): the rules part of a small game with
// every kind of ending, for the optional members that no game of game/ builds with yet (game_end, DRAWS_AS_NO_RESULT).
//
// The board is 4 x 4. A move puts the mover's piece on an empty point, and the players alternate. Pass is never legal.
// - A full row or column of one player's pieces wins for that player, whatever the score (LINE).
// - A full board without a line ends by the score, with komi (COUNT, and COUNT_TIE when the margin is 0). A player's
//   score is its pieces on the two upper rows.
// - A game whose corner a1 is empty after 14 moves is a draw, whatever the score (EMPTY_CORNER).
// Each ending comes up in random play, and a line's winner often has the lower score.

#include <cstdint>
#include <string>
#include <vector>

#include "../../../core/hash.h"
#include "../../../game/gameinterface.h"

namespace endings {

constexpr int LEN = 4, POINTS = LEN * LEN, PASS = POINTS;
constexpr int CORNER_MOVES = 14;
enum Kind { LINE = 0, COUNT, COUNT_TIE, EMPTY_CORNER };

struct Position {
  int8_t cell[POINTS];
  int moves = 0;
  Position() {
    for(int8_t& c : cell)
      c = -1;
  }
};

struct GameRules {
  static constexpr const char* NAME = "endings";
  using Position = endings::Position;
  static constexpr int MAX_BOARD_LEN = LEN;
  static constexpr int NUM_ACTIONS = POINTS + 1;
  static constexpr int PASS_ACTION = PASS;

  static bool new_game(int xSize, int ySize, Position& p) {
    p = Position();
    return xSize == LEN && ySize == LEN;
  }
  static int x_size(const Position&) { return LEN; }
  static int y_size(const Position&) { return LEN; }
  static int next_player(const Position& p) { return p.moves % 2; }
  static int turn_moves_left(const Position&) { return 1; }
  static int moves_played(const Position& p) { return p.moves; }

  // The player with a full row or column, or -1.
  static int line_winner(const Position& p) {
    for(int i = 0; i < LEN; i++) {
      const int row = p.cell[i * LEN], column = p.cell[i];
      bool isRow = row >= 0, isColumn = column >= 0;
      for(int j = 1; j < LEN; j++) {
        isRow = isRow && p.cell[i * LEN + j] == row;
        isColumn = isColumn && p.cell[j * LEN + i] == column;
      }
      if(isRow)
        return row;
      if(isColumn)
        return column;
    }
    return -1;
  }
  static bool terminal(const Position& p) {
    return line_winner(p) >= 0 || p.moves == POINTS || (p.moves >= CORNER_MOVES && p.cell[0] < 0);
  }
  static GameInterface::GameEnd game_end(const Position& p, double komi) {
    GameInterface::GameEnd end;
    const int winner = line_winner(p);
    if(winner >= 0) {
      end.result = winner == 0 ? 1 : -1;
      end.kind = LINE;
      end.scored = false;
    }
    else if(p.moves == POINTS) {
      const double margin = score(p, 1) - score(p, 0) + komi;
      end.result = margin > 0 ? -1 : margin < 0 ? 1 : 0;
      end.kind = margin == 0 ? COUNT_TIE : COUNT;
    }
    else {
      end.kind = EMPTY_CORNER;
      end.scored = false;
    }
    return end;
  }
  static float result(const Position& p) { return (float)game_end(p, 0.0).result; }
  static double score(const Position& p, int player) {
    int n = 0;
    for(int i = POINTS / 2; i < POINTS; i++)
      n += p.cell[i] == player;
    return n;
  }

  static void legal_actions(const Position& p, std::vector<int>& out) {
    out.clear();
    if(terminal(p))
      return;
    for(int a = 0; a < POINTS; a++)
      if(p.cell[a] < 0)
        out.push_back(a);
  }
  static bool is_legal(const Position& p, int a) { return a >= 0 && a < POINTS && p.cell[a] < 0 && !terminal(p); }
  static void play(Position& p, int a) {
    p.cell[a] = (int8_t)next_player(p);
    p.moves++;
  }

  static Hash128 hash(const Position& p) {
    uint64_t code = 0;
    for(int8_t c : p.cell)
      code = code * 3 + (uint64_t)(c + 1);
    return Hash128((code + 1) * 0x9E3779B97F4A7C15ULL, (code + 1) * 0xC2B2AE3D27D4EB4FULL);
  }
  static void pieces(const Position& p, int8_t* owner) {
    for(int i = 0; i < POINTS; i++)
      owner[i] = p.cell[i];
  }
  static void area(const Position& p, int8_t* owner) { pieces(p, owner); }

  // The points a1 to d4, row 1 at the bottom, and the position as its 16 cells from a1.
  static std::string action_text(int a) {
    return a == PASS_ACTION ? std::string("pass") : std::string(1, char('a' + a % LEN)) + char('1' + a / LEN);
  }
  static int parse_action(const Position& p, const std::string& text, std::string* why) {
    for(int a = 0; a < POINTS; a++)
      if(text == action_text(a) && is_legal(p, a))
        return a;
    if(why != nullptr)
      *why = "not a legal point";
    return -1;
  }
  static std::string position_text(const Position& p) {
    std::string s;
    for(int8_t c : p.cell)
      s += c < 0 ? '.' : c == 0 ? 'x' : 'o';
    return s;
  }
  static bool parse_position(const std::string& text, Position& p, std::string& why) {
    Position q;
    int counts[2] = {0, 0};
    bool good = text.size() == (size_t)POINTS;
    for(int i = 0; good && i < POINTS; i++) {
      const char c = text[i];
      good = c == '.' || c == 'x' || c == 'o';
      if(c != '.') {
        q.cell[i] = c == 'x' ? 0 : 1;
        counts[q.cell[i]]++;
      }
    }
    q.moves = counts[0] + counts[1];
    if(!good || (counts[0] - counts[1]) != q.moves % 2) {
      why = "a position is 16 cells of ., x and o, with as many x as o or one more";
      return false;
    }
    p = q;
    return true;
  }
};

static_assert(GameInterface::checkRules<GameRules>(), "the test game's rules part");

}  // namespace endings
