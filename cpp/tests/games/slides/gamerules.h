#pragma once
// A game of the tests, not of the game layer (GAMES.md, "A game of the tests"): the rules part of a small game with a
// policy of several hundred slots and two sizes of board, for the legal moves in the rows (LEGAL_MOVES_IN_ROWS), whose
// Tak is the first game (the engine repository's games/tak/docs/design/03-moves-and-policy.md).
//
// The board is 5 x 5 or 6 x 6. A point has 13 channels of a move, on the board's largest size, 36 points and 469
// actions with pass (which is never legal), and each action is its policy index:
// - channel 0 puts a piece on the empty point. In the first 2 plies, the opening, it is the opponent's piece, as Tak's
//   first 2 plies. Later it is the mover's.
// - channels 1 to 12 slide the mover's piece on the point k = 1 to 3 points in a direction d (+y, -y, +x, -x), channel
//   1 + 4 * (k - 1) + d, when every point of its path is on the board and empty. There is none in the opening.
// The game ends after 22 plies, and its result is the count of pieces.
//
// The mask of a position has the moves that a player can make, not the moves of its pieces alone: both the opening and
// the size of the board change it.

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "../../../core/hash.h"
#include "../../../game/gameinterface.h"

namespace slides {

constexpr int LEN = 6, MAX_POINTS = LEN * LEN, CHANNELS = 13, PASS = MAX_POINTS * CHANNELS;
constexpr int OPENING_PLIES = 2, GAME_PLIES = 22, MAX_SLIDE = 3;
constexpr int DX[4] = {0, 0, 1, -1}, DY[4] = {1, -1, 0, 0};

struct Position {
  int8_t cell[MAX_POINTS];  //The owner of each point of the largest board's numbering (y * LEN + x), -1 empty.
  int size = LEN;
  int moves = 0;
  Position() {
    for(int8_t& c : cell)
      c = -1;
  }
};

struct GameRules {
  static constexpr const char* NAME = "slides";
  using Position = slides::Position;
  static constexpr int MAX_BOARD_LEN = LEN;
  static constexpr int NUM_ACTIONS = MAX_POINTS * CHANNELS + 1;
  static constexpr int PASS_ACTION = PASS;

  static bool new_game(int xSize, int ySize, Position& p) {
    p = Position();
    p.size = xSize;
    return xSize == ySize && (xSize == LEN || xSize == LEN - 1);
  }
  static int x_size(const Position& p) { return p.size; }
  static int y_size(const Position& p) { return p.size; }
  static int next_player(const Position& p) { return p.moves % 2; }
  static int turn_moves_left(const Position&) { return 1; }
  static int moves_played(const Position& p) { return p.moves; }

  static bool terminal(const Position& p) { return p.moves >= GAME_PLIES; }
  static float result(const Position& p) {
    const double margin = score(p, 0) - score(p, 1);
    return margin > 0 ? 1.0f : margin < 0 ? -1.0f : 0.0f;
  }
  static double score(const Position& p, int player) {
    int n = 0;
    for(const int8_t c : p.cell)
      n += c == player;
    return n;
  }

  // The last point of a slide of k points in direction d, or -1 when a point of its path is off the board or taken.
  static int slide_end(const Position& p, int point, int d, int k) {
    int x = point % LEN, y = point / LEN;
    for(int i = 0; i < k; i++) {
      x += DX[d];
      y += DY[d];
      if(x < 0 || y < 0 || x >= p.size || y >= p.size || p.cell[y * LEN + x] >= 0)
        return -1;
    }
    return y * LEN + x;
  }
  static bool on_board(const Position& p, int point) { return point % LEN < p.size && point / LEN < p.size; }
  static bool can_play(const Position& p, int a) {
    const int point = a / CHANNELS, channel = a % CHANNELS;
    if(!on_board(p, point))
      return false;
    if(channel == 0)
      return p.cell[point] < 0;
    if(p.moves < OPENING_PLIES || p.cell[point] != next_player(p))
      return false;
    return slide_end(p, point, (channel - 1) % 4, (channel - 1) / 4 + 1) >= 0;
  }
  static void legal_actions(const Position& p, std::vector<int>& out) {
    out.clear();
    if(terminal(p))
      return;
    for(int a = 0; a < PASS; a++)
      if(can_play(p, a))
        out.push_back(a);
  }
  static bool is_legal(const Position& p, int a) { return a >= 0 && a < PASS && !terminal(p) && can_play(p, a); }
  static void play(Position& p, int a) {
    const int point = a / CHANNELS, channel = a % CHANNELS;
    if(channel == 0)
      p.cell[point] = (int8_t)(p.moves < OPENING_PLIES ? 1 - next_player(p) : next_player(p));
    else {
      const int end = slide_end(p, point, (channel - 1) % 4, (channel - 1) / 4 + 1);
      p.cell[end] = p.cell[point];
      p.cell[point] = -1;
    }
    p.moves++;
  }

  static Hash128 hash(const Position& p) {
    uint64_t code = (uint64_t)p.size * 64 + (uint64_t)p.moves;
    for(const int8_t c : p.cell)
      code = code * 3 + (uint64_t)(c + 1);
    return Hash128((code + 1) * 0x9E3779B97F4A7C15ULL, (code + 1) * 0xC2B2AE3D27D4EB4FULL);
  }
  static void pieces(const Position& p, int8_t* owner) {
    for(int y = 0; y < p.size; y++)
      for(int x = 0; x < p.size; x++)
        owner[y * p.size + x] = p.cell[y * LEN + x];
  }
  static void area(const Position& p, int8_t* owner) { pieces(p, owner); }

  // A point's letter and number (a1 at the bottom left), then the channel after a dot for a slide: "c3", "c3.4".
  static std::string action_text(int a) {
    if(a == PASS_ACTION)
      return "pass";
    const int point = a / CHANNELS, channel = a % CHANNELS;
    std::string s = std::string(1, char('a' + point % LEN)) + std::to_string(point / LEN + 1);
    return channel == 0 ? s : s + "." + std::to_string(channel);
  }
  static int parse_action(const Position& p, const std::string& text, std::string* why) {
    std::vector<int> legal;
    legal_actions(p, legal);
    for(const int a : legal)
      if(text == action_text(a))
        return a;
    if(why != nullptr)
      *why = "not a legal move";
    return -1;
  }
  // The size, the plies and the points of the largest board in order: "5 7 .x.o...".
  static std::string position_text(const Position& p) {
    std::string s = std::to_string(p.size) + " " + std::to_string(p.moves) + " ";
    for(const int8_t c : p.cell)
      s += c < 0 ? '.' : c == 0 ? 'x' : 'o';
    return s;
  }
  static bool parse_position(const std::string& text, Position& p, std::string& why) {
    Position q;
    std::istringstream in(text);
    std::string cells;
    bool good = (bool)(in >> q.size >> q.moves >> cells) && (q.size == LEN || q.size == LEN - 1) && q.moves >= 0 &&
      q.moves <= GAME_PLIES && cells.size() == (size_t)MAX_POINTS;
    for(int i = 0; good && i < MAX_POINTS; i++) {
      good = cells[i] == '.' || cells[i] == 'x' || cells[i] == 'o';
      if(cells[i] != '.')
        q.cell[i] = cells[i] == 'x' ? 0 : 1;
    }
    if(!good) {
      why = "a position is the size, the plies, and 36 points of ., x and o";
      return false;
    }
    p = q;
    return true;
  }
};

}  // namespace slides
