#pragma once
// Enclosure's rules part of the game interface (game/gameinterface.h): each member calls the rules in engine.h and
// the notation in notation.h. The hash is Board's (board.cpp, Board::syncFromPosition) without the term that the
// front adds, KataGo's hash of the player to move: its tables come from the same seed in the same order, so the values
// are the same. The edge count and the edges left in the turn already fix the player to move.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../../core/hash.h"
#include "../../core/rand.h"
#include "../gameinterface.h"
#include "engine.h"
#include "notation.h"

namespace enclosure {

// An edge as the action that places it from its lower end (the canonical Edge's a): below PASS.
inline int edge_action(const Edge& e) {
  return (e.ay * SIZE + e.ax) * NOFF + OFF_INDEX[e.bx - e.ax + 3][e.by - e.ay + 3];
}

struct GameRules {
  static constexpr const char* NAME = "enclosure";
  using Position = enclosure::Position;
  static constexpr int MAX_BOARD_LEN = SIZE;
  static constexpr int NUM_ACTIONS = PASS + 1;
  static constexpr int PASS_ACTION = PASS;

  static bool new_game(int xSize, int ySize, Position& p) {
    if(xSize != SIZE || ySize != SIZE)
      return false;
    p = Position();
    return true;
  }
  static int x_size(const Position&) { return SIZE; }
  static int y_size(const Position&) { return SIZE; }
  static int next_player(const Position& p) { return p.player; }
  static int turn_moves_left(const Position& p) { return p.moves_left; }
  static int moves_played(const Position& p) { return p.move_count; }
  static bool terminal(const Position& p) { return p.terminal(); }
  static void legal_actions(const Position& p, std::vector<int>& out) { p.legal_actions(out); }
  static bool is_legal(const Position& p, int action) { return p.is_legal(action); }
  static void play(Position& p, int action) { p.play(action); }
  static float result(const Position& p) { return p.result(); }
  static double score(const Position& p, int player) { return p.scores[player]; }

  // Board::initHash's Enclosure tables: the edges by owner, the protected and the placed edges, the edge count and
  // the edges left in the turn.
  struct HashTables {
    Hash128 edge[2][PASS];
    Hash128 protectedEdge[PASS];
    Hash128 placedEdge[PASS];
    Hash128 moveCount[MAX_MOVES + 2];
    Hash128 movesLeft[3];
    HashTables() {
      Rand rand("Board::initHash() for Enclosure hashes");
      auto next = [&rand]() {
        uint64_t h0 = rand.nextUInt64();
        uint64_t h1 = rand.nextUInt64();
        return Hash128(h0, h1);
      };
      for(int owner = 0; owner < 2; owner++)
        for(int i = 0; i < PASS; i++)
          edge[owner][i] = next();
      for(int i = 0; i < PASS; i++) {
        protectedEdge[i] = next();
        placedEdge[i] = next();
      }
      for(int i = 0; i < MAX_MOVES + 2; i++)
        moveCount[i] = next();
      for(int i = 0; i < 3; i++)
        movesLeft[i] = next();
    }
  };
  static const HashTables& hash_tables() {
    static const HashTables tables;
    return tables;
  }
  static Hash128 score_hash(double score, uint64_t salt) {
    uint64_t bits;
    std::memcpy(&bits, &score, sizeof(bits));
    return Hash128(Hash::splitMix64(bits ^ salt), Hash::murmurMix(bits + salt));
  }
  static Hash128 hash(const Position& p) {
    const HashTables& t = hash_tables();
    Hash128 h;
    for(int owner = 0; owner < 2; owner++)
      for(const Edge& e : p.edges[owner])
        h ^= t.edge[owner][edge_action(e)];
    for(const Edge& e : p.protected_)
      h ^= t.protectedEdge[edge_action(e)];
    for(const Edge& e : p.placed)
      h ^= t.placedEdge[edge_action(e)];
    h ^= t.moveCount[std::min(p.move_count, MAX_MOVES + 1)];
    h ^= t.movesLeft[std::max(0, std::min(p.moves_left, 2))];
    h ^= score_hash(p.scores[0], 0x9E3779B97F4A7C15ULL);
    h ^= score_hash(p.scores[1], 0xC2B2AE3D27D4EB4FULL);
    return h;
  }

  // A point holds a player's piece where it is one of that player's nodes. In a position that a game reaches, no
  // point is a node of both players: an edge that touches an enemy edge cuts it. In a position given otherwise, such a
  // point is Red's, as in Board::syncFromPosition.
  static void pieces(const Position& p, int8_t* owner) {
    std::fill(owner, owner + SIZE * SIZE, int8_t(-1));
    for(int player = 0; player < 2; player++)
      for(const Edge& e : p.edges[player]) {
        owner[e.ay * SIZE + e.ax] = int8_t(player);
        owner[e.by * SIZE + e.bx] = int8_t(player);
      }
  }

  // A point is a player's where only that player's regions enclose it (Board::calculateArea).
  static void area(const Position& p, int8_t* owner) {
    uint8_t inside[2][SIZE * SIZE];  // x * SIZE + y, the engine's point order
    for(int player = 0; player < 2; player++)
      enclosed_area(p.edges[player], inside[player]);
    for(int y = 0; y < SIZE; y++)
      for(int x = 0; x < SIZE; x++) {
        const bool blue = inside[0][x * SIZE + y] != 0, red = inside[1][x * SIZE + y] != 0;
        owner[y * SIZE + x] = int8_t(blue == red ? -1 : blue ? 0 : 1);
      }
  }

  static std::string action_text(int action) { return notation::move_name(action); }
  static int parse_action(const Position& p, const std::string& text, std::string* why) {
    return notation::parse_move(text, p, why);
  }
  static std::string position_text(const Position& p) { return notation::position_string(p); }
  static bool parse_position(const std::string& text, Position& p, std::string& why) {
    return notation::parse_position(text, p, why);
  }
};

static_assert(GameInterface::checkRules<GameRules>(), "Enclosure's rules part");

}  // namespace enclosure
