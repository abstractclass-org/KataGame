#pragma once
// Enclosure's rules part of the game interface (game/gameinterface.h): each member calls the rules in engine.h and
// the notation in notation.h. The hash is the one that Board's syncFromPosition computed before the interface, without
// the term that the front adds, KataGo's hash of the player to move: its tables come from Board::initHash's seed, in
// its order. The edge count and the edges left in the turn already fix the player to move. The optional members keep
// the engine protocol's moves and the local pattern hash as they were before the interface.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../core/global.h"
#include "../../core/hash.h"
#include "../../core/rand.h"
#include "../gameinterface.h"
#include "engine.h"
#include "notation.h"
#include "queryposition.h"

namespace enclosure {

// An edge as the action that places it from its end a (the lower end, for a canonical Edge): below PASS.
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

  // The tables of the hash: the edges by owner, the protected and the placed edges, the edge count and the edges left
  // in the turn. They are drawn at the first call, from the seed and in the order that Board::initHash used before the
  // interface, so the hash keeps its values.
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
    static const HashTables hashTables;  // not "tables": engine.h's enclosure::tables
    return hashTables;
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

  // The engine protocol's moves (GTP, the analysis engine): the edge's two points, source first, such as "D10-D13",
  // with the column letters A to S (I included, unlike the notation) and the rows 1 to 19. The move server and the
  // sites read this form.
  static std::string protocol_point(int x, int y) { return std::string(1, char('A' + x)) + std::to_string(y + 1); }
  // A point in that form, upper case.
  static bool parse_protocol_point(const std::string& s, int& x, int& y) {
    if(s.size() < 2 || s[0] < 'A' || s[0] > 'S')
      return false;
    x = s[0] - 'A';
    int row;
    if(!Global::tryStringToInt(s.substr(1), row) || row < 1 || row > SIZE)
      return false;
    y = row - 1;
    return true;
  }
  static std::string protocol_text(int action) {
    const int cell = action / NOFF, k = action % NOFF;
    const int ax = cell % SIZE, ay = cell / SIZE;
    return protocol_point(ax, ay) + "-" + protocol_point(ax + OFFS[k].dx, ay + OFFS[k].dy);
  }
  static bool parse_protocol_text(const std::string& text, int& action) {
    const std::string s = Global::toUpper(text);
    const size_t dash = s.find('-');
    int ax, ay, bx, by;
    if(dash == std::string::npos || !parse_protocol_point(s.substr(0, dash), ax, ay) ||
       !parse_protocol_point(s.substr(dash + 1), bx, by))
      return false;
    const int dx = bx - ax, dy = by - ay;
    if(std::abs(dx) > 3 || std::abs(dy) > 3 || (dx == 0 && dy == 0))
      return false;
    // The action from the source: the edge with the source as its end a, not canonical.
    action = edge_action(Edge{int8_t(ax), int8_t(ay), int8_t(bx), int8_t(by)});
    return true;
  }

  // The position near a point, for the search's local pattern hash (ENCLOSURE.md, "Subtree value bias"). An edge has no
  // single point, and nodes alone do not show which of them are joined, so the window hashes the edges of either
  // player with an end inside it, and whether each is protected (KataGo's atari flag marks what is fragile now; a
  // protected edge cannot be cut this turn). The keys: each owner's edges by the edge's action, then the protected
  // edges, in the order that the hasher drew them before the interface. The window of 7 covers a whole edge from its
  // source (offsets up to 3).
  static constexpr int NUM_LOCAL_KEYS = 3 * PASS;
  static constexpr int LOCAL_WINDOW = 7;
  static Hash128 local_hash(const Position& p, int cx, int cy, int xRadius, int yRadius, bool flipPlayers,
                            const Hash128* keys) {
    auto inWindow = [&](int x, int y) { return std::abs(x - cx) <= xRadius && std::abs(y - cy) <= yRadius; };
    Hash128 h;
    for(int owner = 0; owner < 2; owner++)
      for(const Edge& e : p.edges[owner]) {
        if(!inWindow(e.ax, e.ay) && !inWindow(e.bx, e.by))
          continue;
        const int idx = edge_action(e);
        h ^= keys[(flipPlayers ? 1 - owner : owner) * PASS + idx];
        if(contains(p.protected_, e))
          h ^= keys[2 * PASS + idx];
      }
    return h;
  }

  // The analysis engine's field "enclosurePosition" (queryposition.h): the move server sends it.
  static constexpr const char* QUERY_POSITION_FIELD = "enclosurePosition";
  template<class Json>
  static bool parse_query_position(const Json& value, Position& p, std::string& why) {
    return read_query_position(value, p, why);
  }
};

static_assert(GameInterface::checkRules<GameRules>(), "Enclosure's rules part");

}  // namespace enclosure
