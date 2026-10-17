#pragma once
// Enclosure's own position field in an analysis query, "enclosurePosition" (GameRules::QUERY_POSITION_FIELD): the
// start position with engine.Position's fields, for a game whose move list is not known (a meaf.us game: the site's
// page gives a board, not its moves). The move server sends it. Beside the one-line position, it carries the edges of
// the last turns, for the network's history planes:
//   {"edges": [[blue edges], [red edges]], "protected": [...], "placed": [...], "scores": [blue, red],
//    "player": 0 (Blue) or 1, "movesLeft": 0-2, "moveCount": 0-MAX_MOVES, "history": [[edges of a turn], ...]},
// an edge as [ax, ay, bx, by]; "history" newest first (Position::history), at most HISTORY_TURNS turns of 2 edges.
//
// A template on the JSON type (nlohmann::json in the analysis engine), so that the rules part needs no JSON library.
// It reads the value with the JSON library's checked accessors, which throw on a missing field or a wrong type: the
// analysis engine catches them.

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine.h"

namespace enclosure {

template<class Json>
bool read_query_position(const Json& p, Position& position, std::string& why) {
  auto edgesOf = [&why](const Json& list, std::vector<Edge>& out) {
    out.clear();
    for(const Json& q : list) {
      const std::vector<int> v = q.template get<std::vector<int>>();
      const int dx = v.size() == 4 ? v[2] - v[0] : 0, dy = v.size() == 4 ? v[3] - v[1] : 0;
      if(v.size() != 4 || std::abs(dx) > 3 || std::abs(dy) > 3 || (dx == 0 && dy == 0)) {
        why = "an edge must be [ax, ay, bx, by] with both ends at most 3 apart: " + q.dump();
        return false;
      }
      for(int c : v)
        if(c < 0 || c >= SIZE) {
          why = "an edge is off the board: " + q.dump();
          return false;
        }
      out.push_back(make_edge(v[0], v[1], v[2], v[3]));
    }
    return true;
  };
  const Json& edges = p.at("edges");
  if(edges.size() != 2)
    why = "edges must have 2 lists: Blue's edges, then Red's";
  bool ok = edges.size() == 2 && edgesOf(edges[0], position.edges[0]) && edgesOf(edges[1], position.edges[1]) &&
            edgesOf(p.at("protected"), position.protected_) && edgesOf(p.at("placed"), position.placed);
  const std::vector<double> scores = p.at("scores").template get<std::vector<double>>();
  position.player = p.at("player").template get<int>();
  position.moves_left = p.at("movesLeft").template get<int>();
  position.move_count = p.at("moveCount").template get<int>();
  if(ok && (scores.size() != 2 || (position.player != 0 && position.player != 1) || position.moves_left < 0 ||
            position.moves_left > 2 || position.move_count < 0 || position.move_count > MAX_MOVES)) {
    why = "scores must have 2 numbers, player 0 or 1, movesLeft 0-2, moveCount 0-" + std::to_string(MAX_MOVES);
    ok = false;
  }
  if(ok) {
    position.scores[0] = scores[0];
    position.scores[1] = scores[1];
    position.area_ok[0] = position.area_ok[1] = false;
    const Json empty = Json::array();
    const Json& history = p.contains("history") ? p["history"] : empty;
    if(history.size() > HISTORY_TURNS) {
      why = "history has at most " + std::to_string(HISTORY_TURNS) + " turns";
      ok = false;
    }
    for(int h = 0; ok && h < (int)history.size(); h++) {
      std::vector<Edge> turn;
      ok = edgesOf(history[h], turn);
      if(ok && turn.size() > 2) {
        why = "a history turn has at most 2 edges";
        ok = false;
      }
      for(int i = 0; ok && i < (int)turn.size(); i++)
        position.history[h][i] = turn[i];
      if(ok)
        position.history_n[h] = (uint8_t)turn.size();
    }
  }
  return ok;
}

}  // namespace enclosure
