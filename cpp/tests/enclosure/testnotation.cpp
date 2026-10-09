// Enclosure fork: the notation (game/enclosure/notation.h) against the shared test vectors.
//
//   testnotation tests/data/enclosure_notation.txt
//
// The file comes from the Enclosure repository's tools/notation_vectors.py: archived games replayed in its Python
// engine (verified against the sites), each game's moves in the notation, and the position after every edge. The
// Python notation (enclosure/notation.py) is checked against the same file. For every position: the C++ replay
// prints the same text, the text parses back to the same rules state and the same legal moves, and each move's name
// reads back as the move. Then malformed positions are refused.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../../game/enclosure/notation.h"

using namespace std;
using namespace enclosure;
using namespace enclosure::notation;

static vector<Edge> sorted_edges(vector<Edge> v) {
  sort(v.begin(), v.end(), [](const Edge& a, const Edge& b) {
    return tie(a.ax, a.ay, a.bx, a.by) < tie(b.ax, b.ay, b.bx, b.by);
  });
  return v;
}

static bool same_state(const Position& a, const Position& b) {
  for(int s = 0; s < 2; s++)
    if(!(sorted_edges(a.edges[s]) == sorted_edges(b.edges[s])) || a.scores[s] != b.scores[s])
      return false;
  return sorted_edges(a.protected_) == sorted_edges(b.protected_) && sorted_edges(a.placed) == sorted_edges(b.placed) &&
         a.player == b.player && a.moves_left == b.moves_left && a.move_count == b.move_count;
}

int main(int argc, char** argv) {
  if(argc != 2) {
    fprintf(stderr, "usage: testnotation VECTORS_FILE\n");
    return 2;
  }
  ifstream in(argv[1]);
  if(!in) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  int games = 0, positions = 0, failures = 0;
  auto fail = [&](const string& what) {
    if(failures++ < 20)
      fprintf(stderr, "FAIL game %d: %s\n", games, what.c_str());
  };
  Position p;
  vector<string> moves;
  int n = 0;
  string line;
  while(getline(in, line)) {
    if(!line.empty() && line.back() == '\r')  // a file written with Windows line ends
      line.pop_back();
    if(line.rfind("game ", 0) == 0) {
      istringstream words(line.substr(5));
      string id;
      words >> id;
      moves.clear();
      for(string m; words >> m;)
        moves.push_back(m);
      p = Position();
      n = 0;
      games++;
      continue;
    }
    const size_t space = line.find(' ');
    const int number = atoi(line.substr(0, space).c_str());
    const string text = line.substr(space + 1);
    positions++;
    if(number != n)
      fail("line for edge " + to_string(number) + " where " + to_string(n) + " was expected");
    if(position_string(p) != text)
      fail("after " + to_string(n) + " edges: printed\n  " + position_string(p) + "\nexpected\n  " + text);
    Position q;
    string why;
    if(!parse_position(text, q, why))
      fail("after " + to_string(n) + " edges: refused (" + why + ")");
    else {
      if(position_string(q) != text)
        fail("after " + to_string(n) + " edges: parsed and printed again differs");
      if(!same_state(p, q))
        fail("after " + to_string(n) + " edges: the parsed rules state differs");
      vector<int> lp, lq;
      p.legal_actions(lp);
      q.legal_actions(lq);
      sort(lp.begin(), lp.end());
      sort(lq.begin(), lq.end());
      if(lp != lq)
        fail("after " + to_string(n) + " edges: the parsed position's legal moves differ");
    }
    if(n < (int)moves.size()) {
      const int action = parse_move(moves[n], p, &why);
      if(action < 0) {
        fail("move " + to_string(n + 1) + " " + moves[n] + ": " + why);
        continue;
      }
      if(move_name(action) != moves[n])
        fail("move " + to_string(n + 1) + " " + moves[n] + " is named " + move_name(action));
      p.play(action);
      n++;
    }
  }

  // Malformed positions, each refused (the same as tests/test_notation.py's).
  const vector<pair<string, string>> bad = {
    {"A10-D10 Q10-T10 b1 0/0", "5 fields"},
    {"A10-D10 Q10-T10 x1 0/0 0", "mover"},
    {"A10-D10* Q10-T10 b1 0/0 0", "protected"},
    {"A10-D10 Q10-T10+ b1 0/0 0", "placed"},
    {"A10-D10 A10-D10 r2 0/0 1", "twice"},
    {"A10-D10 Q10-T10 b1 0-0 0", "scores"},
    {"A10-D10 Q10-T10 b1 0/0 121", "count"},
    {"A10-I10 Q10-T10 b1 0/0 0", "not an edge"},
    {"A10-E10 Q10-T10 b1 0/0 0", "at most 3 apart"},
  };
  for(const auto& [text, expected] : bad) {
    Position q;
    string why;
    if(parse_position(text, q, why))
      fail("accepted: " + text);
    else if(why.find(expected) == string::npos)
      fail("refused " + text + " for '" + why + "', expected '" + expected + "'");
  }
  Position start;
  if(position_string(start) != "A10-D10 Q10-T10 b1 0/0 0")
    fail("the start prints as " + position_string(start));

  printf("notation: %d games, %d positions, %d failures\n", games, positions, failures);
  return failures == 0 ? 0 : 1;
}
