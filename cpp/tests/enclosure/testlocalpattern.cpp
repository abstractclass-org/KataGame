// Enclosure fork: the subtree value bias's local pattern (LocalPatternHasher::getHash)
// buckets the same local shape together and different shapes apart.
//
//   testlocalpattern
//
// The window is 7x7 around the source point of the move, and hashes the edges
// that reach into it and whether each is protected (ENCLOSURE.md, "Subtree
// value bias"). Positions are built by setting the engine's edge lists
// directly: the hasher reads nothing else.

#include <cstdio>

#include "../../core/rand.h"
#include "../../search/localpattern.h"

using namespace std;

static int failures = 0;

static void check(bool ok, const char* what) {
  if(!ok) {
    fprintf(stderr, "FAIL %s\n", what);
    failures++;
  }
}

static enclosure::Edge edge(int ax, int ay, int bx, int by) {
  return enclosure::make_edge(ax, ay, bx, by);
}

//The move from (x, y) to (x + dx, y + dy).
static Loc move(int x, int y, int dx, int dy) {
  return Board::moveLocOfAction((y * enclosure::SIZE + x) * enclosure::NOFF + enclosure::OFF_INDEX[dx + 3][dy + 3]);
}

//A board with no edges but these (index 0 blue, 1 red).
static Board boardWith(const vector<enclosure::Edge>& blue, const vector<enclosure::Edge>& red) {
  Board board;
  board.pos.edges[0] = blue;
  board.pos.edges[1] = red;
  board.pos.protected_.clear();
  board.pos.placed.clear();
  return board;
}

int main() {
  Board::initHash();
  Rand rand("testlocalpattern");
  LocalPatternHasher hasher;
  hasher.init(7, 7, rand);

  //Red's edge S -> T from S = (9, 9) to T = (11, 10).
  const Loc st = move(9, 9, 2, 1);
  const vector<enclosure::Edge> redNet = {edge(9, 9, 9, 12)};

  //Position 1: Blue's closed wall around S, corners (7, 7), (11, 7), (7, 11), (11, 11)
  //(edges of length at most 3, so each side is two edges).
  const vector<enclosure::Edge> wall = {
    edge(7, 7, 9, 7), edge(9, 7, 11, 7), edge(11, 7, 11, 9), edge(11, 9, 11, 11),
    edge(11, 11, 9, 11), edge(9, 11, 7, 11), edge(7, 11, 7, 9), edge(7, 9, 7, 7),
  };
  //Position 2: nodes at the same four corners, but their edges run away from the window.
  const vector<enclosure::Edge> loose = {
    edge(7, 7, 4, 4), edge(11, 7, 14, 4), edge(7, 11, 4, 14), edge(11, 11, 14, 14),
  };
  Hash128 closed = hasher.getHash(boardWith(wall, redNet), st, P_WHITE);
  Hash128 open = hasher.getHash(boardWith(loose, redNet), st, P_WHITE);
  check(closed != open, "a closed wall and loose nodes at the same points share a bucket");

  //Edges wholly outside the window do not matter (the network judges the rest of the board).
  vector<enclosure::Edge> wallAndFar = wall;
  wallAndFar.push_back(edge(1, 1, 3, 2));
  vector<enclosure::Edge> redFar = redNet;
  redFar.push_back(edge(16, 16, 18, 18));
  check(hasher.getHash(boardWith(wallAndFar, redFar), st, P_WHITE) == closed, "an edge outside the window changes the bucket");

  //An edge with one endpoint on the window's border (3 away) counts; one reaching 4 away does not.
  vector<enclosure::Edge> border = wall;
  border.push_back(edge(12, 12, 14, 14));  //(12, 12) is 3 from S in both directions
  check(hasher.getHash(boardWith(border, redNet), st, P_WHITE) != closed, "an edge ending on the window's border is left out");
  vector<enclosure::Edge> beyond = wall;
  beyond.push_back(edge(13, 13, 15, 15));  //4 away
  check(hasher.getHash(boardWith(beyond, redNet), st, P_WHITE) == closed, "an edge 4 away is counted");

  //Owner matters: the same edges with colours swapped are a different shape.
  check(hasher.getHash(boardWith(redNet, wall), st, P_WHITE) != closed, "swapping the owners keeps the bucket");

  //Protection matters: a protected wall edge cannot be cut this turn.
  Board protectedWall = boardWith(wall, redNet);
  protectedWall.pos.protected_.push_back(edge(11, 9, 11, 11));
  check(hasher.getHash(protectedWall, st, P_WHITE) != closed, "protecting an edge in the window keeps the bucket");
  Board protectedFar = boardWith(wallAndFar, redNet);
  protectedFar.pos.protected_.push_back(edge(1, 1, 3, 2));
  check(hasher.getHash(protectedFar, st, P_WHITE) == closed, "protecting an edge outside the window changes the bucket");

  //The mover matters, and the window follows the move's source point.
  check(hasher.getHash(boardWith(wall, redNet), st, P_BLACK) != closed, "the mover is not in the key");
  const Loc farMove = move(1, 9, 1, 1);  //Window x 0-4, y 6-12: no edge of either position reaches it
  check(hasher.getHash(boardWith(wall, redNet), farMove, P_WHITE) == hasher.getHash(boardWith(loose, redNet), farMove, P_WHITE),
        "the window is not centred on the move's source point");

  //Order of the edge lists does not matter.
  vector<enclosure::Edge> reversed(wall.rbegin(), wall.rend());
  check(hasher.getHash(boardWith(reversed, redNet), st, P_WHITE) == closed, "the hash depends on edge order");

  //Near the board's corner the window is simply clipped.
  const Loc corner = move(0, 0, 1, 2);
  Hash128 c1 = hasher.getHash(boardWith({edge(0, 0, 2, 1)}, {}), corner, P_BLACK);
  Hash128 c2 = hasher.getHash(boardWith({edge(0, 0, 2, 1), edge(10, 10, 12, 12)}, {}), corner, P_BLACK);
  check(c1 == c2, "clipping at the corner");

  printf("%s (%d failures)\n", failures == 0 ? "ok" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
