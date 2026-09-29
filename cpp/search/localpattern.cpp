#include "../search/localpattern.h"

#include "../core/test.h"
#include "../neuralnet/nninputs.h"

using namespace std;

LocalPatternHasher::LocalPatternHasher()
  : xSize(),
    ySize(),
    zobristLocalPattern(),
    zobristPla(),
    zobristAtari()
{}


void LocalPatternHasher::init(int x, int y, Rand& rand) {
  xSize = x;
  ySize = y;
  testAssert(xSize > 0 && xSize % 2 == 1);
  testAssert(ySize > 0 && ySize % 2 == 1);
  zobristLocalPattern.resize(NUM_BOARD_COLORS * xSize * ySize);
  zobristPla.resize(NUM_BOARD_COLORS);
  zobristAtari.resize(xSize * ySize);

  for(int i = 0; i<NUM_BOARD_COLORS; i++) {
    for(int dy = 0; dy<ySize; dy++) {
      for(int dx = 0; dx<xSize; dx++) {
        uint64_t h0 = rand.nextUInt64();
        uint64_t h1 = rand.nextUInt64();
        zobristLocalPattern[i * ySize*xSize + dy*xSize + dx] = Hash128(h0,h1);
      }
    }
  }
  for(int i = 0; i<NUM_BOARD_COLORS; i++) {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    zobristPla[i] = Hash128(h0,h1);
  }
  for(int dy = 0; dy<ySize; dy++) {
    for(int dx = 0; dx<xSize; dx++) {
      uint64_t h0 = rand.nextUInt64();
      uint64_t h1 = rand.nextUInt64();
      zobristAtari[dy*xSize + dx] = Hash128(h0,h1);
    }
  }
  zobristEdge.resize(2 * enclosure::PASS);
  zobristProtected.resize(enclosure::PASS);
  for(int i = 0; i<2 * enclosure::PASS; i++) {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    zobristEdge[i] = Hash128(h0,h1);
  }
  for(int i = 0; i<enclosure::PASS; i++) {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    zobristProtected[i] = Hash128(h0,h1);
  }
}

LocalPatternHasher::~LocalPatternHasher() {
}


//Enclosure fork. Go hashes point colours and atari around the move's point. An edge has no single
//point, and nodes alone do not show which of them are joined, so the window is centred on the edge's
//source point and hashes the edges that reach into it, with their protection (Go's atari flag marks
//what is fragile now; a protected edge cannot be cut this turn). flipColors swaps the edges' owners.
static Hash128 hashEdgeWindow(const LocalPatternHasher& hasher, const Board& board, Loc loc, Player pla, bool flipColors) {
  Hash128 hash = hasher.zobristPla[flipColors ? getOpp(pla) : pla];
  if(loc != Board::PASS_LOC && loc != Board::NULL_LOC) {
    assert(loc >= Board::MOVE_LOC_BASE && loc < Board::NUM_MOVE_LOCS);
    int cell = Board::actionOfMoveLoc(loc) / enclosure::NOFF;
    int cx = cell % enclosure::SIZE;
    int cy = cell / enclosure::SIZE;
    int xRadius = hasher.xSize/2;
    int yRadius = hasher.ySize/2;
    auto inWindow = [&](int x, int y) { return std::abs(x - cx) <= xRadius && std::abs(y - cy) <= yRadius; };
    for(int owner = 0; owner < 2; owner++) {
      for(const enclosure::Edge& e : board.pos.edges[owner]) {
        if(!inWindow(e.ax, e.ay) && !inWindow(e.bx, e.by))
          continue;
        int idx = Board::edgeIndex(e);
        hash ^= hasher.zobristEdge[(flipColors ? 1 - owner : owner) * enclosure::PASS + idx];
        if(enclosure::contains(board.pos.protected_, e))
          hash ^= hasher.zobristProtected[idx];
      }
    }
  }
  return hash;
}

Hash128 LocalPatternHasher::getHash(const Board& board, Loc loc, Player pla) const {
  return hashEdgeWindow(*this, board, loc, pla, false);
}

//Enclosure fork: the hash of the image of board and loc (SymmetryHelpers acts on edges), so a
//symmetry of 0 without flipColors gives getHash.
Hash128 LocalPatternHasher::getHashWithSym(const Board& board, Loc loc, Player pla, int symmetry, bool flipColors) const {
  if(symmetry == 0)
    return hashEdgeWindow(*this, board, loc, pla, flipColors);
  Board symBoard = SymmetryHelpers::getSymBoard(board, symmetry);
  return hashEdgeWindow(*this, symBoard, SymmetryHelpers::getSymMoveLoc(loc, symBoard, symmetry), pla, flipColors);
}
