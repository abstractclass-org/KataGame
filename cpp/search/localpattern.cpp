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
  //KataGame fork: the game's keys come last, so KataGo's keys keep their values. A caller's later draws from rand
  //come after them (the subtree value bias's ZOBRIST_MOVE_LOCS), so those keep their values only while the game's
  //count stays the same (Enclosure's 3 x PASS: the edge and protected keys of before the interface).
  zobristGame.resize(GameInterface::numLocalKeys<GameRules>());
  for(size_t i = 0; i<zobristGame.size(); i++) {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    zobristGame[i] = Hash128(h0,h1);
  }
}

LocalPatternHasher::~LocalPatternHasher() {
}


//KataGame fork. Go hashes point colours and atari around the move's point. A game's move is centred on its point
//(the edge's source point in Enclosure), and the game hashes the position near it (GameInterface::localHash).
//flipColors swaps the players.
static Hash128 hashWindow(const LocalPatternHasher& hasher, const Board& board, Loc loc, Player pla, bool flipColors) {
  Hash128 hash = hasher.zobristPla[flipColors ? getOpp(pla) : pla];
  if(loc != Board::PASS_LOC && loc != Board::NULL_LOC) {
    assert(loc >= Board::MOVE_LOC_BASE && loc < Board::NUM_MOVE_LOCS);
    int cx, cy, channel;
    GameEncoding::action_point(Board::actionOfMoveLoc(loc), cx, cy, channel);
    hash ^= GameInterface::localHash<GameRules>(board.pos, cx, cy, hasher.xSize/2, hasher.ySize/2, flipColors,
                                                hasher.zobristGame.data(), hasher.zobristLocalPattern.data());
  }
  return hash;
}

Hash128 LocalPatternHasher::getHash(const Board& board, Loc loc, Player pla) const {
  return hashWindow(*this, board, loc, pla, false);
}

//KataGame fork: the hash of the image of board and loc (SymmetryHelpers maps the position), so a
//symmetry of 0 without flipColors gives getHash.
Hash128 LocalPatternHasher::getHashWithSym(const Board& board, Loc loc, Player pla, int symmetry, bool flipColors) const {
  if(symmetry == 0)
    return hashWindow(*this, board, loc, pla, flipColors);
  Board symBoard = SymmetryHelpers::getSymBoard(board, symmetry);
  return hashWindow(*this, symBoard, SymmetryHelpers::getSymMoveLoc(loc, symBoard, symmetry), pla, flipColors);
}
