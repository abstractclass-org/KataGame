#ifndef SEARCH_LOCALPATTERN_H
#define SEARCH_LOCALPATTERN_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/rand.h"
#include "../game/board.h"

struct LocalPatternHasher {
  int xSize;
  int ySize;
  std::vector<Hash128> zobristLocalPattern;
  std::vector<Hash128> zobristPla;
  std::vector<Hash128> zobristAtari;
  //KataGame fork: the keys of the game's own hash of the window (GameInterface::numLocalKeys).
  std::vector<Hash128> zobristGame;

  LocalPatternHasher();
  ~LocalPatternHasher();

  void init(int xSize, int ySize, Rand& rand);

  //KataGame fork: loc is a move. The window (xSize by ySize points) is centred on the move's point
  //(GameEncoding::action_point) and hashes the position near it: the game's local_hash (Enclosure's: every edge
  //of either player with an endpoint inside it, and whether it is protected; ENCLOSURE.md, "Subtree value bias"),
  //else the pieces in the window, as KataGo hashes stones (GameInterface::localHash).
  Hash128 getHash(const Board& board, Loc loc, Player pla) const;

  //Returns the hash that would occur if symmetry were applied to both board and loc.
  //So basically, the only thing that changes is the zobrist indexing.
  Hash128 getHashWithSym(const Board& board, Loc loc, Player pla, int symmetry, bool flipColors) const;
};

#endif //SEARCH_LOCALPATTERN_H
