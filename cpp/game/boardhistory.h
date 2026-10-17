#ifndef GAME_BOARDHISTORY_H_
#define GAME_BOARDHISTORY_H_

// KataGame fork (GAMES.md, ENCLOSURE.md): the history of a game. The
// interface is KataGo's. Moves, legality, the end and the result come from
// the Board, which reaches the game's rules through the game interface; the
// next player always comes from the position (Enclosure's turns are 1-2-2).
// Go-only state (ko, superko, encore phases, the button, handicap, komi)
// keeps its name at a neutral value.

#include "../core/global.h"
#include "../core/hash.h"
#include "../game/board.h"
#include "../game/rules.h"

struct KoHashTable;

//Go modes, kept for the interface (they change nothing in Enclosure).
struct BoardHistoryModes {
  bool alwaysComputePassAliveUnderSuicideRules;
  bool excludeTerritoryAdjacentToAtari;

  BoardHistoryModes();
  BoardHistoryModes(bool alwaysComputePassAliveUnderSuicideRules, bool excludeTerritoryAdjacentToAtari);

  bool operator==(const BoardHistoryModes& other) const;
  bool operator!=(const BoardHistoryModes& other) const;
};

struct BoardHistory {
  Rules rules;

  //Chronological history of moves
  std::vector<Move> moveHistory;
  //Unused (Go encore), kept for the interface: one entry per move, all false
  std::vector<bool> preventEncoreHistory;
  //Situation hash after each move (getSituationAndSimpleKoHash)
  std::vector<Hash128> koHashHistory;
  size_t firstTurnIdxWithKoHistory;

  //The board and player to move when the history started
  Board initialBoard;
  Player initialPla;
  int initialEncorePhase; //0
  int64_t initialTurnNumber;

  bool assumeMultipleStartingBlackMovesAreHandicap; //Unused (Go)
  bool whiteHasMoved;
  int overrideNumHandicapStones; //Unused (Go)

  BoardHistoryModes modes;

  static const int NUM_RECENT_BOARDS = 6;
  Board recentBoards[NUM_RECENT_BOARDS];
  int currentRecentBoardIdx;
  //The player to move (from the position: turns are 1-2-2, not alternating)
  Player presumedNextMovePla;

  //Go per-point state, all false / empty
  bool wasEverOccupiedOrPlayed[Board::MAX_ARR_SIZE];
  bool superKoBanned[Board::MAX_ARR_SIZE];
  int consecutiveEndingPasses; //0
  std::vector<Hash128> hashesBeforeBlackPass;
  std::vector<Hash128> hashesBeforeWhitePass;
  int encorePhase; //0
  int numTurnsThisPhase;
  int numApproxValidTurnsThisPhase;
  int numConsecValidTurnsThisGame;
  bool koRecapBlocked[Board::MAX_ARR_SIZE];
  Hash128 koRecapBlockHash;
  STRUCT_NAMED_TRIPLE(Hash128,posHashBeforeMove,Loc,moveLoc,Player,movePla,EncoreKoCapture);
  std::vector<EncoreKoCapture> koCapturesInEncore;
  Color secondEncoreStartColors[Board::MAX_ARR_SIZE];

  float whiteBonusScore; //0
  float whiteHandicapBonusScore; //0
  bool hasButton; //false
  bool isPastNormalPhaseEnd; //false

  //Is the game over?
  bool isGameFinished;
  //Winner of the game if over, C_EMPTY for a draw.
  Player winner;
  //Red's score minus Blue's (White is Red), once the game is over.
  float finalWhiteMinusBlackScore;
  bool isScored;
  //The rules' draw: the player to move has no legal edge (not produced by this engine, which forfeits the
  //turn instead, as meaf.us does after a timeout).
  bool isNoResult;
  bool isResignation;

  BoardHistory();
  ~BoardHistory();

  BoardHistory(const Board& board, Player pla, const Rules& rules, int encorePhase);
  BoardHistory(const Board& board, Player pla, const Rules& rules, int encorePhase, const BoardHistoryModes& modes);

  BoardHistory(const BoardHistory& other);
  BoardHistory& operator=(const BoardHistory& other);

  BoardHistory(BoardHistory&& other) noexcept;
  BoardHistory& operator=(BoardHistory&& other) noexcept;

  //Clears all history and status and bonus points, sets encore phase and rules
  void clear(const Board& board, Player pla, const Rules& rules, int encorePhase);
  //Sets only the komi (kept at 0 for Enclosure)
  void setKomi(float newKomi);
  void setInitialTurnNumber(int64_t n);
  void setAssumeMultipleStartingBlackMovesAreHandicap(bool b);
  void setOverrideNumHandicapStones(int n);
  void setModes(const BoardHistoryModes& modes);

  bool suicideLegalForPassAlive() const;

  BoardHistory copyToInitial() const;

  //No komi in Enclosure: 0.
  float whiteKomiAdjustmentForDraws(double drawEquivalentWinsForWhite) const;
  float currentSelfKomi(Player pla, double drawEquivalentWinsForWhite) const;

  //Returns a reference a recent board state, where 0 is the current board, 1 is 1 move ago, etc.
  const Board& getRecentBoard(int numMovesAgo) const;

  //Is the move legal for the player to move (who must be movePla), and the game not over?
  bool isLegal(const Board& board, Loc moveLoc, Player movePla) const;
  //Passes neither end a phase nor the game in Enclosure.
  bool passWouldEndPhase(const Board& board, Player movePla) const;
  bool passWouldEndGame(const Board& board, Player movePla) const;
  bool shouldSuppressEndGameFromFriendlyPass(const Board& board, Player movePla) const;
  bool isFinalPhase() const;
  bool isPassForKo(const Board& board, Loc moveLoc, Player movePla) const;

  int64_t getCurrentTurnNumber() const;

  //Plays the move (assumed legal) and records it; ends and scores the game after line 120.
  void makeBoardMoveAssumeLegal(Board& board, Loc moveLoc, Player movePla, const KoHashTable* rootKoHashTable);
  void makeBoardMoveAssumeLegal(Board& board, Loc moveLoc, Player movePla, const KoHashTable* rootKoHashTable, bool preventEncore);
  //Plays the move if legal; returns whether it was.
  bool makeBoardMoveTolerant(Board& board, Loc moveLoc, Player movePla);
  bool makeBoardMoveTolerant(Board& board, Loc moveLoc, Player movePla, bool preventEncore);
  bool isLegalTolerant(const Board& board, Loc moveLoc, Player movePla) const;

  //Nothing to do in Enclosure (no pass-alive areas).
  void endGameIfAllPassAlive(const Board& board);
  //Ends the game now, scored by the players' scores.
  void endAndScoreGameNow(const Board& board);
  void endAndScoreGameNow(const Board& board, Color area[Board::MAX_ARR_SIZE]);
  void getAreaNow(const Board& board, Color area[Board::MAX_ARR_SIZE]) const;

  void setWinnerByResignation(Player pla);

  void printBasicInfo(std::ostream& out, const Board& board) const;
  void printDebugInfo(std::ostream& out, const Board& board) const;
  int numberOfKoHashOccurrencesInHistory(Hash128 koHash, const KoHashTable* rootKoHashTable) const;

  //No handicap in Enclosure: 0.
  static int numHandicapStonesOnBoard(const Board& b);
  int computeNumHandicapStones() const;
  int computeWhiteHandicapBonus() const;
  bool hasBlackPassOrWhiteFirst() const;

  //The position (board.pos_hash, which covers the turn state) and the player to move.
  static Hash128 getSituationAndSimpleKoHash(const Board& board, Player nextPlayer);
  //The same, and the previous move's position.
  static Hash128 getSituationAndSimpleKoAndPrevPosHash(const Board& board, const BoardHistory& hist, Player nextPlayer);
  //The same, and the position's history planes (the network sees the edges of the last turns), so that it
  //keys the network's cache.
  static Hash128 getSituationRulesAndKoHash(const Board& board, const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite);
  static Hash128 getSituationRulesAndKoHash(const Board& board, const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite, const BoardHistoryModes& modes);

private:
  void setFinalScoreAndWinner(float score);
};

//Go's ko repetition table, kept for the interface. Enclosure positions never repeat (the move count only
//grows), so it is always empty.
struct KoHashTable {
  uint32_t* idxTable;
  std::vector<Hash128> koHashHistorySortedByLowBits;
  size_t firstTurnIdxWithKoHistory;

  static const int TABLE_SIZE = 1 << 10;
  static const uint64_t TABLE_MASK = TABLE_SIZE-1;

  KoHashTable();
  ~KoHashTable();
  KoHashTable(const KoHashTable& other) = delete;
  KoHashTable& operator=(const KoHashTable& other) = delete;

  size_t size() const;

  void recompute(const BoardHistory& history);
  bool containsHash(Hash128 hash) const;
  int numberOfOccurrencesOfHash(Hash128 hash) const;
};

#endif  // GAME_BOARDHISTORY_H_
