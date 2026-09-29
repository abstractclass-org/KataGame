/*
 * board.h
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modifications).
 *
 * Enclosure fork (see ENCLOSURE.md): the Board holds an Enclosure position
 * (game/enclosure/engine.h, the verified engine). The public interface is
 * KataGo's; Go-only members keep their names with neutral values.
 *
 * A Loc names either a point (Location::getLoc, as in KataGo: ownership,
 * network planes, symmetries) or a move: NULL_LOC, PASS_LOC, or
 * MOVE_LOC_BASE + the Enclosure action index (y * 19 + x) * 48 + offset. MOVE_LOC_BASE is
 * MAX_ARR_SIZE: moves and points never share a value.
 */

#ifndef GAME_BOARD_H_
#define GAME_BOARD_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../external/nlohmann_json/json.hpp"
#include "../game/enclosure/engine.h"

#include <bitset>

#ifndef COMPILE_MAX_BOARD_LEN
#define COMPILE_MAX_BOARD_LEN 19
#endif

struct Board;

//Player: Blue moves first and is P_BLACK; Red is P_WHITE.
typedef int8_t Player;
static constexpr Player P_BLACK = 1;
static constexpr Player P_WHITE = 2;

//Color of a point: whose node it is, if anyone's.
typedef int8_t Color;
static constexpr Color C_EMPTY = 0;
static constexpr Color C_BLACK = 1;
static constexpr Color C_WHITE = 2;
static constexpr Color C_WALL = 3;
static constexpr int NUM_BOARD_COLORS = 4;

static inline Color getOpp(Color c)
{return c ^ 3;}

//Conversions for players and colors
namespace PlayerIO {
  char colorToChar(Color c);
  std::string playerToStringShort(Player p);
  std::string playerToString(Player p);
  bool tryParsePlayer(const std::string& s, Player& pla);
  Player parsePlayer(const std::string& s);
}

//A point or a move, see the top of this file.
typedef short Loc;

namespace Location
{
  //Points, as in KataGo.
  inline Loc getLoc(int x, int y, int x_size) { return (Loc)((x+1) + (y+1)*(x_size+1)); }
  inline int getX(Loc loc, int x_size) { return (loc % (x_size+1)) - 1; }
  inline int getY(Loc loc, int x_size) { return (loc / (x_size+1)) - 1; }

  void getAdjacentOffsets(short adj_offsets[8], int x_size);
  bool isAdjacent(Loc loc0, Loc loc1, int x_size);
  Loc getMirrorLoc(Loc loc, int x_size, int y_size);
  Loc getCenterLoc(int x_size, int y_size);
  Loc getCenterLoc(const Board& b);
  bool isCentral(Loc loc, int x_size, int y_size);
  bool isNearCentral(Loc loc, int x_size, int y_size);
  int distance(Loc loc0, Loc loc1, int x_size);
  int euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size);

  //Moves, in the site's notation: "pass", or the edge's two points, source first, such as "D10-D13"
  //(columns A-S are x 0-18, rows 1-19 are y 0-18).
  std::string toString(Loc loc, int x_size, int y_size);
  std::string toString(Loc loc, const Board& b);
  std::string toStringMach(Loc loc, int x_size);
  std::string toStringMach(Loc loc, const Board& b);

  bool tryOfString(const std::string& str, int x_size, int y_size, Loc& result);
  bool tryOfString(const std::string& str, const Board& b, Loc& result);
  Loc ofString(const std::string& str, int x_size, int y_size);
  Loc ofString(const std::string& str, const Board& b);

  //Same, but will parse "null" as Board::NULL_LOC
  bool tryOfStringAllowNull(const std::string& str, int x_size, int y_size, Loc& result);
  bool tryOfStringAllowNull(const std::string& str, const Board& b, Loc& result);
  Loc ofStringAllowNull(const std::string& str, int x_size, int y_size);
  Loc ofStringAllowNull(const std::string& str, const Board& b);

  std::vector<Loc> parseSequence(const std::string& str, const Board& b);
}

//Simple structure for storing moves. Not used below, but this is a convenient place to define it.
STRUCT_NAMED_PAIR(Loc,loc,Player,pla,Move);

struct Board
{
  //Initialization------------------------------
  //Initialize the zobrist hash.
  //MUST BE CALLED AT PROGRAM START!
  static void initHash();

  //Board parameters and Constants----------------------------------------

  static constexpr int MAX_LEN = COMPILE_MAX_BOARD_LEN;  //Maximum edge length allowed for the board
  static constexpr int DEFAULT_LEN = std::min(MAX_LEN,19); //Default edge length for board if unspecified
  static constexpr int MAX_PLAY_SIZE = MAX_LEN * MAX_LEN;  //Maximum number of playable spaces

  //Enclosure fork (owner): KataGo measures time horizons (short-term targets, value surprise, forks, resignation,
  //opening moves) by the board area, its proxy for a Go game's length. An Enclosure game has enclosure::MAX_MOVES
  //(120) edges against GO_GAME_LENGTH moves of a 19x19 KataGo self-play game, so KataGo's move counts are scaled by
  //TIME_SCALE, and its area-based formulas use GAME_LENGTH_AREA. GO_GAME_LENGTH is measured: the mean length of
  //KataGo's own 19x19 self-play games without handicap (katagoarchive.org kata1 training games: 2020-12-08, 732
  //games, mean 328.7; 2026-09-26, 51,094 games, mean 328.1). The specific numbers are Go's and may deserve tuning
  //for Enclosure later.
  static constexpr double GO_GAME_LENGTH = 328.0;
  static constexpr double TIME_SCALE = enclosure::MAX_MOVES / GO_GAME_LENGTH;
  static constexpr double GAME_LENGTH_AREA = 19 * 19 * TIME_SCALE;
  static constexpr int MAX_ARR_SIZE = (MAX_LEN+1)*(MAX_LEN+2)+1; //Maximum size of arrays needed (points)

  //Location used to indicate an invalid spot on the board.
  static constexpr Loc NULL_LOC = 0;
  //Location used to indicate a pass move is desired.
  static constexpr Loc PASS_LOC = 1;
  //Moves: MOVE_LOC_BASE + the Enclosure action index. Moves start after the points, so no Loc is
  //both, and a move used as a point is out of range of every point array (see ENCLOSURE.md).
  static constexpr Loc MOVE_LOC_BASE = MAX_ARR_SIZE;
  static constexpr int NUM_MOVE_LOCS = MOVE_LOC_BASE + enclosure::PASS;  //Every move Loc is below this
  static_assert(NUM_MOVE_LOCS <= 32767, "move Locs must fit in a short");

  //Enclosure actions (enclosure::PASS for a pass) and move Locs.
  static Loc moveLocOfAction(int action) { return action == enclosure::PASS ? PASS_LOC : (Loc)(MOVE_LOC_BASE + action); }
  static int actionOfMoveLoc(Loc loc) { return loc == PASS_LOC ? enclosure::PASS : loc - MOVE_LOC_BASE; }
  //Enclosure players (0 blue, 1 red) and Players.
  static Player playerOfIndex(int p) { return p == 0 ? P_BLACK : P_WHITE; }
  //A stored (canonical, undirected) edge as an action index from its lower end: below enclosure::PASS.
  static int edgeIndex(const enclosure::Edge& e);

  //Replace the position (tests and analysis of a given position); colours, hash and legal moves follow.
  void setPosition(const enclosure::Position& p);
  static int indexOfPlayer(Player pla) { return pla == P_BLACK ? 0 : 1; }

  //Zobrist hashing. The Go tables stay for callers that mix them in; the Enclosure position is hashed
  //from its edges, protected and placed edges, scores and turn state (what the engine compares).
  static bool IS_ZOBRIST_INITALIZED;
  static Hash128 ZOBRIST_SIZE_X_HASH[MAX_LEN+1];
  static Hash128 ZOBRIST_SIZE_Y_HASH[MAX_LEN+1];
  static Hash128 ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][4];
  static Hash128 ZOBRIST_BOARD_HASH2[MAX_ARR_SIZE][4];
  static Hash128 ZOBRIST_PLAYER_HASH[4];
  static Hash128 ZOBRIST_KO_LOC_HASH[MAX_ARR_SIZE];
  static Hash128 ZOBRIST_KO_MARK_HASH[MAX_ARR_SIZE][4];
  static Hash128 ZOBRIST_ENCORE_HASH[3];
  static Hash128 ZOBRIST_SECOND_ENCORE_START_HASH[MAX_ARR_SIZE][4];
  static const Hash128 ZOBRIST_PASS_ENDS_PHASE;
  static const Hash128 ZOBRIST_GAME_IS_OVER;
  static Hash128 ZOBRIST_EDGE_HASH[2][enclosure::PASS];  //[owner][canonical edge as an action]
  static Hash128 ZOBRIST_PROTECTED_HASH[enclosure::PASS];
  static Hash128 ZOBRIST_PLACED_HASH[enclosure::PASS];
  static Hash128 ZOBRIST_MOVE_COUNT_HASH[enclosure::MAX_MOVES+2];
  static Hash128 ZOBRIST_MOVES_LEFT_HASH[3];

  //Structs---------------------------------------

  //Go chains. Unused (no stones in Enclosure); kept for the interface.
  struct ChainData {
    Player owner;
    short num_locs;
    short num_liberties;
  };

  //Undo information: the position before the move.
  struct MoveRecord {
    Player pla;
    Loc loc;
    Loc ko_loc;
    uint8_t capDirs;
    enclosure::Position before;
  };

  //Constructors---------------------------------
  Board();  //The Enclosure starting position (19 x 19)
  Board(int x, int y); //The same; the size must be 19 x 19
  Board(const Board& other);

  Board& operator=(const Board&) = default;

  //Functions------------------------------------

  double sqrtBoardArea() const;

  //Go stones (none in Enclosure): neutral answers.
  int getChainSize(Loc loc) const;
  int getNumLiberties(Loc loc) const;
  int getNumLibertiesAfterPlay(Loc loc, Player pla, int max) const;
  void getBoundNumLibertiesAfterPlay(Loc loc, Player pla, int& lowerBound, int& upperBound) const;
  int getNumImmediateLiberties(Loc loc) const;
  bool isSuicide(Loc loc, Player pla) const;
  bool isIllegalSuicide(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const;
  bool isKoBanned(Loc loc) const;
  bool isSimpleEye(Loc loc, Player pla) const;
  bool wouldBeCapture(Loc loc, Player pla) const;
  bool wouldBeKoCapture(Loc loc, Player pla) const;
  Loc getKoCaptureLoc(Loc loc, Player pla) const;
  bool isAdjacentToChain(Loc loc, Loc chain) const;
  bool isNonPassAliveSelfConnection(Loc loc, Player pla, const Color* passAliveArea) const;

  //Moves. `pla` must be the player to move (nextPla()): Enclosure's turns are 1-2-2, not alternating.
  bool isLegalIgnoringKo(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const;
  bool isLegal(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const;
  //Every legal move Loc (PASS_LOC only when no edge can be placed).
  void getLegalMoveLocs(std::vector<Loc>& out) const;
  //The player to move, and whether the game is over (after line 120).
  Player nextPla() const;
  bool isGameOver() const;

  //Points.
  bool isOnBoard(Loc loc) const;
  //Is the point next to one of pla's nodes (4 or 8 neighbours)?
  bool isAdjacentToPla(Loc loc, Player pla) const;
  bool isAdjacentOrDiagonalToPla(Loc loc, Player pla) const;
  //Is the board in its starting position (no move played)?
  bool isEmpty() const;
  //Number of nodes on the board, both players', and one player's.
  int numStonesOnBoard() const;
  int numPlaStonesOnBoard(Player pla) const;

  Hash128 getSitHashWithSimpleKo(Player pla) const;

  //No ko in Enclosure.
  void clearSimpleKoLoc();
  void setSimpleKoLoc(Loc loc);

  //Placing Go stones is not possible in Enclosure: these do nothing and return false (0 placed).
  bool setStone(Loc loc, Color color);
  bool setStoneFailIfNoLibs(Loc loc, Color color);
  bool setStonesFailIfNoLibs(const std::vector<Move>& placements);
  int setStonesTolerant(const std::vector<Move>& placements);
  void regenChainsFromColors();

  //Plays the move if legal (pla must be the player to move); returns whether it was.
  bool playMove(Loc loc, Player pla, bool isMultiStoneSuicideLegal);
  void playMoveAssumeLegal(Loc loc, Player pla);
  MoveRecord playMoveRecorded(Loc loc, Player pla);
  void undo(MoveRecord record);
  Hash128 getPosHashAfterMove(Loc loc, Player pla) const;
  bool simpleRepetitionBoundGt(Loc loc, int bound) const;

  bool searchIsLadderCaptured(Loc loc, bool defenderFirst, std::vector<Loc>& buf);
  bool searchIsLadderCapturedAttackerFirst2Libs(Loc loc, std::vector<Loc>& buf, std::vector<Loc>& workingMoves);

  //Areas: result[point] = C_BLACK / C_WHITE where only Blue's / Red's regions enclose the point,
  //C_EMPTY otherwise (the flags are Go's and ignored).
  void calculateArea(
    Color* result,
    bool nonPassAliveStones,
    bool safeBigTerritories,
    bool unsafeBigTerritories,
    bool isMultiStoneSuicideLegal
  ) const;
  void calculateIndependentLifeArea(
    Color* result,
    int& whiteMinusBlackIndependentLifeRegionCount,
    bool keepTerritories,
    bool keepStones,
    bool excludeTerritoryAdjacentToAtari,
    bool isMultiStoneSuicideLegal
  ) const;

  //Scores (the engine's): Blue's is P_BLACK's.
  double getScore(Player pla) const;

  void checkConsistency() const;
  bool isEqualForTesting(const Board& other, bool checkNumCaptures = true, bool checkSimpleKo = true) const;

  static Board parseBoard(int xSize, int ySize, const std::string& s, char lineDelimiter = '\n');
  static void printBoard(std::ostream& out, const Board& board, Loc markLoc, const std::vector<Move>* hist);
  static std::string toStringSimple(const Board& board, char lineDelimiter = '\n');
  static nlohmann::json toJson(const Board& board);
  static Board ofJson(const nlohmann::json& data);

  //Data--------------------------------------------

  int x_size;                  //19
  int y_size;                  //19
  Color colors[MAX_ARR_SIZE];  //Whose node each point is (C_WALL off the board).
  ChainData chain_data[MAX_ARR_SIZE]; //Unused (Go).
  Loc chain_head[MAX_ARR_SIZE];       //Unused (Go).
  Loc next_in_chain[MAX_ARR_SIZE];    //Unused (Go).
  Loc ko_loc;   //Always NULL_LOC.

  Hash128 pos_hash; //Zobrist hash of the Enclosure position (including the turn state).
  int numBlackCaptures; //0 (Go).
  int numWhiteCaptures; //0 (Go).

  short adj_offsets[8]; //Point offsets: 0-3 adjacent, 4-7 diagonal, as in KataGo.

  enclosure::Position pos; //The Enclosure position (the rules live here).
  //pos.legal_actions() by action index, so isLegal is a lookup (the evaluator asks for every policy entry).
  std::bitset<enclosure::PASS + 1> legalActions;

  private:
  void init(int xS, int yS);
  void syncFromPosition(); //colors, pos_hash and legalActions from pos

  friend std::ostream& operator<<(std::ostream& out, const Board& board);
};

#endif // GAME_BOARD_H_
