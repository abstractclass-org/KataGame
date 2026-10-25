/*
 * board.h
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modifications).
 *
 * KataGame fork (GAMES.md): the Board holds a position of the game that the
 * build plays, and reaches its rules only through the game interface
 * (game/gameinterface.h, game/chosengame.h). The public interface is
 * KataGo's; Go-only members keep their names with neutral values.
 *
 * A Loc names either a point (Location::getLoc, as in KataGo: ownership,
 * network planes, symmetries) or a move: NULL_LOC, PASS_LOC, or
 * MOVE_LOC_BASE + the game's action (GameRules). MOVE_LOC_BASE is
 * MAX_ARR_SIZE: moves and points never share a value.
 */

#ifndef GAME_BOARD_H_
#define GAME_BOARD_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../external/nlohmann_json/json.hpp"
#include "../game/chosengame.h"

#include <atomic>
#include <bitset>

//The build's largest board: the game's (GameRules::MAX_BOARD_LEN), unless the build asks for a larger one
//(USE_BIGGER_BOARDS_EXPENSIVE).
#ifndef COMPILE_MAX_BOARD_LEN
#define COMPILE_MAX_BOARD_LEN GameRules::MAX_BOARD_LEN
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

  //Moves, in the engine protocol's text: "pass", or the game's text of an action (GameInterface::protocolText and
  //parseProtocolText; Enclosure's is the edge's two points, source first, such as "D10-D13").
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
  static_assert(MAX_LEN >= GameRules::MAX_BOARD_LEN, "the build's largest board holds the game's");
  //KataGame fork: the default board, if unspecified, is the game's standard board (KataGo: DEFAULT_LEN, 19 x 19).
  static constexpr int DEFAULT_X_SIZE = GameInterface::standardXSize<GameRules>();
  static constexpr int DEFAULT_Y_SIZE = GameInterface::standardYSize<GameRules>();
  static constexpr int MAX_PLAY_SIZE = MAX_LEN * MAX_LEN;  //Maximum number of playable spaces

  //Enclosure fork: KataGo sets its time constants from the board area, which in Go sets both the game's
  //length and the reach of its tactics. A game can have Go's board and a shorter game (Enclosure: 19x19 and 120
  //edges), so the constants are split by what they measure (ENCLOSURE.md, "Time horizons"):
  //- look-ahead horizons (short-term value and score targets, value surprise) count decisions, moves in our search
  //  tree: KataGo's 19x19 formulas unchanged, with the board area (lookaheadArea: a game can give its own);
  //- game-phase constants (earliest resignation, early forks, opening moves) are fractions of a game: a game has
  //  GameEncoding::GAME_LENGTH moves against GO_GAME_LENGTH moves of a 19x19 KataGo self-play game, so they use
  //  GAME_LENGTH_AREA (the area scaled by TIME_SCALE). GO_GAME_LENGTH is measured: the mean length of KataGo's own
  //  19x19 self-play games without handicap (katagoarchive.org kata1 training games: 2020-12-08, 732 games, mean
  //  328.7; 2026-09-26, 51,094 games, mean 328.1). The specific numbers are Go's and may deserve tuning for a game
  //  later.
  static constexpr double GO_GAME_LENGTH = 328.0;
  static constexpr double TIME_SCALE = GameEncoding::GAME_LENGTH / GO_GAME_LENGTH;
  static constexpr double GAME_LENGTH_AREA = 19 * 19 * TIME_SCALE;
  static constexpr int MAX_ARR_SIZE = (MAX_LEN+1)*(MAX_LEN+2)+1; //Maximum size of arrays needed (points)

  //Location used to indicate an invalid spot on the board.
  static constexpr Loc NULL_LOC = 0;
  //Location used to indicate a pass move is desired.
  static constexpr Loc PASS_LOC = 1;
  //Moves: MOVE_LOC_BASE + the game's action. Moves start after the points, so no Loc is
  //both, and a move used as a point is out of range of every point array (see ENCLOSURE.md).
  static constexpr Loc MOVE_LOC_BASE = MAX_ARR_SIZE;
  static constexpr int NUM_MOVE_LOCS = MOVE_LOC_BASE + GameRules::PASS_ACTION;  //Every move Loc is below this
  static_assert(NUM_MOVE_LOCS <= 32767, "move Locs must fit in a short");

  //The game's actions (GameRules::PASS_ACTION for a pass) and move Locs.
  static Loc moveLocOfAction(int action) { return action == GameRules::PASS_ACTION ? PASS_LOC : (Loc)(MOVE_LOC_BASE + action); }
  static int actionOfMoveLoc(Loc loc) { return loc == PASS_LOC ? GameRules::PASS_ACTION : loc - MOVE_LOC_BASE; }
  //The game's players (0 the first, 1 the second) and Players.
  static Player playerOfIndex(int p) { return p == 0 ? P_BLACK : P_WHITE; }

  //Replace the position (tests and analysis of a given position); size, colours, hash and legal moves follow.
  void setPosition(const GameRules::Position& p);
  static int indexOfPlayer(Player pla) { return pla == P_BLACK ? 0 : 1; }

  //Zobrist hashing. The Go tables stay for callers that mix them in; the position's hash is the game's
  //(GameRules::hash), with KataGo's hash of the player to move.
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

  //Structs---------------------------------------

  //Go chains. Unused (the game's pieces are not Go's chains); kept for the interface.
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
    GameRules::Position before;
  };

  //Constructors---------------------------------
  Board();  //The game's start on its standard board, DEFAULT_X_SIZE x DEFAULT_Y_SIZE
  Board(int x, int y); //The game's start on that board (GameRules::new_game); throws if the game is not played on it
  Board(const Board& other);

  Board& operator=(const Board& other);

  //Functions------------------------------------

  double sqrtBoardArea() const;
  //Enclosure fork: the score utility's unit, in points, where KataGo uses sqrtBoardArea (19 on 19x19, the square
  //root of Go's largest score): the game's GameEncoding::SCORE_UNIT (Enclosure: the square root of its largest total
  //score, about 141 points). Read by the search's score utility and the score values it reports; KataGo's factors
  //and scales apply on top.
  static double scoreUnit() { return GameEncoding::SCORE_UNIT; }
  //KataGame fork: the areas of KataGo's look-ahead horizons and of its komi sampler: the game's, else the board's
  //(GameInterface::lookaheadArea and komiArea, gameinterface.h).
  double lookaheadArea() const { return GameInterface::lookaheadArea<GameEncoding>(x_size, y_size); }
  double komiArea() const { return GameInterface::komiArea<GameEncoding>(x_size, y_size); }

  //Go stones (none in the game): neutral answers.
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

  //Moves. `pla` must be the player to move (nextPla()): the position gives it, as a turn can have several moves
  //(Enclosure's turns are 1-2-2).
  bool isLegalIgnoringKo(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const;
  bool isLegal(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const;
  //Every legal move Loc (GameRules::legal_actions).
  void getLegalMoveLocs(std::vector<Loc>& out) const;
  //The player to move, and whether the game is over.
  Player nextPla() const;
  bool isGameOver() const;

  //Points.
  bool isOnBoard(Loc loc) const;
  //Is the point next to one of pla's pieces (4 or 8 neighbours)?
  bool isAdjacentToPla(Loc loc, Player pla) const;
  bool isAdjacentOrDiagonalToPla(Loc loc, Player pla) const;
  //Is the board in its starting position (no move played)?
  bool isEmpty() const;
  //Number of pieces on the board, both players', and one player's.
  int numStonesOnBoard() const;
  int numPlaStonesOnBoard(Player pla) const;

  Hash128 getSitHashWithSimpleKo(Player pla) const;

  //No ko.
  void clearSimpleKoLoc();
  void setSimpleKoLoc(Loc loc);

  //Placing Go stones is not possible: these do nothing and return false (0 placed).
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

  //Areas: result[point] = C_BLACK / C_WHITE where the first / the second player owns the point now
  //(GameRules::area), C_EMPTY otherwise (the flags are Go's and ignored).
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

  //Scores (GameRules::score): the first player's is P_BLACK's.
  double getScore(Player pla) const;

  void checkConsistency() const;
  bool isEqualForTesting(const Board& other, bool checkNumCaptures = true, bool checkSimpleKo = true) const;

  static Board parseBoard(int xSize, int ySize, const std::string& s, char lineDelimiter = '\n');
  static void printBoard(std::ostream& out, const Board& board, Loc markLoc, const std::vector<Move>* hist);
  static std::string toStringSimple(const Board& board, char lineDelimiter = '\n');
  static nlohmann::json toJson(const Board& board);
  static Board ofJson(const nlohmann::json& data);

  //Data--------------------------------------------

  int x_size;                  //The position's board (GameRules::x_size, y_size)
  int y_size;
  Color colors[MAX_ARR_SIZE];  //Whose piece each point holds (GameRules::pieces; C_WALL off the board).
  //Unused (Go): all 0, one copy for every board, so that a copy of a board does not move them (abscls-iwsg: they
  //were 47 KB of each copy of a Blueprints board).
  static ChainData chain_data[MAX_ARR_SIZE];
  static Loc chain_head[MAX_ARR_SIZE];
  static Loc next_in_chain[MAX_ARR_SIZE];
  Loc ko_loc;   //Always NULL_LOC.

  Hash128 pos_hash; //GameRules::hash of the position, with KataGo's hash of the player to move.
  int numBlackCaptures; //0 (Go).
  int numWhiteCaptures; //0 (Go).

  short adj_offsets[8]; //Point offsets: 0-3 adjacent, 4-7 diagonal, as in KataGo.

  GameRules::Position pos; //The game's position (the rules live in the game).
  //GameRules::legal_actions by action, so isLegal is a lookup (the evaluator asks for every policy entry).
  //Computed on first use after a move, then kept: the search plays many moves (every step of every descent)
  //whose legal moves nobody asks for (2026-10-02: recomputing them after every move was about a fifth of
  //self-play's CPU). Safe to call from several threads at once.
  const std::bitset<GameRules::NUM_ACTIONS>& legalActions() const;
  //Sets the legal moves when they are known otherwise (the image of another board's under a symmetry: legality
  //is geometry, which a symmetry keeps). Only on a board no other thread reads yet.
  void setLegalActions(const std::bitset<GameRules::NUM_ACTIONS>& bits) const;

  private:
  mutable std::bitset<GameRules::NUM_ACTIONS> legalActionsCache;
  mutable std::atomic<uint8_t> legalActionsState; //LEGAL_STALE, LEGAL_COMPUTING or LEGAL_READY
  void computeLegalActions() const;
  void copyFrom(const Board& other);
  void init(int xS, int yS);
  void syncSizeFromPosition(); //x_size, y_size and adj_offsets from pos
  void syncFromPosition(); //colors and pos_hash from pos; marks legalActions stale

  friend std::ostream& operator<<(std::ostream& out, const Board& board);
};

#endif // GAME_BOARD_H_
