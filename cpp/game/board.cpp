#include "../game/board.h"

/*
 * board.cpp
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modifications).
 *
 * KataGame fork: the rules are the game's, through the game interface's rules
 * part (GameRules, GAMES.md). Point geometry, player names and the Go hash
 * tables are KataGo's; Go stone logic answers neutrally (ENCLOSURE.md).
 */

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "../core/rand.h"
#include "../core/test.h"

using namespace std;

//STATIC VARS-----------------------------------------------------------------------------
bool Board::IS_ZOBRIST_INITALIZED = false;
Hash128 Board::ZOBRIST_SIZE_X_HASH[MAX_LEN+1];
Hash128 Board::ZOBRIST_SIZE_Y_HASH[MAX_LEN+1];
Hash128 Board::ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][4];
Hash128 Board::ZOBRIST_PLAYER_HASH[4];
Hash128 Board::ZOBRIST_KO_LOC_HASH[MAX_ARR_SIZE];
Hash128 Board::ZOBRIST_KO_MARK_HASH[MAX_ARR_SIZE][4];
Hash128 Board::ZOBRIST_ENCORE_HASH[3];
Hash128 Board::ZOBRIST_BOARD_HASH2[MAX_ARR_SIZE][4];
Hash128 Board::ZOBRIST_SECOND_ENCORE_START_HASH[MAX_ARR_SIZE][4];
const Hash128 Board::ZOBRIST_PASS_ENDS_PHASE = //Based on sha256 hash of Board::ZOBRIST_PASS_ENDS_PHASE
  Hash128(0x853E097C279EBF4EULL, 0xE3153DEF9E14A62CULL);
const Hash128 Board::ZOBRIST_GAME_IS_OVER = //Based on sha256 hash of Board::ZOBRIST_GAME_IS_OVER
  Hash128(0xb6f9e465597a77eeULL, 0xf1d583d960a4ce7fULL);

//LOCATION (points, as in KataGo)----------------------------------------------------------
void Location::getAdjacentOffsets(short adj_offsets[8], int x_size)
{
  adj_offsets[0] = -(x_size+1);
  adj_offsets[1] = -1;
  adj_offsets[2] = 1;
  adj_offsets[3] = (x_size+1);
  adj_offsets[4] = -(x_size+1)-1;
  adj_offsets[5] = -(x_size+1)+1;
  adj_offsets[6] = (x_size+1)-1;
  adj_offsets[7] = (x_size+1)+1;
}

bool Location::isAdjacent(Loc loc0, Loc loc1, int x_size)
{
  return loc0 == loc1 - (x_size+1) || loc0 == loc1 - 1 || loc0 == loc1 + 1 || loc0 == loc1 + (x_size+1);
}

Loc Location::getMirrorLoc(Loc loc, int x_size, int y_size) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getLoc(x_size-1-getX(loc,x_size),y_size-1-getY(loc,x_size),x_size);
}

Loc Location::getCenterLoc(int x_size, int y_size) {
  if(x_size % 2 == 0 || y_size % 2 == 0)
    return Board::NULL_LOC;
  return getLoc(x_size / 2, y_size / 2, x_size);
}

Loc Location::getCenterLoc(const Board& b) {
  return getCenterLoc(b.x_size,b.y_size);
}

bool Location::isCentral(Loc loc, int x_size, int y_size) {
  int x = getX(loc,x_size);
  int y = getY(loc,x_size);
  return x >= (x_size-1)/2 && x <= x_size/2 && y >= (y_size-1)/2 && y <= y_size/2;
}

bool Location::isNearCentral(Loc loc, int x_size, int y_size) {
  int x = getX(loc,x_size);
  int y = getY(loc,x_size);
  return x >= (x_size-1)/2-1 && x <= x_size/2+1 && y >= (y_size-1)/2-1 && y <= y_size/2+1;
}

int Location::distance(Loc loc0, Loc loc1, int x_size) {
  int dx = getX(loc1,x_size) - getX(loc0,x_size);
  int dy = (loc1-loc0-dx) / (x_size+1);
  return (dx >= 0 ? dx : -dx) + (dy >= 0 ? dy : -dy);
}

int Location::euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size) {
  int dx = getX(loc1,x_size) - getX(loc0,x_size);
  int dy = (loc1-loc0-dx) / (x_size+1);
  return dx*dx + dy*dy;
}

//LOCATION (moves, in the engine protocol's text)-----------------------------------------
string Location::toString(Loc loc, int x_size, int y_size) {
  (void)x_size; (void)y_size;
  if(loc == Board::NULL_LOC)
    return string("null");
  if(loc == Board::PASS_LOC)
    return string("pass");
  if(loc < Board::MOVE_LOC_BASE || loc >= Board::NUM_MOVE_LOCS)
    return string("illegalmove:") + Global::intToString((int)loc);
  return GameInterface::protocolText<GameRules>(Board::actionOfMoveLoc(loc));
}

string Location::toString(Loc loc, const Board& b) {
  return toString(loc,b.x_size,b.y_size);
}

string Location::toStringMach(Loc loc, int x_size) {
  return toString(loc, x_size, x_size);
}

string Location::toStringMach(Loc loc, const Board& b) {
  return toString(loc,b.x_size,b.y_size);
}

bool Location::tryOfString(const string& str, int x_size, int y_size, Loc& result) {
  (void)x_size; (void)y_size;
  string s = Global::trim(str);
  if(Global::isEqualCaseInsensitive(s, "pass")) {
    result = Board::PASS_LOC;
    return true;
  }
  int action;
  if(!GameInterface::parseProtocolText<GameRules>(s, action) || action < 0 || action >= GameRules::PASS_ACTION)
    return false;
  result = Board::moveLocOfAction(action);
  return true;
}

bool Location::tryOfString(const string& str, const Board& b, Loc& result) {
  return tryOfString(str,b.x_size,b.y_size,result);
}

Loc Location::ofString(const string& str, int x_size, int y_size) {
  Loc result;
  if(tryOfString(str,x_size,y_size,result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofString(const string& str, const Board& b) {
  return ofString(str,b.x_size,b.y_size);
}

bool Location::tryOfStringAllowNull(const string& str, int x_size, int y_size, Loc& result) {
  if(str == "null") {
    result = Board::NULL_LOC;
    return true;
  }
  return tryOfString(str, x_size, y_size, result);
}

bool Location::tryOfStringAllowNull(const string& str, const Board& b, Loc& result) {
  return tryOfStringAllowNull(str,b.x_size,b.y_size,result);
}

Loc Location::ofStringAllowNull(const string& str, int x_size, int y_size) {
  Loc result;
  if(tryOfStringAllowNull(str,x_size,y_size,result))
    return result;
  throw StringError("Could not parse board location: " + str);
}

Loc Location::ofStringAllowNull(const string& str, const Board& b) {
  return ofStringAllowNull(str,b.x_size,b.y_size);
}

vector<Loc> Location::parseSequence(const string& str, const Board& board) {
  vector<string> pieces = Global::split(Global::trim(str),' ');
  vector<Loc> locs;
  for(size_t i = 0; i<pieces.size(); i++) {
    string piece = Global::trim(pieces[i]);
    if(piece.length() <= 0)
      continue;
    locs.push_back(Location::ofString(piece,board));
  }
  return locs;
}

//CONSTRUCTORS AND INITIALIZATION----------------------------------------------------------

Board::Board()
{
  init(DEFAULT_X_SIZE,DEFAULT_Y_SIZE);
}

Board::Board(int x, int y)
{
  init(x,y);
}

Board::Board(const Board& other)
{
  copyFrom(other);
}

Board& Board::operator=(const Board& other)
{
  if(this != &other)
    copyFrom(other);
  return *this;
}

static constexpr uint8_t LEGAL_STALE = 0, LEGAL_COMPUTING = 1, LEGAL_READY = 2;

void Board::copyFrom(const Board& other)
{
  x_size = other.x_size;
  y_size = other.y_size;
  memcpy(colors, other.colors, sizeof(Color)*MAX_ARR_SIZE);
  memcpy(chain_data, other.chain_data, sizeof(ChainData)*MAX_ARR_SIZE);
  memcpy(chain_head, other.chain_head, sizeof(Loc)*MAX_ARR_SIZE);
  memcpy(next_in_chain, other.next_in_chain, sizeof(Loc)*MAX_ARR_SIZE);
  ko_loc = other.ko_loc;
  pos_hash = other.pos_hash;
  numBlackCaptures = other.numBlackCaptures;
  numWhiteCaptures = other.numWhiteCaptures;
  memcpy(adj_offsets, other.adj_offsets, sizeof(short)*8);
  pos = other.pos;
  //A copy keeps the other board's legal moves only if they are complete; else it computes its own when asked.
  if(other.legalActionsState.load(std::memory_order_acquire) == LEGAL_READY) {
    legalActionsCache = other.legalActionsCache;
    legalActionsState.store(LEGAL_READY, std::memory_order_relaxed);
  }
  else
    legalActionsState.store(LEGAL_STALE, std::memory_order_relaxed);
}

const std::bitset<GameRules::NUM_ACTIONS>& Board::legalActions() const {
  if(legalActionsState.load(std::memory_order_acquire) != LEGAL_READY)
    computeLegalActions();
  return legalActionsCache;
}

void Board::setLegalActions(const std::bitset<GameRules::NUM_ACTIONS>& bits) const {
  legalActionsCache = bits;
  legalActionsState.store(LEGAL_READY, std::memory_order_release);
}

void Board::computeLegalActions() const {
  uint8_t expected = LEGAL_STALE;
  if(legalActionsState.compare_exchange_strong(expected, LEGAL_COMPUTING, std::memory_order_acq_rel)) {
    legalActionsCache.reset();
    std::vector<int> actions;
    GameRules::legal_actions(pos, actions);
    for(int a : actions)
      legalActionsCache.set(a);
    legalActionsState.store(LEGAL_READY, std::memory_order_release);
    return;
  }
  //Another thread is computing them: wait for it.
  while(legalActionsState.load(std::memory_order_acquire) != LEGAL_READY)
    std::this_thread::yield();
}

void Board::init(int xS, int yS)
{
  assert(IS_ZOBRIST_INITALIZED);
  if(!GameRules::new_game(xS, yS, pos))
    throw StringError(string("The game ") + GameRules::NAME + " is not played on a " + Global::intToString(xS) +
                      " x " + Global::intToString(yS) + " board");
  memset(chain_data, 0, sizeof(chain_data));
  memset(chain_head, 0, sizeof(chain_head));
  memset(next_in_chain, 0, sizeof(next_in_chain));
  ko_loc = NULL_LOC;
  numBlackCaptures = 0;
  numWhiteCaptures = 0;
  syncSizeFromPosition();
  syncFromPosition();
}

void Board::initHash()
{
  if(IS_ZOBRIST_INITALIZED)
    return;
  Rand rand("Board::initHash()");

  auto nextHash = [&rand]() {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    return Hash128(h0,h1);
  };

  for(int i = 0; i<4; i++)
    ZOBRIST_PLAYER_HASH[i] = nextHash();
  for(int i = 0; i<3; i++)
    ZOBRIST_ENCORE_HASH[i] = nextHash();

  for(int i = 0; i<MAX_ARR_SIZE; i++) {
    for(Color j = 0; j<4; j++) {
      if(j == C_EMPTY || j == C_WALL)
        ZOBRIST_BOARD_HASH[i][j] = Hash128();
      else
        ZOBRIST_BOARD_HASH[i][j] = nextHash();

      if(j == C_EMPTY || j == C_WALL)
        ZOBRIST_KO_MARK_HASH[i][j] = Hash128();
      else
        ZOBRIST_KO_MARK_HASH[i][j] = nextHash();
    }
    ZOBRIST_KO_LOC_HASH[i] = nextHash();
  }

  rand.init("Board::initHash() for ZOBRIST_SECOND_ENCORE_START hashes");
  for(int i = 0; i<MAX_ARR_SIZE; i++) {
    for(Color j = 0; j<4; j++) {
      if(j == C_EMPTY || j == C_WALL)
        ZOBRIST_SECOND_ENCORE_START_HASH[i][j] = Hash128();
      else
        ZOBRIST_SECOND_ENCORE_START_HASH[i][j] = nextHash();
    }
  }

  rand.init("Board::initHash() for ZOBRIST_SIZE hashes");
  for(int i = 0; i<MAX_LEN+1; i++) {
    ZOBRIST_SIZE_X_HASH[i] = nextHash();
    ZOBRIST_SIZE_Y_HASH[i] = nextHash();
  }

  rand.init("Board::initHash() for second set of ZOBRIST hashes");
  for(int i = 0; i<MAX_ARR_SIZE; i++) {
    for(Color j = 0; j<4; j++) {
      ZOBRIST_BOARD_HASH2[i][j] = nextHash();
      ZOBRIST_BOARD_HASH2[i][j].hash0 = Hash::murmurMix(ZOBRIST_BOARD_HASH2[i][j].hash0);
      ZOBRIST_BOARD_HASH2[i][j].hash1 = Hash::splitMix64(ZOBRIST_BOARD_HASH2[i][j].hash1);
    }
  }

  IS_ZOBRIST_INITALIZED = true;
}

//A point's owner in the game's terms (-1 none, 0 the first player, 1 the second) as a Color.
static_assert(C_EMPTY == 0 && C_BLACK == 1 && C_WHITE == 2, "a player's index + 1 is its Color");
static inline Color colorOfOwner(int8_t owner) {
  return (Color)(owner + 1);
}

void Board::setPosition(const GameRules::Position& p) {
  pos = p;
  syncSizeFromPosition();
  syncFromPosition();
}

void Board::syncSizeFromPosition() {
  x_size = GameRules::x_size(pos);
  y_size = GameRules::y_size(pos);
  Location::getAdjacentOffsets(adj_offsets,x_size);
}

void Board::syncFromPosition() {
  for(int i = 0; i < MAX_ARR_SIZE; i++)
    colors[i] = C_WALL;
  int8_t owner[GameRules::MAX_BOARD_LEN * GameRules::MAX_BOARD_LEN];
  GameRules::pieces(pos, owner);
  //The game's size (x_size and y_size hold the same), a constant for a game of one size.
  const int xSize = GameRules::x_size(pos), ySize = GameRules::y_size(pos);
  for(int y = 0; y < ySize; y++)
    for(int x = 0; x < xSize; x++)
      colors[Location::getLoc(x,y,xSize)] = colorOfOwner(owner[y * xSize + x]);
  pos_hash = GameRules::hash(pos) ^ ZOBRIST_PLAYER_HASH[nextPla()];
  legalActionsState.store(LEGAL_STALE, std::memory_order_release);
}

Hash128 Board::getSitHashWithSimpleKo(Player pla) const {
  //The turn state is already in pos_hash; the player is mixed in as KataGo does.
  return pos_hash ^ Board::ZOBRIST_PLAYER_HASH[pla];
}

void Board::clearSimpleKoLoc() {
  ko_loc = NULL_LOC;
}
void Board::setSimpleKoLoc(Loc loc) {
  (void)loc;
  ko_loc = NULL_LOC;
}

double Board::sqrtBoardArea() const {
  return std::sqrt((double)x_size * (double)y_size);
}

//GO STONES: none in the game---------------------------------------------------------------
int Board::getChainSize(Loc) const { return 0; }
int Board::getNumLiberties(Loc) const { return 0; }
int Board::getNumLibertiesAfterPlay(Loc, Player, int) const { return 0; }
void Board::getBoundNumLibertiesAfterPlay(Loc, Player, int& lowerBound, int& upperBound) const {
  lowerBound = 0;
  upperBound = 0;
}
int Board::getNumImmediateLiberties(Loc) const { return 0; }
bool Board::isSuicide(Loc, Player) const { return false; }
bool Board::isIllegalSuicide(Loc, Player, bool) const { return false; }
bool Board::isKoBanned(Loc) const { return false; }
bool Board::isSimpleEye(Loc, Player) const { return false; }
bool Board::wouldBeCapture(Loc, Player) const { return false; }
bool Board::wouldBeKoCapture(Loc, Player) const { return false; }
Loc Board::getKoCaptureLoc(Loc, Player) const { return NULL_LOC; }
bool Board::isAdjacentToChain(Loc, Loc) const { return false; }
bool Board::isNonPassAliveSelfConnection(Loc, Player, const Color*) const { return false; }
bool Board::setStone(Loc, Color) { return false; }
bool Board::setStoneFailIfNoLibs(Loc, Color) { return false; }
bool Board::setStonesFailIfNoLibs(const std::vector<Move>&) { return false; }
int Board::setStonesTolerant(const std::vector<Move>&) { return 0; }
void Board::regenChainsFromColors() {}
//Enclosure positions never recur (the move count only rises and is hashed), so the state alone identifies a
//node and the graph search merges transposed turns. (false, Go's safe answer, would chain the move path.)
bool Board::simpleRepetitionBoundGt(Loc, int) const { return true; }
bool Board::searchIsLadderCaptured(Loc, bool, std::vector<Loc>&) { return false; }
bool Board::searchIsLadderCapturedAttackerFirst2Libs(Loc, std::vector<Loc>&, std::vector<Loc>&) { return false; }

//MOVES----------------------------------------------------------------------------------

Player Board::nextPla() const {
  return playerOfIndex(GameRules::next_player(pos));
}

bool Board::isGameOver() const {
  return GameRules::terminal(pos);
}

void Board::getLegalMoveLocs(std::vector<Loc>& out) const {
  std::vector<int> actions;
  GameRules::legal_actions(pos, actions);
  out.clear();
  for(int a : actions)
    out.push_back(moveLocOfAction(a));
}

bool Board::isLegalIgnoringKo(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const {
  return isLegal(loc, pla, isMultiStoneSuicideLegal);
}

bool Board::isLegal(Loc loc, Player pla, bool) const {
  if(pla != nextPla() || GameRules::terminal(pos))
    return false;
  if(loc != PASS_LOC && (loc < MOVE_LOC_BASE || loc >= NUM_MOVE_LOCS))
    return false;
  //A lookup once the legal moves are listed; else this move alone (the search checks its chosen child at every
  //step of a descent, search.cpp, where listing all of them cost about 24 us a step).
  if(legalActionsState.load(std::memory_order_acquire) == LEGAL_READY)
    return legalActionsCache.test(actionOfMoveLoc(loc));
  return GameRules::is_legal(pos, actionOfMoveLoc(loc));
}

bool Board::playMove(Loc loc, Player pla, bool isMultiStoneSuicideLegal) {
  if(!isLegal(loc, pla, isMultiStoneSuicideLegal))
    return false;
  playMoveAssumeLegal(loc, pla);
  return true;
}

void Board::playMoveAssumeLegal(Loc loc, Player pla) {
  (void)pla;
  GameRules::play(pos, actionOfMoveLoc(loc));
  syncFromPosition();
}

Board::MoveRecord Board::playMoveRecorded(Loc loc, Player pla) {
  MoveRecord record;
  record.pla = pla;
  record.loc = loc;
  record.ko_loc = ko_loc;
  record.capDirs = 0;
  record.before = pos;
  playMoveAssumeLegal(loc, pla);
  return record;
}

void Board::undo(Board::MoveRecord record) {
  pos = record.before;
  syncFromPosition();
}

Hash128 Board::getPosHashAfterMove(Loc loc, Player pla) const {
  Board copy(*this);
  copy.playMoveAssumeLegal(loc, pla);
  return copy.pos_hash;
}

//POINTS---------------------------------------------------------------------------------

bool Board::isOnBoard(Loc loc) const {
  return loc >= 0 && loc < MAX_ARR_SIZE && colors[loc] != C_WALL;
}

bool Board::isAdjacentToPla(Loc loc, Player pla) const {
  for(int i = 0; i < 4; i++)
    if(colors[loc + adj_offsets[i]] == pla)
      return true;
  return false;
}

bool Board::isAdjacentOrDiagonalToPla(Loc loc, Player pla) const {
  for(int i = 0; i < 8; i++)
    if(colors[loc + adj_offsets[i]] == pla)
      return true;
  return false;
}

bool Board::isEmpty() const {
  return GameRules::moves_played(pos) == 0;
}

int Board::numStonesOnBoard() const {
  return numPlaStonesOnBoard(P_BLACK) + numPlaStonesOnBoard(P_WHITE);
}

int Board::numPlaStonesOnBoard(Player pla) const {
  int n = 0;
  for(int i = 0; i < MAX_ARR_SIZE; i++)
    if(colors[i] == pla)
      n++;
  return n;
}

//AREAS AND SCORES-----------------------------------------------------------------------

void Board::calculateArea(Color* result, bool, bool, bool, bool) const {
  int8_t owner[GameRules::MAX_BOARD_LEN * GameRules::MAX_BOARD_LEN];
  GameRules::area(pos, owner);
  for(int i = 0; i < MAX_ARR_SIZE; i++)
    result[i] = C_EMPTY;
  for(int y = 0; y < y_size; y++)
    for(int x = 0; x < x_size; x++)
      result[Location::getLoc(x,y,x_size)] = colorOfOwner(owner[y * x_size + x]);
}

void Board::calculateIndependentLifeArea(
  Color* result, int& whiteMinusBlackIndependentLifeRegionCount, bool, bool, bool, bool
) const {
  calculateArea(result, false, false, false, false);
  whiteMinusBlackIndependentLifeRegionCount = 0;
}

double Board::getScore(Player pla) const {
  return GameRules::score(pos, indexOfPlayer(pla));
}

//TESTING AND IO-------------------------------------------------------------------------

void Board::checkConsistency() const {
  Board copy(*this);
  copy.syncFromPosition();
  if(copy.pos_hash != pos_hash)
    throw StringError("Board::checkConsistency: pos_hash does not match the position");
  for(int i = 0; i < MAX_ARR_SIZE; i++)
    if(copy.colors[i] != colors[i])
      throw StringError("Board::checkConsistency: colors do not match the position");
}

bool Board::isEqualForTesting(const Board& other, bool, bool) const {
  return x_size == other.x_size && y_size == other.y_size && pos_hash == other.pos_hash;
}

Board Board::parseBoard(int xSize, int ySize, const string& s, char lineDelimiter) {
  (void)s; (void)lineDelimiter;
  //Enclosure positions come from move sequences (BoardHistory), not diagrams: the starting position.
  return Board(xSize, ySize);
}

void Board::printBoard(ostream& out, const Board& board, Loc markLoc, const vector<Move>* hist) {
  (void)markLoc;
  if(hist != NULL)
    out << "MoveNum: " << hist->size() << " ";
  out << "HASH: " << board.pos_hash << "\n";
  out << "Line " << GameRules::moves_played(board.pos) << ", " << PlayerIO::playerToString(board.nextPla())
      << " to move (" << GameRules::turn_moves_left(board.pos) << " left in the turn), scores "
      << GameRules::score(board.pos, 0) << " / " << GameRules::score(board.pos, 1) << "\n";
  for(int y = board.y_size - 1; y >= 0; y--) {
    out << (y + 1 < 10 ? " " : "") << (y + 1) << " ";
    for(int x = 0; x < board.x_size; x++)
      out << PlayerIO::colorToChar(board.colors[Location::getLoc(x,y,board.x_size)]) << ' ';
    out << "\n";
  }
  out << "   ";
  for(int x = 0; x < board.x_size; x++)
    out << (char)('A' + x) << ' ';
  out << "\n";
}

ostream& operator<<(ostream& out, const Board& board) {
  Board::printBoard(out, board, Board::NULL_LOC, NULL);
  return out;
}

string Board::toStringSimple(const Board& board, char lineDelimiter) {
  string s;
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++)
      s += PlayerIO::colorToChar(board.colors[Location::getLoc(x,y,board.x_size)]);
    s += lineDelimiter;
  }
  return s;
}

nlohmann::json Board::toJson(const Board& board) {
  //Positions are replayed from moves; the diagram is for display only.
  nlohmann::json data;
  data["xSize"] = board.x_size;
  data["ySize"] = board.y_size;
  data["stones"] = Board::toStringSimple(board,'|');
  return data;
}

Board Board::ofJson(const nlohmann::json& data) {
  return Board(data["xSize"].get<int>(), data["ySize"].get<int>());
}

//PLAYER IO------------------------------------------------------------------------------
char PlayerIO::colorToChar(Color c)
{
  switch(c) {
  case C_BLACK: return 'X';
  case C_WHITE: return 'O';
  case C_EMPTY: return '.';
  default:  return '#';
  }
}

string PlayerIO::playerToString(Color c)
{
  switch(c) {
  case C_BLACK: return "Black";
  case C_WHITE: return "White";
  case C_EMPTY: return "Empty";
  default:  return "Wall";
  }
}

string PlayerIO::playerToStringShort(Color c)
{
  switch(c) {
  case C_BLACK: return "B";
  case C_WHITE: return "W";
  case C_EMPTY: return "E";
  default:  return "";
  }
}

bool PlayerIO::tryParsePlayer(const string& s, Player& pla) {
  string str = Global::toLower(s);
  if(str == "black" || str == "b" || str == "blue") {
    pla = P_BLACK;
    return true;
  }
  else if(str == "white" || str == "w" || str == "red") {
    pla = P_WHITE;
    return true;
  }
  return false;
}

Player PlayerIO::parsePlayer(const string& s) {
  Player pla = C_EMPTY;
  bool suc = tryParsePlayer(s,pla);
  if(!suc)
    throw StringError("Could not parse player: " + s);
  return pla;
}
