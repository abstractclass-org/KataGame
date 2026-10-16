#include "../game/boardhistory.h"

// Enclosure fork: see boardhistory.h and ENCLOSURE.md.

#include <algorithm>
#include <cstring>

//The chosen game's interface, both parts (game/gameinterface.h): so every build checks the network encoding part.
#include "../game/chosengame.h"

using namespace std;

BoardHistoryModes::BoardHistoryModes()
  : alwaysComputePassAliveUnderSuicideRules(false),
    excludeTerritoryAdjacentToAtari(false)
{}

BoardHistoryModes::BoardHistoryModes(bool alwaysComputePassAliveUnderSuicideRules_, bool excludeTerritoryAdjacentToAtari_)
  : alwaysComputePassAliveUnderSuicideRules(alwaysComputePassAliveUnderSuicideRules_),
    excludeTerritoryAdjacentToAtari(excludeTerritoryAdjacentToAtari_)
{}

bool BoardHistoryModes::operator==(const BoardHistoryModes& other) const {
  return alwaysComputePassAliveUnderSuicideRules == other.alwaysComputePassAliveUnderSuicideRules &&
    excludeTerritoryAdjacentToAtari == other.excludeTerritoryAdjacentToAtari;
}
bool BoardHistoryModes::operator!=(const BoardHistoryModes& other) const {
  return !(*this == other);
}

BoardHistory::BoardHistory()
  : rules(),
    moveHistory(),preventEncoreHistory(),koHashHistory(),
    firstTurnIdxWithKoHistory(0),
    initialBoard(),
    initialPla(P_BLACK),
    initialEncorePhase(0),
    initialTurnNumber(0),
    assumeMultipleStartingBlackMovesAreHandicap(false),
    whiteHasMoved(false),
    overrideNumHandicapStones(-1),
    modes(),
    recentBoards(),
    currentRecentBoardIdx(0),
    presumedNextMovePla(P_BLACK),
    consecutiveEndingPasses(0),
    hashesBeforeBlackPass(),hashesBeforeWhitePass(),
    encorePhase(0),
    numTurnsThisPhase(0),
    numApproxValidTurnsThisPhase(0),
    numConsecValidTurnsThisGame(0),
    koRecapBlockHash(),
    koCapturesInEncore(),
    whiteBonusScore(0.0f),
    whiteHandicapBonusScore(0.0f),
    hasButton(false),
    isPastNormalPhaseEnd(false),
    isGameFinished(false),winner(C_EMPTY),finalWhiteMinusBlackScore(0.0f),
    isScored(false),isNoResult(false),isResignation(false)
{
  std::fill(wasEverOccupiedOrPlayed, wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, false);
  std::fill(superKoBanned, superKoBanned+Board::MAX_ARR_SIZE, false);
  std::fill(koRecapBlocked, koRecapBlocked+Board::MAX_ARR_SIZE, false);
  std::fill(secondEncoreStartColors, secondEncoreStartColors+Board::MAX_ARR_SIZE, C_EMPTY);
}

BoardHistory::~BoardHistory()
{}

BoardHistory::BoardHistory(const Board& board, Player pla, const Rules& r, int ePhase)
  : BoardHistory()
{
  clear(board,pla,r,ePhase);
}

BoardHistory::BoardHistory(const Board& board, Player pla, const Rules& r, int ePhase, const BoardHistoryModes& m)
  : BoardHistory()
{
  modes = m;
  clear(board,pla,r,ePhase);
}

BoardHistory::BoardHistory(const BoardHistory& other) = default;
BoardHistory& BoardHistory::operator=(const BoardHistory& other) = default;
BoardHistory::BoardHistory(BoardHistory&& other) noexcept = default;
BoardHistory& BoardHistory::operator=(BoardHistory&& other) noexcept = default;

void BoardHistory::clear(const Board& board, Player pla, const Rules& r, int ePhase) {
  (void)pla; //The player to move is the position's.
  (void)ePhase;
  rules = r;
  moveHistory.clear();
  preventEncoreHistory.clear();
  koHashHistory.clear();
  firstTurnIdxWithKoHistory = 0;
  initialBoard = board;
  initialPla = board.nextPla();
  initialEncorePhase = 0;
  initialTurnNumber = 0;
  whiteHasMoved = false;
  overrideNumHandicapStones = -1;
  for(int i = 0; i < NUM_RECENT_BOARDS; i++)
    recentBoards[i] = board;
  currentRecentBoardIdx = 0;
  presumedNextMovePla = board.nextPla();
  std::fill(wasEverOccupiedOrPlayed, wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, false);
  std::fill(superKoBanned, superKoBanned+Board::MAX_ARR_SIZE, false);
  std::fill(koRecapBlocked, koRecapBlocked+Board::MAX_ARR_SIZE, false);
  std::fill(secondEncoreStartColors, secondEncoreStartColors+Board::MAX_ARR_SIZE, C_EMPTY);
  consecutiveEndingPasses = 0;
  hashesBeforeBlackPass.clear();
  hashesBeforeWhitePass.clear();
  encorePhase = 0;
  numTurnsThisPhase = 0;
  numApproxValidTurnsThisPhase = 0;
  numConsecValidTurnsThisGame = 0;
  koRecapBlockHash = Hash128();
  koCapturesInEncore.clear();
  whiteBonusScore = 0.0f;
  whiteHandicapBonusScore = 0.0f;
  hasButton = false;
  isPastNormalPhaseEnd = false;
  isGameFinished = false;
  winner = C_EMPTY;
  finalWhiteMinusBlackScore = 0.0f;
  isScored = false;
  isNoResult = false;
  isResignation = false;
  koHashHistory.push_back(getSituationAndSimpleKoHash(board,presumedNextMovePla));
  if(board.isGameOver())
    endAndScoreGameNow(board);
}

BoardHistory BoardHistory::copyToInitial() const {
  BoardHistory hist(initialBoard, initialPla, rules, initialEncorePhase, modes);
  hist.setInitialTurnNumber(initialTurnNumber);
  return hist;
}

void BoardHistory::setKomi(float newKomi) {
  rules.komi = newKomi;
}

void BoardHistory::setInitialTurnNumber(int64_t n) {
  initialTurnNumber = n;
}

void BoardHistory::setAssumeMultipleStartingBlackMovesAreHandicap(bool b) {
  assumeMultipleStartingBlackMovesAreHandicap = b;
}

void BoardHistory::setOverrideNumHandicapStones(int n) {
  overrideNumHandicapStones = n;
}

void BoardHistory::setModes(const BoardHistoryModes& m) {
  modes = m;
}

bool BoardHistory::suicideLegalForPassAlive() const {
  return false;
}

float BoardHistory::whiteKomiAdjustmentForDraws(double) const {
  return 0.0f;
}

//Enclosure fork (owner, 2026-10-04): komi is points added to Red's (White's) final score; positive favours Red,
//negative Blue. The player's own view, as KataGo's.
float BoardHistory::currentSelfKomi(Player pla, double) const {
  return pla == P_WHITE ? rules.komi : -rules.komi;
}

const Board& BoardHistory::getRecentBoard(int numMovesAgo) const {
  assert(numMovesAgo >= 0 && numMovesAgo < NUM_RECENT_BOARDS);
  int idx = (currentRecentBoardIdx - numMovesAgo + NUM_RECENT_BOARDS) % NUM_RECENT_BOARDS;
  return recentBoards[idx];
}

bool BoardHistory::isLegal(const Board& board, Loc moveLoc, Player movePla) const {
  if(isGameFinished)
    return false;
  return board.isLegal(moveLoc, movePla, rules.multiStoneSuicideLegal);
}

bool BoardHistory::isLegalTolerant(const Board& board, Loc moveLoc, Player movePla) const {
  //Enclosure fork (owner, 2026-10-04): the site's pass forfeits the rest of the turn and is allowed at any time (its
  //rules' pass_; online it comes from clock timeouts), so a replayed game (analysis queries, GTP, SGF and start
  //positions) accepts it anywhere. The search and self-play still offer it only without a legal edge (isLegal).
  if(moveLoc == Board::PASS_LOC && !isGameFinished && movePla == board.nextPla())
    return true;
  return isLegal(board, moveLoc, movePla);
}

bool BoardHistory::passWouldEndPhase(const Board&, Player) const { return false; }
bool BoardHistory::passWouldEndGame(const Board&, Player) const { return false; }
bool BoardHistory::shouldSuppressEndGameFromFriendlyPass(const Board&, Player) const { return false; }
bool BoardHistory::isFinalPhase() const { return true; }
bool BoardHistory::isPassForKo(const Board&, Loc, Player) const { return false; }

int64_t BoardHistory::getCurrentTurnNumber() const {
  return initialTurnNumber + (int64_t)moveHistory.size();
}

void BoardHistory::makeBoardMoveAssumeLegal(Board& board, Loc moveLoc, Player movePla, const KoHashTable* rootKoHashTable) {
  makeBoardMoveAssumeLegal(board, moveLoc, movePla, rootKoHashTable, false);
}

void BoardHistory::makeBoardMoveAssumeLegal(Board& board, Loc moveLoc, Player movePla, const KoHashTable*, bool) {
  assert(movePla == board.nextPla());
  board.playMoveAssumeLegal(moveLoc, movePla);
  moveHistory.push_back(Move(moveLoc, movePla));
  preventEncoreHistory.push_back(false);
  if(movePla == P_WHITE)
    whiteHasMoved = true;
  currentRecentBoardIdx = (currentRecentBoardIdx + 1) % NUM_RECENT_BOARDS;
  recentBoards[currentRecentBoardIdx] = board;
  presumedNextMovePla = board.nextPla();
  numTurnsThisPhase += 1;
  numApproxValidTurnsThisPhase += 1;
  numConsecValidTurnsThisGame += 1;
  koHashHistory.push_back(getSituationAndSimpleKoHash(board,presumedNextMovePla));
  if(board.isGameOver())
    endAndScoreGameNow(board);
}

bool BoardHistory::makeBoardMoveTolerant(Board& board, Loc moveLoc, Player movePla) {
  return makeBoardMoveTolerant(board, moveLoc, movePla, false);
}

bool BoardHistory::makeBoardMoveTolerant(Board& board, Loc moveLoc, Player movePla, bool preventEncore) {
  if(!isLegalTolerant(board, moveLoc, movePla))
    return false;
  makeBoardMoveAssumeLegal(board, moveLoc, movePla, NULL, preventEncore);
  return true;
}

void BoardHistory::endGameIfAllPassAlive(const Board&) {}

void BoardHistory::setFinalScoreAndWinner(float score) {
  finalWhiteMinusBlackScore = score;
  if(score > 0.0f)
    winner = P_WHITE;
  else if(score < 0.0f)
    winner = P_BLACK;
  else
    winner = C_EMPTY;
}

void BoardHistory::endAndScoreGameNow(const Board& board) {
  //Red (White) minus Blue (Black). The engine's scores are exact to its 1e-9 rounding, so this sign is the
  //sign the engine's result uses.
  //Komi (owner, 2026-10-04) is added to Red's final score: the winner is the sign of the margin with it.
  double margin = board.getScore(P_WHITE) - board.getScore(P_BLACK) + rules.komi;
  setFinalScoreAndWinner(margin > 0 ? std::max((float)margin, 1e-9f) : margin < 0 ? std::min((float)margin, -1e-9f) : 0.0f);
  isScored = true;
  isNoResult = false;
  isResignation = false;
  isGameFinished = true;
  isPastNormalPhaseEnd = false;
}

void BoardHistory::endAndScoreGameNow(const Board& board, Color area[Board::MAX_ARR_SIZE]) {
  getAreaNow(board, area);
  endAndScoreGameNow(board);
}

void BoardHistory::getAreaNow(const Board& board, Color area[Board::MAX_ARR_SIZE]) const {
  board.calculateArea(area, false, false, false, false);
}

void BoardHistory::setWinnerByResignation(Player pla) {
  isGameFinished = true;
  isPastNormalPhaseEnd = false;
  isScored = false;
  isNoResult = false;
  isResignation = true;
  winner = pla;
  finalWhiteMinusBlackScore = 0.0f;
}

void BoardHistory::printBasicInfo(ostream& out, const Board& board) const {
  Board::printBoard(out, board, Board::NULL_LOC, &moveHistory);
  out << "Next player: " << PlayerIO::playerToString(presumedNextMovePla) << endl;
  if(isGameFinished)
    out << "Game over, winner " << PlayerIO::playerToString(winner) << ", red minus blue " << finalWhiteMinusBlackScore << endl;
}

void BoardHistory::printDebugInfo(ostream& out, const Board& board) const {
  printBasicInfo(out, board);
  out << "Moves:";
  for(const Move& m : moveHistory)
    out << " " << PlayerIO::playerToStringShort(m.pla) << ":" << Location::toString(m.loc, board);
  out << endl;
}

int BoardHistory::numberOfKoHashOccurrencesInHistory(Hash128 koHash, const KoHashTable*) const {
  int count = 0;
  for(const Hash128& h : koHashHistory)
    if(h == koHash)
      count++;
  return count;
}

int BoardHistory::numHandicapStonesOnBoard(const Board&) { return 0; }
int BoardHistory::computeNumHandicapStones() const { return 0; }
int BoardHistory::computeWhiteHandicapBonus() const { return 0; }
bool BoardHistory::hasBlackPassOrWhiteFirst() const { return false; }

Hash128 BoardHistory::getSituationAndSimpleKoHash(const Board& board, Player nextPlayer) {
  return board.pos_hash ^ Board::ZOBRIST_PLAYER_HASH[nextPlayer];
}

Hash128 BoardHistory::getSituationAndSimpleKoAndPrevPosHash(const Board& board, const BoardHistory& hist, Player nextPlayer) {
  Hash128 h = getSituationAndSimpleKoHash(board, nextPlayer);
  const Board& prev = hist.getRecentBoard(1);
  h.hash0 ^= Hash::murmurMix(prev.pos_hash.hash0);
  h.hash1 ^= Hash::splitMix64(prev.pos_hash.hash1);
  return h;
}

//The edges of the earlier turns the network sees (Position::history), mixed by turn.
static Hash128 historyHash(const enclosure::Position& p) {
  Hash128 h;
  for(int t = 0; t < enclosure::HISTORY_TURNS; t++) {
    for(int i = 0; i < p.history_n[t]; i++) {
      const enclosure::Edge& e = p.history[t][i];
      uint64_t key = (uint64_t)((e.ay * enclosure::SIZE + e.ax) * enclosure::NOFF +
                                enclosure::OFF_INDEX[e.bx - e.ax + 3][e.by - e.ay + 3]) * 8 + (uint64_t)t;
      h.hash0 ^= Hash::splitMix64(key ^ 0xD6E8FEB86659FD93ULL);
      h.hash1 ^= Hash::murmurMix(key + 0xA0761D6478BD642FULL);
    }
  }
  return h;
}

Hash128 BoardHistory::getSituationRulesAndKoHash(const Board& board, const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite) {
  return getSituationRulesAndKoHash(board, hist, nextPlayer, drawEquivalentWinsForWhite, hist.modes);
}

Hash128 BoardHistory::getSituationRulesAndKoHash(const Board& board, const BoardHistory& hist, Player nextPlayer, double, const BoardHistoryModes&) {
  Hash128 hash = getSituationAndSimpleKoHash(board, nextPlayer) ^ historyHash(board.pos);
  //Komi changes the network's inputs and the result (owner, 2026-10-04), so evaluations at different komis must not
  //share a cache entry. Folded in only when nonzero: every hash at komi 0 is as before.
  if(hist.rules.komi != 0.0f) {
    uint32_t bits;
    float komi = hist.rules.komi;
    std::memcpy(&bits, &komi, sizeof(bits));
    hash ^= Hash128(Hash::murmurMix(0x6b6f6d6900000000ULL | bits), Hash::splitMix64(0x4b4f4d4900000000ULL | bits));
  }
  return hash;
}

//KO HASH TABLE: always empty (positions never repeat)---------------------------------------

KoHashTable::KoHashTable()
  : koHashHistorySortedByLowBits(),
    firstTurnIdxWithKoHistory(0)
{
  idxTable = new uint32_t[TABLE_SIZE];
  std::fill(idxTable, idxTable+TABLE_SIZE, (uint32_t)0);
}
KoHashTable::~KoHashTable() {
  delete[] idxTable;
}

size_t KoHashTable::size() const {
  return koHashHistorySortedByLowBits.size();
}

void KoHashTable::recompute(const BoardHistory& history) {
  (void)history;
  koHashHistorySortedByLowBits.clear();
  firstTurnIdxWithKoHistory = 0;
  std::fill(idxTable, idxTable+TABLE_SIZE, (uint32_t)0);
}

bool KoHashTable::containsHash(Hash128) const {
  return false;
}

int KoHashTable::numberOfOccurrencesOfHash(Hash128) const {
  return 0;
}
