#include "../program/rgsc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/test.h"
#include "../neuralnet/nninputs.h"

using namespace std;

Rgsc::Settings Rgsc::Settings::load(ConfigParser& cfg) {
  Settings s;
  s.enabled = cfg.contains("rgscEnabled") ? cfg.getBool("rgscEnabled") : false;
  //Read every key either way, so that a config with RGSC off raises no unused-key warnings.
  if(s.enabled || cfg.contains("rgscBufferFile"))
    s.bufferFile = cfg.getString("rgscBufferFile");
  if(cfg.contains("rgscBufferSize")) s.bufferSize = cfg.getInt("rgscBufferSize", 1, 1000000);
  if(cfg.contains("rgscStartProb")) s.startProb = cfg.getDouble("rgscStartProb", 0.0, 1.0);
  if(cfg.contains("rgscTemperature")) s.temperature = cfg.getDouble("rgscTemperature", 1e-3, 100.0);
  if(cfg.contains("rgscEmaAlpha")) s.emaAlpha = cfg.getDouble("rgscEmaAlpha", 0.0, 1.0);
  if(cfg.contains("rgscWarmupGames")) s.warmupGames = cfg.getInt64("rgscWarmupGames", 0, (int64_t)1 << 40);
  //The swing source (2026-10-11).
  if(cfg.contains("rgscSwingStartProb")) s.swingStartProb = cfg.getDouble("rgscSwingStartProb", 0.0, 1.0);
  if((s.enabled && s.swingStartProb > 0) || cfg.contains("rgscSwingBufferFile"))
    s.swingBufferFile = cfg.getString("rgscSwingBufferFile");
  if(cfg.contains("rgscSwingGamma")) s.swingGamma = cfg.getDouble("rgscSwingGamma", 0.0, 1.0);
  if(cfg.contains("rgscSwingScale")) s.swingScale = cfg.getDouble("rgscSwingScale", 0.0, 1e6);
  if(cfg.contains("rgscSwingCap")) s.swingCap = cfg.getDouble("rgscSwingCap", 0.0, 4.0);
  if(s.startProb + s.swingStartProb > 1.0)
    throw StringError("RGSC: rgscStartProb + rgscSwingStartProb must be at most 1");
  if(s.swingStartProb > 0 && s.swingBufferFile == s.bufferFile)
    throw StringError("RGSC: the swing buffer needs its own file (rgscSwingBufferFile)");
  if(s.swingStartProb > 0 && !(s.swingScale > 0))
    throw StringError("RGSC: the swing buffer needs swing targets (rgscSwingScale > 0)");
  return s;
}

Rgsc::Settings Rgsc::Settings::forSwingBuffer() const {
  Settings s = *this;
  s.bufferFile = swingBufferFile;
  s.startProb = swingStartProb;
  s.warmupGames = 0;
  s.sampleOnlyWhenFull = true;
  return s;
}

//One move's part of keyOf.
static uint64_t moveKey(const Move& m) {
  return Hash::murmurMix(((uint64_t)(uint32_t)m.loc << 8) ^ (uint64_t)m.pla ^ 0x5be8a1c3d2f7e649ULL);
}

uint64_t Rgsc::keyOf(const vector<Move>& moves, size_t count) {
  uint64_t key = 0;
  for(size_t i = 0; i < count; i++)
    key ^= moveKey(moves[i]);
  return key;
}

//------------------------------------------------------------------------------------------------------------
//KataGame fork (2026-10-27, abscls-luzl.7): the states of a game that keeps states.

Rgsc::Context Rgsc::Context::of(float komi, Player advantagePla, double advantage) {
  Context c;
  c.komi = komi == 0.0f ? 0.0f : komi;
  c.advantage = advantage == 0.0 ? 0.0 : advantage;
  c.advantagePla = c.advantage == 0.0 ? C_EMPTY : advantagePla;
  return c;
}

bool Rgsc::Context::operator==(const Context& other) const {
  return komi == other.komi && advantagePla == other.advantagePla && advantage == other.advantage;
}

uint64_t Rgsc::stateKey(const Board& board, const Context& context) {
  //What keys the network's cache (BoardHistory::getSituationRulesAndKoHash): the position with the player to move, and
  //what the inputs read beyond it. Then the context, bit for bit.
  const Hash128 h = board.pos_hash ^ GameEncoding::input_hash(board.pos);
  uint32_t komiBits;
  std::memcpy(&komiBits, &context.komi, sizeof(komiBits));
  uint64_t advantageBits;
  std::memcpy(&advantageBits, &context.advantage, sizeof(advantageBits));
  uint64_t key = Hash::murmurMix(h.hash0 ^ 0x2f6b1d0e9c4a8357ULL) ^ Hash::splitMix64(h.hash1);
  key = Hash::murmurMix(key ^ ((uint64_t)komiBits << 8) ^ (uint64_t)(uint8_t)context.advantagePla);
  return Hash::splitMix64(key ^ advantageBits);
}

Rgsc::Entry Rgsc::stateEntry(const Board& board, const Context& context, int64_t turnNumber) {
  Entry e;
  e.key = stateKey(board, context);
  e.state = GameInterface::stateText<GameRules>(board.pos);
  e.context = context;
  e.turnNumber = turnNumber;
  return e;
}

bool Rgsc::restoreState(const Entry& entry, Board& board, string& why) {
  static thread_local GameRules::Position p;
  if(!GameInterface::parseState<GameRules>(entry.state, p, why))
    return false;
  board.setPosition(p);
  const uint64_t key = stateKey(board, entry.context);
  if(key != entry.key) {
    why = "the state's key is " + Global::uint64ToHexString(key) + ", not the entry's " +
          Global::uint64ToHexString(entry.key);
    return false;
  }
  return true;
}

//The board of a game after `length` of hist's moves from its first position (hist.initialBoard), replayed.
static Board playedBoard(const BoardHistory& hist, size_t length) {
  Board board = hist.initialBoard;
  for(size_t i = 0; i < length; i++)
    board.playMoveAssumeLegal(hist.moveHistory[i].loc, hist.moveHistory[i].pla);
  return board;
}

vector<uint64_t> Rgsc::playedKeys(const BoardHistory& hist, size_t start, size_t count, const Context& context) {
  testAssert(start + count <= hist.moveHistory.size());
  vector<uint64_t> keys(count);
  if(!keepsStates()) {
    uint64_t key = keyOf(hist.moveHistory, start);
    for(size_t k = 0; k < count; k++) {
      keys[k] = key;
      key ^= moveKey(hist.moveHistory[start + k]);
    }
    return keys;
  }
  Board board = playedBoard(hist, start);
  for(size_t k = 0; k < count; k++) {
    keys[k] = stateKey(board, context);
    if(k + 1 < count)
      board.playMoveAssumeLegal(hist.moveHistory[start + k].loc, hist.moveHistory[start + k].pla);
  }
  return keys;
}

Rgsc::Entry Rgsc::playedEntry(const BoardHistory& hist, size_t length, const Context& context) {
  testAssert(length <= hist.moveHistory.size());
  if(keepsStates())
    return stateEntry(playedBoard(hist, length), context, hist.initialTurnNumber + (int64_t)length);
  Entry e;
  e.key = keyOf(hist.moveHistory, length);
  e.moves.assign(hist.moveHistory.begin(), hist.moveHistory.begin() + length);
  return e;
}

bool Rgsc::startOf(const Entry& entry, int xSize, int ySize, Sgf::PositionSample& start, string& why) {
  start.hintLoc = Board::NULL_LOC;
  start.weight = 1.0;
  start.trainingWeight = 1.0;
  start.exactStart = true;
  if(!keepsStates()) {
    start.board = Board(xSize, ySize);
    start.nextPla = P_BLACK;
    start.moves = entry.moves;
    start.initialTurnNumber = 0;
    return true;
  }
  if(!restoreState(entry, start.board, why))
    return false;
  start.nextPla = start.board.nextPla();
  start.moves.clear();
  start.initialTurnNumber = entry.turnNumber;
  start.frozenContext = true;
  start.komi = entry.context.komi;
  start.playoutDoublingAdvantagePla = entry.context.advantagePla;
  start.playoutDoublingAdvantage = entry.context.advantage;
  return true;
}

//------------------------------------------------------------------------------------------------------------

Rgsc::Buffer::Buffer(const Settings& s, const string& n)
  : settings(s), name(n), mutex(), entries(), keys(), gamesFinished(0)
{}

void Rgsc::Buffer::load() {
  lock_guard<std::mutex> lock(mutex);
  ifstream in(settings.bufferFile);
  if(!in)
    return;
  string line, word;
  Board restored;  //each state's, for its check
  entries.clear();
  keys.clear();
  while(getline(in, line)) {
    istringstream row(line);
    if(!(row >> word))
      continue;
    if(word == "gamesFinished") {
      row >> gamesFinished;
    }
    else if(word == "entry") {
      //A game that keeps states cannot replay moves exactly: it rejects them.
      if(keepsStates()) {
        rejectedAtLoad++;
        continue;
      }
      Entry e;
      size_t n = 0;
      row >> e.regret >> e.sampleCount >> n;
      for(size_t i = 0; i < n; i++) {
        int loc, pla;
        row >> loc >> pla;
        e.moves.push_back(Move((Loc)loc, (Player)pla));
      }
      if(row.fail())
        throw StringError("RGSC buffer file " + settings.bufferFile + ": unreadable entry");
      //The state's origin, an optional token (files saved before 2026-10-11 have none), then its member and side
      //(KataGame fork, 2026-10-29: files saved before have none).
      int origin, member, side;
      if(row >> origin)
        e.origin = origin;
      if(row >> member >> side) {
        e.member = member;
        e.side = side;
      }
      e.key = keyOf(e.moves, e.moves.size());
      if(keys.insert(e.key).second)
        entries.push_back(std::move(e));
    }
    else if(word == "state") {
      //KataGame fork (2026-10-27): "state VERSION GAME KEY REGRET SAMPLES ORIGIN MEMBER SIDE TURN KOMI ADVANTAGEPLA
      //ADVANTAGE" and the state's text (save). Version 1 has no MEMBER SIDE (2026-10-29), and its states load with
      //both unknown. An entry of another version or game, or whose state does not come back with its key, is
      //rejected.
      Entry e;
      int version = 0, advantagePla = 0;
      string game, key, text;
      float komi = 0.0f;
      double advantage = 0.0;
      row >> version >> game >> key >> e.regret >> e.sampleCount >> e.origin;
      if(version == STATE_RECORD_VERSION)
        row >> e.member >> e.side;
      row >> e.turnNumber >> komi >> advantagePla >> advantage;
      getline(row, text);
      string why;
      bool ok = keepsStates() && !row.fail() && (version == 1 || version == STATE_RECORD_VERSION) &&
                game == GameRules::NAME && text.size() > 1 && text[0] == ' ' &&
                Global::tryHexStringToUInt64(key, e.key);
      if(ok) {
        e.state = text.substr(1);
        e.context = Context::of(komi, (Player)advantagePla, advantage);
        ok = restoreState(e, restored, why);
      }
      if(!ok) {
        rejectedAtLoad++;
        continue;
      }
      if(keys.insert(e.key).second)
        entries.push_back(std::move(e));
    }
  }
}

void Rgsc::Buffer::save() const {
  lock_guard<std::mutex> lock(mutex);
  string tmp = settings.bufferFile + ".tmp";
  {
    ofstream out(tmp);
    if(!out)
      throw StringError("RGSC: cannot write " + tmp);
    out.precision(9);
    out << "gamesFinished " << gamesFinished << "\n";
    for(const Entry& e : entries) {
      if(!e.state.empty()) {
        //A state (keepsStates), its advantage with every digit, so that its key comes back. KataGame fork (2026-10-29):
        //version 2, with the member and the side after the origin.
        ostringstream advantage;
        advantage.precision(17);
        advantage << e.context.advantage;
        out << "state " << STATE_RECORD_VERSION << " " << GameRules::NAME << " " << Global::uint64ToHexString(e.key)
            << " " << e.regret << " " << e.sampleCount << " " << e.origin << " " << e.member << " " << e.side << " "
            << e.turnNumber << " " << e.context.komi << " " << (int)e.context.advantagePla << " " << advantage.str()
            << " " << e.state << "\n";
        continue;
      }
      out << "entry " << e.regret << " " << e.sampleCount << " " << e.moves.size();
      for(const Move& m : e.moves)
        out << " " << (int)m.loc << " " << (int)m.pla;
      out << " " << e.origin << " " << e.member << " " << e.side << "\n";  //KataGame fork (2026-10-29): member, side
    }
  }
  if(std::rename(tmp.c_str(), settings.bufferFile.c_str()) != 0)
    throw StringError("RGSC: cannot rename " + tmp);
}

bool Rgsc::Buffer::maybeSample(Rand& rand, Entry& out) {
  lock_guard<std::mutex> lock(mutex);
  if(entries.empty()) {
    drawsWhileEmpty++;
    return false;
  }
  if(!rand.nextBool(settings.startProb))
    return false;
  drawLocked(rand, out);
  return true;
}

bool Rgsc::Buffer::sample(Rand& rand, Entry& out) {
  lock_guard<std::mutex> lock(mutex);
  if(entries.empty() || (settings.sampleOnlyWhenFull && (int)entries.size() < settings.bufferSize)) {
    drawsWhileEmpty++;
    return false;
  }
  drawLocked(rand, out);
  return true;
}

void Rgsc::Buffer::drawLocked(Rand& rand, Entry& out) {
  restartsDrawn++;
  //The paper's rule, P(i) = R_i^(1/tau) / sum_j R_j^(1/tau), in logs so that tau 0.1 (R^10) does not underflow.
  double maxLog = -numeric_limits<double>::infinity();
  for(const Entry& e : entries)
    if(e.regret > 0)
      maxLog = std::max(maxLog, std::log(e.regret));
  vector<double> weights(entries.size());
  double total = 0.0;
  for(size_t i = 0; i < entries.size(); i++) {
    weights[i] = entries[i].regret > 0 ? std::exp((std::log(entries[i].regret) - maxLog) / settings.temperature) : 0.0;
    total += weights[i];
  }
  size_t chosen = 0;
  if(!(total > 0))
    chosen = (size_t)rand.nextUInt64(entries.size());  //every regret 0: uniformly
  else {
    double r = rand.nextDouble() * total;
    while(chosen + 1 < entries.size() && r >= weights[chosen]) {
      r -= weights[chosen];
      chosen++;
    }
  }
  entries[chosen].sampleCount += 1;
  out = entries[chosen];
}

bool Rgsc::Buffer::contains(uint64_t key) const {
  lock_guard<std::mutex> lock(mutex);
  return keys.count(key) > 0;
}

double Rgsc::Buffer::floorRegret() const {
  lock_guard<std::mutex> lock(mutex);
  if((int)entries.size() < settings.bufferSize)
    return 0.0;
  double lowest = numeric_limits<double>::infinity();
  for(const Entry& e : entries)
    lowest = std::min(lowest, e.regret);
  return lowest;
}

void Rgsc::Buffer::offer(uint64_t key, double regret, const vector<Move>& moves, int origin) {
  Entry e;
  e.key = key;
  e.regret = regret;
  e.origin = origin;
  e.moves = moves;
  offer(std::move(e));
}

void Rgsc::Buffer::offer(Entry e) {
  lock_guard<std::mutex> lock(mutex);
  candidatesOffered++;
  if(keys.count(e.key) > 0) {
    //KataGame fork (2026-10-27): the whole state's check. The same key with another state or context is a
    //collision of the hash.
    if(!e.state.empty()) {
      const auto known =
        std::find_if(entries.begin(), entries.end(), [&e](const Entry& k) { return k.key == e.key; });
      if(known != entries.end() && !(known->state == e.state && known->context == e.context))
        keyCollisions++;
    }
    return;
  }
  const uint64_t key = e.key;
  const double regret = e.regret;
  if((int)entries.size() < settings.bufferSize) {
    keys.insert(key);
    entries.push_back(std::move(e));
    candidatesTaken++;
    return;
  }
  size_t lowest = 0;
  for(size_t i = 1; i < entries.size(); i++)
    if(entries[i].regret < entries[lowest].regret)
      lowest = i;
  if(regret > entries[lowest].regret) {
    keys.erase(entries[lowest].key);
    keys.insert(key);
    entries[lowest] = std::move(e);
    candidatesTaken++;
  }
}

void Rgsc::Buffer::reject(uint64_t key) {
  lock_guard<std::mutex> lock(mutex);
  rejectedAtRestart++;
  for(size_t i = 0; i < entries.size(); i++)
    if(entries[i].key == key) {
      keys.erase(key);
      entries.erase(entries.begin() + i);
      return;
    }
}

bool Rgsc::Buffer::updateAfterRestart(uint64_t key, double regret) {
  lock_guard<std::mutex> lock(mutex);
  for(Entry& e : entries) {
    if(e.key == key) {
      e.regret = (1.0 - settings.emaAlpha) * e.regret + settings.emaAlpha * regret;
      return true;
    }
  }
  return false;
}

int64_t Rgsc::Buffer::countFinishedGame() {
  lock_guard<std::mutex> lock(mutex);
  return gamesFinished++;
}

void Rgsc::Buffer::addRegrets(const vector<float>& regrets, const vector<bool>& continuesMove) {
  lock_guard<std::mutex> lock(mutex);
  const size_t n = regrets.size();
  //Without the kinds, every action is a move.
  const bool known = continuesMove.size() == n;
  for(size_t k = 0; k < n; k++) {
    //0 a move of one action, 1 a move's first action, 2 a later action.
    const int kind = known && continuesMove[k] ? 2 : known && k + 1 < n && continuesMove[k + 1] ? 1 : 0;
    regretCount[kind]++;
    regretSum[kind] += regrets[k];
  }
}

size_t Rgsc::Buffer::size() const {
  lock_guard<std::mutex> lock(mutex);
  return entries.size();
}

void Rgsc::Buffer::countDroppedRestart() {
  lock_guard<std::mutex> lock(mutex);
  restartsDropped++;
}

void Rgsc::Buffer::countRestartStateGone() {
  lock_guard<std::mutex> lock(mutex);
  restartsStateGone++;
}

string Rgsc::Buffer::summary() const {
  lock_guard<std::mutex> lock(mutex);
  vector<double> r;
  int64_t fromTree = 0, fromPlayed = 0;
  std::map<int, int64_t> byMember;  //KataGame fork (2026-10-29): the states of each producing member (Entry::member)
  for(const Entry& e : entries) {
    r.push_back(e.regret);
    fromTree += e.origin == ORIGIN_TREE;
    fromPlayed += e.origin == ORIGIN_PLAYED;
    byMember[e.member]++;
  }
  std::sort(r.begin(), r.end());
  ostringstream out;
  out << name << ": " << entries.size() << " states";
  if(!r.empty()) {
    out << ", regret min " << r.front() << " median " << r[r.size() / 2] << " max " << r.back()
        << ", from search trees " << fromTree << ", played " << fromPlayed << ", by member";
    for(const auto& [member, count] : byMember)
      out << " " << (member == PRODUCER_UNKNOWN ? string("unknown") : Global::intToString(member)) << ":" << count;
  }
  out << "; games finished " << gamesFinished << "; this run: restarts drawn " << restartsDrawn
      << ", written after their state left " << restartsStateGone << ", dropped " << restartsDropped
      << "; candidates offered " << candidatesOffered << ", taken " << candidatesTaken
      << "; draws while empty " << drawsWhileEmpty;
  //KataGame fork (2026-10-27): the rejected entries and the collisions of keys, and the regret targets by kind of
  //action.
  out << "; rejected at the load " << rejectedAtLoad << ", at a restart " << rejectedAtRestart << "; key collisions "
      << keyCollisions << "; regret targets:";
  const char* kinds[3] = {"moves of one action", "first actions of a move", "later actions"};
  for(int kind = 0; kind < 3; kind++)
    out << (kind > 0 ? "," : "") << " " << kinds[kind] << " " << regretCount[kind] << " mean "
        << (regretCount[kind] > 0 ? regretSum[kind] / regretCount[kind] : 0.0);
  return out.str();
}

//------------------------------------------------------------------------------------------------------------

Rgsc::TreeTracker::TreeTracker(const Buffer* b, Source s)
  : buffer(b), source(s), mutex(), bestRanking(-numeric_limits<float>::infinity()), hasCandidate(false),
    candidateRegretValue(0.0f), candidateEntry(), candidateBoard()
{}

void Rgsc::TreeTracker::offer(const NNOutput& nnOutput, const Board& board, const BoardHistory& hist,
                              const Context& context, int side) {
  //As the authors' actor: every evaluated, non-terminal state but the initial one. The swing tracker reads the
  //swing heads.
  const bool swing = source == SOURCE_SWING;
  if(!(swing ? nnOutput.hasSwing : nnOutput.hasRegret) || hist.moveHistory.empty() || hist.isGameFinished)
    return;
  const float ranking = swing ? nnOutput.swingRanking : nnOutput.regretRanking;
  if(!(ranking > bestRanking.load(std::memory_order_acquire)))
    return;
  uint64_t key = keepsStates() ? stateKey(board, context) : keyOf(hist.moveHistory, hist.moveHistory.size());
  if(buffer->contains(key))
    return;
  lock_guard<std::mutex> lock(mutex);
  if(!(ranking > bestRanking.load(std::memory_order_acquire)))
    return;
  hasCandidate = true;
  candidateRegretValue = swing ? nnOutput.swingValue : nnOutput.regretValue;
  candidateEntry.key = key;
  candidateEntry.side = side;  //KataGame fork (2026-10-29)
  if(keepsStates()) {
    candidateBoard = board;
    candidateEntry.context = context;
    candidateEntry.turnNumber = hist.getCurrentTurnNumber();
  }
  else
    candidateEntry.moves = hist.moveHistory;
  bestRanking.store(ranking, std::memory_order_release);
}

Rgsc::Entry Rgsc::TreeTracker::candidate() const {
  Entry e = candidateEntry;
  if(keepsStates())
    e.state = GameInterface::stateText<GameRules>(candidateBoard.pos);
  return e;
}

//------------------------------------------------------------------------------------------------------------

vector<float> Rgsc::regretsOfGame(const vector<float>& selectedWhiteWinLoss, double whiteOutcome) {
  const size_t n = selectedWhiteWinLoss.size();
  vector<float> regrets(n);
  double sum = 0.0;
  for(size_t k = 0; k < n; k++) {
    size_t t = n - 1 - k;
    double d = selectedWhiteWinLoss[t] - whiteOutcome;
    sum += d * d;
    regrets[t] = (float)(sum / (double)(k + 1));
  }
  return regrets;
}

vector<float> Rgsc::swingsOfGame(
  const vector<float>& selectedWhiteWinLoss, double whiteOutcome, double gamma, double scale, double cap
) {
  const size_t n = selectedWhiteWinLoss.size();
  vector<float> swings(n);
  double num = 0.0;
  double den = 0.0;
  for(size_t k = 0; k < n; k++) {
    size_t t = n - 1 - k;
    double next = t + 1 < n ? selectedWhiteWinLoss[t + 1] : whiteOutcome;
    double d = next - selectedWhiteWinLoss[t];
    num = d * d + gamma * num;
    den = 1.0 + gamma * den;
    swings[t] = (float)std::min(cap, scale * num / den);
  }
  return swings;
}
