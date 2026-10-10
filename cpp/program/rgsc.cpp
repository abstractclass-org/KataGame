#include "../program/rgsc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>

#include "../core/global.h"
#include "../core/hash.h"
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

uint64_t Rgsc::keyOf(const vector<Move>& moves, size_t count) {
  uint64_t key = 0;
  for(size_t i = 0; i < count; i++)
    key ^= Hash::murmurMix(((uint64_t)(uint32_t)moves[i].loc << 8) ^ (uint64_t)moves[i].pla ^ 0x5be8a1c3d2f7e649ULL);
  return key;
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
      //The state's origin, an optional last token (files saved before 2026-10-11 have none).
      int origin;
      if(row >> origin)
        e.origin = origin;
      e.key = keyOf(e.moves, e.moves.size());
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
      out << "entry " << e.regret << " " << e.sampleCount << " " << e.moves.size();
      for(const Move& m : e.moves)
        out << " " << (int)m.loc << " " << (int)m.pla;
      out << " " << e.origin << "\n";
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
  lock_guard<std::mutex> lock(mutex);
  candidatesOffered++;
  if(keys.count(key) > 0)
    return;
  Entry e;
  e.key = key;
  e.regret = regret;
  e.origin = origin;
  e.moves = moves;
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
  for(const Entry& e : entries) {
    r.push_back(e.regret);
    fromTree += e.origin == ORIGIN_TREE;
    fromPlayed += e.origin == ORIGIN_PLAYED;
  }
  std::sort(r.begin(), r.end());
  ostringstream out;
  out << name << ": " << entries.size() << " states";
  if(!r.empty())
    out << ", regret min " << r.front() << " median " << r[r.size() / 2] << " max " << r.back()
        << ", from search trees " << fromTree << ", played " << fromPlayed;
  out << "; games finished " << gamesFinished << "; this run: restarts drawn " << restartsDrawn
      << ", written after their state left " << restartsStateGone << ", dropped " << restartsDropped
      << "; candidates offered " << candidatesOffered << ", taken " << candidatesTaken
      << "; draws while empty " << drawsWhileEmpty;
  return out.str();
}

//------------------------------------------------------------------------------------------------------------

Rgsc::TreeTracker::TreeTracker(const Buffer* b, Source s)
  : buffer(b), source(s), mutex(), bestRanking(-numeric_limits<float>::infinity()), hasCandidate(false),
    candidateRegretValue(0.0f), candidateKey(0), candidateMoves()
{}

void Rgsc::TreeTracker::offer(const NNOutput& nnOutput, const BoardHistory& hist) {
  //As the authors' actor: every evaluated, non-terminal state but the initial one. The swing tracker reads the
  //swing heads.
  const bool swing = source == SOURCE_SWING;
  if(!(swing ? nnOutput.hasSwing : nnOutput.hasRegret) || hist.moveHistory.empty() || hist.isGameFinished)
    return;
  const float ranking = swing ? nnOutput.swingRanking : nnOutput.regretRanking;
  if(!(ranking > bestRanking.load(std::memory_order_acquire)))
    return;
  uint64_t key = keyOf(hist.moveHistory, hist.moveHistory.size());
  if(buffer->contains(key))
    return;
  lock_guard<std::mutex> lock(mutex);
  if(!(ranking > bestRanking.load(std::memory_order_acquire)))
    return;
  hasCandidate = true;
  candidateRegretValue = swing ? nnOutput.swingValue : nnOutput.regretValue;
  candidateKey = key;
  candidateMoves = hist.moveHistory;
  bestRanking.store(ranking, std::memory_order_release);
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
