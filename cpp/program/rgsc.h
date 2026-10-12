#ifndef PROGRAM_RGSC_H_
#define PROGRAM_RGSC_H_

//Enclosure fork (owner, 2026-10-04): Regret-Guided Search Control (Tsai et al., ICLR 2026; the authors' MiniZero code,
//minizero/actor/zero_actor.cpp), in KataGo's self-play. See v2/docs/decisions/rgsc.md, "RGSC: the authors' code
//read again", for the owner's decisions:
// - a game's regret target (Eq. 2): at each of its positions, the mean over that position and every later one of
//   (the search value of the move played - the game's outcome)^2, every move counting (full and cheap searches);
// - the game's candidate (games from the normal start only): the state with the highest regret ranking among every
//   state the game's searches evaluated (its regret from the regret value head, at least the buffer's lowest) and
//   every played position (its regret from Eq. 2), skipping states already in the buffer;
// - a buffer of the highest-regret states (500), shared by the process's game threads and saved across the loop's
//   restarts; a candidate enters if there is room or it beats the lowest, which it replaces;
// - with probability 0.5 a game starts from a buffer state, drawn with probability R^(1/tau) / sum (tau 0.1, the
//   paper's rule); after the game the state's regret becomes (1 - alpha) old + alpha new (alpha 0.5), and a game
//   whose start state left the buffer meanwhile is dropped, as in the authors' code;
// - the first 1,000 games (the warm-up) fill the buffer and are kept out of the training data.
//
//Enclosure fork (2026-10-11): a second source, value swing (v2/docs/decisions/rgsc.md, "RGSC with two sources"; not
//published, a named departure). Its regret at a position is sum_k gamma^k (v_{t+k+1} - v_{t+k})^2 / sum_k gamma^k
//over the rest of the game (v the search value of the move played, the outcome as the value after the last move),
//times a scale that brings it to the vanilla regret's range, capped at 4 (the regret value head's range). It has its
//own two heads (the swing value and the swing ranking), its own tracker and its own buffer with the same rules,
//except that its warm-up drops no game: its restarts begin when its buffer is full. Each game draws at most one
//restart: from the vanilla buffer with rgscStartProb, else from the swing buffer with rgscSwingStartProb.

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "../core/config_parser.h"
#include "../core/rand.h"
#include "../game/boardhistory.h"

struct NNOutput;

namespace Rgsc {

  //The buffer a game restarted from (C74 of its rows).
  enum Source { SOURCE_NONE = 0, SOURCE_VANILLA = 1, SOURCE_SWING = 2 };
  //Where a buffer state came from: a state that a search evaluated (the tracker's candidate), or a played position.
  //Unknown for states saved before 2026-10-11.
  enum Origin { ORIGIN_UNKNOWN = 0, ORIGIN_TREE = 1, ORIGIN_PLAYED = 2 };

  struct Settings {
    bool enabled = false;
    std::string bufferFile;
    int bufferSize = 500;
    double startProb = 0.5;
    double temperature = 0.1;
    double emaAlpha = 0.5;
    int64_t warmupGames = 1000;
    //A buffer that draws only when it holds bufferSize states (the swing buffer).
    bool sampleOnlyWhenFull = false;
    //The swing source (2026-10-11). Its targets are written whenever RGSC is on; its buffer exists only when
    //swingStartProb > 0.
    double swingStartProb = 0.0;
    std::string swingBufferFile;
    double swingGamma = 0.9;
    double swingScale = 50.0;
    double swingCap = 4.0;
    static Settings load(ConfigParser& cfg);
    //The swing buffer's settings: its file and start probability, no warm-up, draws only when full.
    Settings forSwingBuffer() const;
  };

  //A position's key: the moves that reach it, each with its player, combined so that the order does not matter
  //(the authors' key; in Enclosure a position is the edges each player has placed).
  uint64_t keyOf(const std::vector<Move>& moves, size_t count);

  struct Entry {
    uint64_t key = 0;
    double regret = 0.0;
    int64_t sampleCount = 0;
    int origin = ORIGIN_UNKNOWN;
    std::vector<Move> moves;  //from the empty board
  };

  class Buffer {
   public:
    //name: how the log names it ("RGSC", "RGSC swing").
    Buffer(const Settings& settings, const std::string& name = "RGSC");

    //Loads the buffer and the finished-game count from the settings' file, if it exists.
    void load();
    //Saves them (a temporary file, then a rename).
    void save() const;

    //With probability startProb (and a non-empty buffer), draws a start state; false otherwise.
    bool maybeSample(Rand& rand, Entry& out);
    //Draws a start state by the paper's rule; false if the buffer is empty (or not full, with sampleOnlyWhenFull).
    bool sample(Rand& rand, Entry& out);
    bool contains(uint64_t key) const;
    //The lowest regret in a full buffer, else 0: the floor for a tree candidate's estimated regret.
    double floorRegret() const;
    //A game's candidate: enters if there is room or its regret beats the lowest, which it replaces.
    void offer(uint64_t key, double regret, const std::vector<Move>& moves, int origin = ORIGIN_UNKNOWN);
    //After a game started from the state with this key: the EMA update. False if the state left the buffer.
    bool updateAfterRestart(uint64_t key, double regret);
    //Counts a finished game; returns how many had finished before it (for the warm-up).
    int64_t countFinishedGame();
    size_t size() const;
    //A one-line summary for the log: states and their regrets and origins, and this run's counts (restarts drawn,
    //written after their state left the buffer, and dropped; candidates offered and taken; sample attempts while
    //the buffer was empty, or not yet full).
    std::string summary() const;
    void countDroppedRestart();
    void countRestartStateGone();

   private:
    //The paper's draw, with the mutex held: counts the restart and the entry's draws.
    void drawLocked(Rand& rand, Entry& out);

    Settings settings;
    std::string name;
    mutable std::mutex mutex;
    std::vector<Entry> entries;
    std::unordered_set<uint64_t> keys;
    int64_t gamesFinished;
    int64_t restartsDrawn = 0;
    int64_t restartsDropped = 0;
    int64_t restartsStateGone = 0;
    int64_t candidatesOffered = 0;
    int64_t candidatesTaken = 0;
    int64_t drawsWhileEmpty = 0;
  };

  //Fed every network evaluation of one game's searches (Search::rgscTracker, rgscSwingTracker): keeps the state
  //that the source's ranking head ranks highest.
  class TreeTracker {
   public:
    TreeTracker(const Buffer* buffer, Source source = SOURCE_VANILLA);
    void offer(const NNOutput& nnOutput, const BoardHistory& hist);
    bool found() const { return hasCandidate; }
    float ranking() const { return bestRanking.load(std::memory_order_acquire); }
    float regretValue() const { return candidateRegretValue; }
    uint64_t key() const { return candidateKey; }
    const std::vector<Move>& moves() const { return candidateMoves; }

   private:
    const Buffer* buffer;
    Source source;
    std::mutex mutex;
    std::atomic<float> bestRanking;
    bool hasCandidate;
    float candidateRegretValue;
    uint64_t candidateKey;
    std::vector<Move> candidateMoves;
  };

  //Eq. 2 for each turn: selectedWhiteWinLoss[t] is the search value (white's win minus loss) of the move played at
  //turn t, whiteOutcome the game's (+1 white won, -1 black won, 0 otherwise).
  std::vector<float> regretsOfGame(const std::vector<float>& selectedWhiteWinLoss, double whiteOutcome);
  //The swing regret for each turn (2026-10-11): min(cap, scale * sum_k gamma^k (v_{t+k+1} - v_{t+k})^2 /
  //sum_k gamma^k), k from 0 to the last turn, with v the same search values and whiteOutcome after the last move.
  std::vector<float> swingsOfGame(
    const std::vector<float>& selectedWhiteWinLoss, double whiteOutcome, double gamma, double scale, double cap);
}

#endif  // PROGRAM_RGSC_H_
