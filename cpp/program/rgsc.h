#ifndef PROGRAM_RGSC_H_
#define PROGRAM_RGSC_H_

//Enclosure fork (owner, 2026-10-04): Regret-Guided Search Control (Tsai et al., ICLR 2026; the authors' MiniZero code,
//minizero/actor/zero_actor.cpp), in KataGo's self-play. See docs/decisions/katago-fork.md, "RGSC: the authors' code
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

  struct Settings {
    bool enabled = false;
    std::string bufferFile;
    int bufferSize = 500;
    double startProb = 0.5;
    double temperature = 0.1;
    double emaAlpha = 0.5;
    int64_t warmupGames = 1000;
    static Settings load(ConfigParser& cfg);
  };

  //A position's key: the moves that reach it, each with its player, combined so that the order does not matter
  //(the authors' key; in Enclosure a position is the edges each player has placed).
  uint64_t keyOf(const std::vector<Move>& moves, size_t count);

  struct Entry {
    uint64_t key = 0;
    double regret = 0.0;
    int64_t sampleCount = 0;
    std::vector<Move> moves;  //from the empty board
  };

  class Buffer {
   public:
    explicit Buffer(const Settings& settings);

    //Loads the buffer and the finished-game count from the settings' file, if it exists.
    void load();
    //Saves them (a temporary file, then a rename).
    void save() const;

    //With probability startProb (and a non-empty buffer), draws a start state; false otherwise.
    bool maybeSample(Rand& rand, Entry& out);
    bool contains(uint64_t key) const;
    //The lowest regret in a full buffer, else 0: the floor for a tree candidate's estimated regret.
    double floorRegret() const;
    //A game's candidate: enters if there is room or its regret beats the lowest, which it replaces.
    void offer(uint64_t key, double regret, const std::vector<Move>& moves);
    //After a game started from the state with this key: the EMA update. False if the state left the buffer.
    bool updateAfterRestart(uint64_t key, double regret);
    //Counts a finished game; returns how many had finished before it (for the warm-up).
    int64_t countFinishedGame();
    size_t size() const;
    //A one-line summary for the log: states and their regrets, and this run's counts (restarts drawn and dropped,
    //candidates offered and taken, sample attempts while the buffer was empty).
    std::string summary() const;
    void countDroppedRestart();

   private:
    Settings settings;
    mutable std::mutex mutex;
    std::vector<Entry> entries;
    std::unordered_set<uint64_t> keys;
    int64_t gamesFinished;
    int64_t restartsDrawn = 0;
    int64_t restartsDropped = 0;
    int64_t candidatesOffered = 0;
    int64_t candidatesTaken = 0;
    int64_t drawsWhileEmpty = 0;
  };

  //Fed every network evaluation of one game's searches (Search::rgscTracker): keeps the highest-ranked state.
  class TreeTracker {
   public:
    explicit TreeTracker(const Buffer* buffer);
    void offer(const NNOutput& nnOutput, const BoardHistory& hist);
    bool found() const { return hasCandidate; }
    float ranking() const { return bestRanking.load(std::memory_order_acquire); }
    float regretValue() const { return candidateRegretValue; }
    uint64_t key() const { return candidateKey; }
    const std::vector<Move>& moves() const { return candidateMoves; }

   private:
    const Buffer* buffer;
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
}

#endif  // PROGRAM_RGSC_H_
