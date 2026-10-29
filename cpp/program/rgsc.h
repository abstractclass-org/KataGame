#ifndef PROGRAM_RGSC_H_
#define PROGRAM_RGSC_H_

//Enclosure fork (owner, 2026-10-04): Regret-Guided Search Control (Tsai et al., ICLR 2026; the authors' MiniZero code,
//minizero/actor/zero_actor.cpp), in KataGo's self-play. See games/enclosure/docs/decisions/rgsc.md, "RGSC: the authors' code
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
//Enclosure fork (2026-10-11): a second source, value swing (games/enclosure/docs/decisions/rgsc.md, "RGSC with two sources"; not
//published, a named departure). Its regret at a position is sum_k gamma^k (v_{t+k+1} - v_{t+k})^2 / sum_k gamma^k
//over the rest of the game (v the search value of the move played, the outcome as the value after the last move),
//times a scale that brings it to the vanilla regret's range, capped at 4 (the regret value head's range). It has its
//own two heads (the swing value and the swing ranking), its own tracker and its own buffer with the same rules,
//except that its warm-up drops no game: its restarts begin when its buffer is full. Each game draws at most one
//restart: from the vanilla buffer with rgscStartProb, else from the swing buffer with rgscSwingStartProb.
//
//KataGame fork (2026-10-27, abscls-luzl.7): exact states, a choice of the game. A game that gives its whole state as
//a text (GameInterface::stateText: Blueprints) keeps states in the buffer, not moves: each entry holds the state's
//text, the context of its network evaluation (the komi and KataGo's asymmetric playouts) and its turn number. Its key
//is a hash of the state (the position's hash with the player to move, what the inputs read beyond it) and of the
//context, and the buffer checks the whole state when two keys are equal. A restart starts from the state itself with
//that context: no new komi and no asymmetric playouts are drawn. An entry whose state does not come back with its key
//is rejected. A game without a state's text (Enclosure) keeps the moves from the empty board and their unordered key
//(keyOf), as before. The engine repository's games/blueprints/README.md, "RGSC".

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "../core/config_parser.h"
#include "../core/rand.h"
#include "../dataio/sgf.h"
#include "../game/boardhistory.h"

struct NNOutput;

namespace Rgsc {

  //Whether the buffer keeps states (the game gives its state's text) or moves.
  constexpr bool keepsStates() { return GameInterface::hasStateText<GameRules>(); }
  //The version of a state's line in the buffer's file. Version 2 (2026-10-29) adds the state's member and side
  //(Entry), and a line of version 1 still loads, with both unknown.
  constexpr int STATE_RECORD_VERSION = 2;

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
    //The swing source (2026-10-11). Its targets are written whenever RGSC is on and swingScale > 0 (0: no swing
    //targets, as in Blueprints, which does not take Enclosure's scale); its buffer exists only when
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

  //The context of a state's network evaluation, which a restart keeps (a game that keeps states): the rules' komi, and
  //KataGo's asymmetric playouts, the player with the advantage (C_EMPTY for none) and its doublings.
  struct Context {
    float komi = 0.0f;
    Player advantagePla = C_EMPTY;
    double advantage = 0.0;
    //The context, with no player when there is no advantage, and 0 for -0.
    static Context of(float komi, Player advantagePla, double advantage);
    bool operator==(const Context& other) const;
  };
  //A state's key: the position's hash with the player to move (Board::pos_hash), what the inputs read beyond it
  //(GameEncoding::input_hash: the history planes), and the context.
  uint64_t stateKey(const Board& board, const Context& context);

  struct Entry {
    uint64_t key = 0;
    double regret = 0.0;
    int64_t sampleCount = 0;
    int origin = ORIGIN_UNKNOWN;
    //KataGame fork (2026-10-29): the AZdb member (SearchParams::diversityMember, 0 without AZdb) and the side (the
    //rules' player, Board::indexOfPlayer) of the search that produced the state: the search that evaluated a tree
    //state, or the search of the player to move at a played position. For the analysis only: no draw reads them.
    //-1 for unknown (states saved before 2026-10-29).
    int member = -1;
    int side = -1;
    std::vector<Move> moves;  //from the empty board, in a game that keeps moves
    //In a game that keeps states: the state's text (GameInterface::stateText), its context, and its turn number in
    //its game (the actions since the start, which the search's schedules read, such as the move's temperature).
    std::string state;
    Context context;
    int64_t turnNumber = 0;
  };
  //The entry of a state (a game that keeps states), with its key. Its regret, origin, member and side are the
  //caller's.
  Entry stateEntry(const Board& board, const Context& context, int64_t turnNumber);
  //Sets board to an entry's state (a game that keeps states). False, with the reason, when the text gives no state or
  //the state's key is not the entry's.
  bool restoreState(const Entry& entry, Board& board, std::string& why);
  //The keys of a game's played positions: keys[k] of the position after start + k of hist's moves, for k from 0 to
  //count - 1. The moves' key (keyOf), or in a game that keeps states the state's key with the context.
  std::vector<uint64_t> playedKeys(const BoardHistory& hist, size_t start, size_t count, const Context& context);
  //The entry of the position after `length` of hist's moves, with its key: the moves, or in a game that keeps states
  //the state (replayed from hist's first position), the context and the turn number. Its regret, origin, member and
  //side are the caller's.
  Entry playedEntry(const BoardHistory& hist, size_t length, const Context& context);
  //Sets start to a game's exact start from an entry: its moves from an empty board of xSize x ySize, or in a game that
  //keeps states the state itself, with its turn number and its context frozen (Sgf::PositionSample::frozenContext).
  //False, with the reason, when the state does not come back with its key (restoreState).
  bool startOf(const Entry& entry, int xSize, int ySize, Sgf::PositionSample& start, std::string& why);

  class Buffer {
   public:
    //name: how the log names it ("RGSC", "RGSC swing").
    Buffer(const Settings& settings, const std::string& name = "RGSC");

    //Loads the buffer and the finished-game count from the settings' file, if it exists. In a game that keeps states,
    //each state is restored (restoreState), after the game's settings are set: an entry that does not restore with
    //its key is rejected, and so is an entry of the other kind (moves in a game that keeps states, or the reverse).
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
    //A game's candidate: enters if there is room or its regret beats the lowest, which it replaces. A state already
    //in the buffer is skipped. With states, a key in the buffer whose state or context is another is a collision of
    //the hash: the candidate is skipped too (one state a key) and counted.
    void offer(Entry candidate);
    void offer(uint64_t key, double regret, const std::vector<Move>& moves, int origin = ORIGIN_UNKNOWN);
    //Removes the entry with this key, whose state did not come back for a restart (restoreState), and counts it.
    void reject(uint64_t key);
    //After a game started from the state with this key: the EMA update. False if the state left the buffer.
    bool updateAfterRestart(uint64_t key, double regret);
    //Counts a finished game; returns how many had finished before it (for the warm-up).
    int64_t countFinishedGame();
    //Adds a game's regret targets to this run's regret by kind of action (summary): a turn whose action continues the
    //move of the action before it (continuesMove[k], GameInterface::continuesMove: Blueprints' drop), a turn whose
    //action the next turn's continues (a pick), and a move of one action (a placement).
    void addRegrets(const std::vector<float>& regrets, const std::vector<bool>& continuesMove);
    size_t size() const;
    //A one-line summary for the log: states and their regrets and origins, and this run's counts (restarts drawn,
    //written after their state left the buffer, and dropped; candidates offered and taken; sample attempts while
    //the buffer was empty, or not yet full; entries rejected at the load and at a restart, and collisions of keys),
    //and this run's regret targets by kind of action (addRegrets).
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
    int64_t rejectedAtLoad = 0;
    int64_t rejectedAtRestart = 0;
    int64_t keyCollisions = 0;
    //Regret targets by kind of action: a move of one action, a move's first action, a later action.
    int64_t regretCount[3] = {0, 0, 0};
    double regretSum[3] = {0.0, 0.0, 0.0};
  };

  //Fed every network evaluation of one game's searches (Search::rgscTracker, rgscSwingTracker): keeps the state
  //that the source's ranking head ranks highest.
  class TreeTracker {
   public:
    TreeTracker(const Buffer* buffer, Source source = SOURCE_VANILLA);
    //context: the evaluation's, which a game that keeps states keys and keeps with the state. member and side: the
    //evaluating search's (Entry::member, Entry::side), which the candidate keeps.
    void offer(const NNOutput& nnOutput, const Board& board, const BoardHistory& hist, const Context& context,
               int member, int side);
    bool found() const { return hasCandidate; }
    float ranking() const { return bestRanking.load(std::memory_order_acquire); }
    float regretValue() const { return candidateRegretValue; }
    //The candidate as an entry, its moves or its state, with its member and side: its regret and origin are the
    //caller's.
    Entry candidate() const;

   private:
    const Buffer* buffer;
    Source source;
    std::mutex mutex;
    std::atomic<float> bestRanking;
    bool hasCandidate;
    float candidateRegretValue;
    //The candidate's key, and its moves, or in a game that keeps states its context and turn number with its board,
    //whose text candidate() writes.
    Entry candidateEntry;
    Board candidateBoard;
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
