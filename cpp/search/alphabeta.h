#ifndef SEARCH_ALPHABETA_H_
#define SEARCH_ALPHABETA_H_

/*
 * The alpha-beta search for every game: negamax with principal variation search, iterative deepening under a time
 * budget and a largest depth, and a transposition table on the game's hash. At each position the table's move comes
 * first, then the game's move ordering. It reaches the game through its rules part (game/gameinterface.h) and its bot
 * part (game/botinterface.h): the heuristic and the optional move ordering. GAMES.md, "The bot part".
 *
 * - Values are from the view of the player to move. A move changes the sign only when it changes the player to move,
 *   so turns of several moves (Enclosure's 1-2-2) work. A depth counts moves, not turns.
 * - An end of the game is worth result * WIN_VALUE + margin to the player to move: the result (1, 0 or -1) and the
 *   margin score(mover) - score(other). So every won end is above every heuristic value and every lost end, and of
 *   two wins the larger margin is better (Connect-4's score is the solver's). A position at the search's depth that is
 *   not an end is worth the game's heuristic, kept within +-WIN_VALUE / 2.
 * - A value is exact when every line under it that the search read reached an end of the game. The table keeps an
 *   exact value for every depth, and iterative deepening stops when the root's value is exact.
 * - A search that runs out of time returns the last depth that it completed. If it completed none, it returns the
 *   best move found so far, or the first move of its order.
 * - Of moves of equal value, the search takes the first in its order. With a seed, it first orders each position's
 *   moves at random, so that the choice among them is random, as the greedy bot's.
 *
 * The Python binding exposes it (core/binding/rules.cpp, AlphaBeta). katago does not compile it. One search runs at a
 * time in each AlphaBeta: it is not thread-safe.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "../core/hash.h"
#include "../game/botinterface.h"

template<class Bot>
class AlphaBeta {
 public:
  using Rules = typename Bot::Rules;
  using Position = typename Rules::Position;

  static constexpr double WIN_VALUE = 1e9;

  struct Result {
    int action = -1;      // the best move; -1 when the game is over
    double value = 0.0;   // its value for the player to move
    int depth = 0;        // the last depth that the search completed (0: none)
    bool exact = false;   // the value is the game's value with perfect play
    int64_t nodes = 0;    // the positions that the search visited
    double seconds = 0.0;
  };

  // The result (1, 0 or -1) and the margin that an exact value holds.
  static int resultOf(double value) { return value >= WIN_VALUE / 2 ? 1 : value <= -WIN_VALUE / 2 ? -1 : 0; }
  static double marginOf(double value) { return value - resultOf(value) * WIN_VALUE; }

  static constexpr size_t DEFAULT_TABLE_BYTES = size_t(32) << 20;

  // The table takes the largest power of two of entries that fits in tableBytes. With a seed, moves of equal value are
  // chosen at random.
  explicit AlphaBeta(size_t tableBytes = DEFAULT_TABLE_BYTES, std::optional<uint64_t> seed = std::nullopt)
      : randomTies_(seed.has_value()), tieSeed_(seed.value_or(0)) {
    size_t entries = 1;
    while(entries * 2 * sizeof(Entry) <= tableBytes)
      entries *= 2;
    table_.resize(entries);
    mask_ = entries - 1;
  }

  // seconds <= 0: no time budget. maxDepth <= 0: no largest depth. With neither, it searches until the value is exact.
  Result search(const Position& root, double seconds, int maxDepth) {
    const Clock::time_point start = Clock::now();
    timed_ = seconds > 0;
    if(timed_)
      deadline_ = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    aborted_ = false;
    nodes_ = 0;
    generation_++;
    Result best;
    if(!Rules::terminal(root)) {
      if(stack_.empty())
        stack_.push_back(root);
      stack_[0] = root;
      const int lastDepth = maxDepth > 0 ? std::min(maxDepth, RESOLVED - 1) : RESOLVED - 1;
      for(int depth = 1; depth <= lastDepth; depth++) {
        if(stack_.size() < size_t(depth) + 1) {
          stack_.resize(depth + 1, root);
          actions_.resize(depth + 1);
        }
        rootBestAction_ = -1;
        bool exact;
        const double value = negamax(0, depth, -INF, INF, exact);
        if(aborted_) {
          // At depth 1 the clock is first read after the root has ordered its moves (CLOCK_NODES).
          if(best.depth == 0)
            best.action = rootBestAction_ >= 0 ? rootBestAction_ : actions_[0].empty() ? -1 : actions_[0][0];
          break;
        }
        best.action = rootBestAction_;
        best.value = value;
        best.depth = depth;
        best.exact = exact;
        if(exact)
          break;
      }
    }
    best.nodes = nodes_;
    best.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return best;
  }

 private:
  using Clock = std::chrono::steady_clock;
  static constexpr double INF = std::numeric_limits<double>::infinity();
  enum Bound : uint8_t { EXACT_VALUE, LOWER_BOUND, UPPER_BOUND };
  static constexpr int RESOLVED = std::numeric_limits<int16_t>::max();  // the depth of an exact value: every depth
  static constexpr int CLOCK_NODES = 16;  // the nodes between two readings of the clock

  struct Entry {
    Hash128 key;
    double value = 0.0;
    int32_t action = -1;
    int16_t depth = -1;  // -1: empty
    uint8_t bound = EXACT_VALUE;
    uint8_t generation = 0;
  };

  std::vector<Entry> table_;
  uint64_t mask_ = 0;
  uint8_t generation_ = 0;
  std::vector<Position> stack_;  // the position at each ply: a copy keeps its containers' capacity
  std::vector<std::vector<int>> actions_;  // the ordered actions at each ply
  Clock::time_point deadline_;
  bool timed_ = false, aborted_ = false;
  int64_t nodes_ = 0;
  int rootBestAction_ = -1;
  bool randomTies_;
  uint64_t tieSeed_;

  // The value of an end of the game for the player to move.
  static double endValue(const Position& p, int mover) {
    const double result = Rules::result(p);  // the first player's
    return (mover == 0 ? result : -result) * WIN_VALUE + Rules::score(p, mover) - Rules::score(p, 1 - mover);
  }

  // The legal actions at the ply: the game's order, with the table's move (or -1) first. With random ties, the moves are
  // first shuffled in an order drawn from the seed and the position (Fisher-Yates on splitmix64), so each position
  // keeps its order at every depth and in every search, and the table's moves are random picks too.
  void orderedActions(int ply, const Hash128& key, int tableAction) {
    std::vector<int>& actions = actions_[ply];
    Rules::legal_actions(stack_[ply], actions);
    if(randomTies_) {
      const uint64_t order = tieSeed_ ^ key.hash0;
      for(size_t i = actions.size(); i > 1; i--)
        std::swap(actions[i - 1], actions[Hash::splitMix64(order + i) % i]);
    }
    GameInterface::orderActions<Bot>(stack_[ply], actions);
    const auto it = std::find(actions.begin(), actions.end(), tableAction);
    if(it != actions.end())
      std::rotate(actions.begin(), it, it + 1);
  }

  // The value of the child at ply + 1 for the player to move at ply, searched in the window (alpha, beta) of that
  // player.
  double childValue(int ply, int depth, bool sameMover, double alpha, double beta, bool& exact) {
    if(sameMover)
      return negamax(ply + 1, depth - 1, alpha, beta, exact);
    return -negamax(ply + 1, depth - 1, -beta, -alpha, exact);
  }

  // Fail-soft negamax at stack_[ply] in the window (alpha, beta). exact: the value holds at every depth.
  double negamax(int ply, int depth, double alpha, double beta, bool& exact) {
    exact = false;
    if(++nodes_ % CLOCK_NODES == 0 && timed_ && Clock::now() >= deadline_)
      aborted_ = true;
    if(aborted_)
      return 0.0;
    const Position& p = stack_[ply];
    const int mover = Rules::next_player(p);
    if(Rules::terminal(p)) {
      exact = true;
      return endValue(p, mover);
    }
    if(depth <= 0)
      return std::clamp(double(Bot::evaluate(p)), -WIN_VALUE / 2, WIN_VALUE / 2);

    const Hash128 key = Rules::hash(p);
    Entry& entry = table_[key.hash0 & mask_];
    int tableAction = -1;
    if(entry.depth >= 0 && entry.key == key) {
      tableAction = entry.action;
      // The root always searches, to give its move.
      if(ply > 0 && entry.depth >= depth &&
         (entry.bound == EXACT_VALUE || (entry.bound == LOWER_BOUND && entry.value >= beta) ||
          (entry.bound == UPPER_BOUND && entry.value <= alpha))) {
        exact = entry.depth == RESOLVED;
        return entry.value;
      }
    }

    orderedActions(ply, key, tableAction);
    const std::vector<int>& actions = actions_[ply];
    if(actions.empty()) {  // a rules part lists a legal action until the end (the conformance suite checks it)
      exact = true;
      return endValue(p, mover);
    }
    const double alphaStart = alpha;
    double best = -INF;
    int bestAction = actions[0];
    bool allExact = true;
    Position& child = stack_[ply + 1];
    for(size_t i = 0; i < actions.size(); i++) {
      child = p;
      Rules::play(child, actions[i]);
      const bool sameMover = Rules::next_player(child) == mover;
      bool childExact;
      double value;
      if(i == 0)
        value = childValue(ply, depth, sameMover, alpha, beta, childExact);
      else {
        // Principal variation search: a null window proves the move no better, else a full window searches again.
        value = childValue(ply, depth, sameMover, alpha, std::nextafter(alpha, INF), childExact);
        if(!aborted_ && value > alpha && value < beta)
          value = childValue(ply, depth, sameMover, alpha, beta, childExact);
      }
      if(aborted_)
        return 0.0;
      allExact = allExact && childExact;
      if(value > best) {
        best = value;
        bestAction = actions[i];
        if(ply == 0)
          rootBestAction_ = bestAction;
        if(value > alpha) {
          alpha = value;
          if(alpha >= beta)
            break;
        }
      }
    }

    const int storedDepth = allExact ? RESOLVED : depth;  // depth < RESOLVED: search() caps it
    // Within a search, a deeper entry of another position stays.
    if(entry.depth < 0 || entry.generation != generation_ || entry.key == key || storedDepth >= entry.depth) {
      entry.key = key;
      entry.value = best;
      entry.action = bestAction;
      entry.depth = int16_t(storedDepth);
      entry.bound = best <= alphaStart ? UPPER_BOUND : best >= beta ? LOWER_BOUND : EXACT_VALUE;
      entry.generation = generation_;
    }
    exact = allExact;
    return best;
  }
};

#endif  // SEARCH_ALPHABETA_H_
