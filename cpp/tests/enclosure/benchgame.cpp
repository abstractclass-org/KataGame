// KataGame fork (abscls-iwsg): the game layer's CPU costs on a build of any game, per position, on the positions of
// GAMES games (by default 6): each even game random play, each odd game the greedy player (the legal action that scores
// the most now, as GameRules::score gives it) against itself. The best of REPS passes (by default 3) over all the
// positions, in microseconds a position: the legal actions, the inputs at the identity and at symmetry 3 (if the game
// allows it), the global inputs, the hash, a copy of the position, a copy and a move, a board's copy, a board's move
// with its sync, a history's copy, a board's legal actions and the image of each legal action. Each line ends with a
// sum of the results, so that the compiler keeps the work. For attribution only: self-play's benchmark decides.
//
//   TESTS=benchgame bash tests/enclosure/build.sh BUILD_DIR OUT_DIR
//   OUT_DIR/benchgame [GAMES] [REPS]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../../game/board.h"
#include "../../game/boardhistory.h"
#include "../../neuralnet/nninputs.h"

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  Board::initHash();
  const int games = argc > 1 ? atoi(argv[1]) : 6;
  const int reps = argc > 2 ? atoi(argv[2]) : 3;
  std::mt19937_64 rng(1);
  std::vector<GameRules::Position> positions;
  std::vector<int> legal;
  static GameRules::Position c;
  for(int g = 0; g < games; g++) {
    GameRules::Position p;
    GameRules::new_game(GameRules::MAX_BOARD_LEN, GameRules::MAX_BOARD_LEN, p);
    while(!GameRules::terminal(p)) {
      positions.push_back(p);
      GameRules::legal_actions(p, legal);
      if(legal.empty()) {
        GameRules::play(p, GameRules::PASS_ACTION);
        continue;
      }
      int chosen = legal[rng() % legal.size()];
      if(g % 2 == 1) {
        const int mover = GameRules::next_player(p);
        double best = 0;
        for(int a : legal) {
          c = p;
          GameRules::play(c, a);
          const double gain = GameRules::score(c, mover) - GameRules::score(p, mover);
          if(gain > best) {
            best = gain;
            chosen = a;
          }
        }
      }
      GameRules::play(p, chosen);
    }
  }
  printf("%zu positions of %d games\n", positions.size(), games);
  auto time = [&](const char* name, auto&& f) {
    double best = 1e30;
    long long sink = 0;
    for(int r = 0; r < reps; r++) {
      const auto t0 = Clock::now();
      for(size_t i = 0; i < positions.size(); i++)
        sink += f(i);
      best = std::min(best, std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
    }
    printf("%-28s %9.2f us/position  (%lld)\n", name, best / positions.size(), sink);
  };
  time("legal_actions", [&](size_t i) {
    GameRules::legal_actions(positions[i], legal);
    return (long long)legal.size();
  });
  const int otherSymmetry = SymmetryHelpers::isAllowed(3) ? 3 : 0;
  for(int sym : {0, otherSymmetry}) {
    GameInterface::InputOptions o;
    o.symmetry = sym;
    o.history = true;
    char label[64];
    snprintf(label, sizeof label, "spatial_inputs sym %d", sym);
    time(label, [&](size_t i) {
      long long n = 0;
      GameEncoding::spatial_inputs(positions[i], o, [&](int, int, int) { n++; });
      return n;
    });
  }
  std::vector<float> globals(NNInputs::NUM_FEATURES_GLOBAL_V9);
  time("global_inputs", [&](size_t i) {
    GameInterface::InputOptions o;
    GameEncoding::global_inputs(positions[i], o, globals.data());
    return (long long)globals[0];
  });
  time("hash", [&](size_t i) { return (long long)GameRules::hash(positions[i]).hash0; });
  time("Position copy", [&](size_t i) {
    c = positions[i];
    return (long long)GameRules::moves_played(c);
  });
  std::vector<int> chosen(positions.size());
  for(size_t i = 0; i < positions.size(); i++) {
    GameRules::legal_actions(positions[i], legal);
    chosen[i] = legal.empty() ? GameRules::PASS_ACTION : legal[rng() % legal.size()];
  }
  time("Position copy + play", [&](size_t i) {
    c = positions[i];
    GameRules::play(c, chosen[i]);
    return (long long)GameRules::moves_played(c);
  });
  std::vector<Board> boards(positions.size());
  for(size_t i = 0; i < positions.size(); i++)
    boards[i].setPosition(positions[i]);
  static Board b;
  time("Board copy", [&](size_t i) {
    b = boards[i];
    return (long long)b.pos_hash.hash0;
  });
  time("Board play (sync)", [&](size_t i) {
    b = boards[i];
    b.playMoveAssumeLegal(Board::moveLocOfAction(chosen[i]), b.nextPla());
    return (long long)b.pos_hash.hash0;
  });
  const BoardHistory h0(boards[0], boards[0].nextPla(), Rules(), 0);
  static BoardHistory h;
  time("BoardHistory copy", [&](size_t i) {
    h = h0;
    return (long long)h.moveHistory.size() + (long long)i;
  });
  time("Board legalActions", [&](size_t i) {
    b = boards[i];
    return (long long)b.legalActions().count();
  });
  time("SymActionMap", [&](size_t i) {
    const SymmetryHelpers::SymActionMap m(boards[i], otherSymmetry);
    GameRules::legal_actions(positions[i], legal);
    long long s = 0;
    for(int a : legal)
      s += m(a);
    return s;
  });
  return 0;
}
