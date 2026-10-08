// Enclosure fork (2026-10-05): the input row's clearing timed, the whole row (77,615 floats) against only the entries
// the last fill set (fillRowV8's setEntries). One thread refills one row position after position, as a search thread
// does, over the inputs fixture's positions. Built by tests/enclosure/build.sh
// (TESTS=benchfillrow); from katago/cpp:
//   OUT/benchfillrow tests/data/enclosure_inputs.txt [ROUNDS [ROWS THREADS]]
// With ROWS and THREADS: as self-play, THREADS threads each refill its share of ROWS rows in turn (self-play: 384 rows,
// one a game, on 16 CPUs), so each row is cold when refilled; the wall time per fill over all threads.
#include <chrono>
#include <thread>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "../../core/global.h"
#include "../../game/boardhistory.h"
#include "../../neuralnet/nninputs.h"

using namespace std;

int main(int argc, char** argv) {
  if(argc < 2) {
    fprintf(stderr, "usage: benchfillrow INPUTS [ROUNDS]\n");
    return 2;
  }
  const int rounds = argc > 2 ? atoi(argv[2]) : 20;
  Board::initHash();
  const int S = enclosure::SIZE, C = NNInputs::NUM_FEATURES_SPATIAL_V8, G = NNInputs::NUM_FEATURES_GLOBAL_V8;

  vector<Board> boards;
  vector<BoardHistory> hists;
  ifstream in(argv[1]);
  string line;
  while(getline(in, line)) {
    istringstream head(line);
    string word, label;
    int n;
    if(!(head >> word >> label >> n) || word != "position")
      continue;
    Board board;
    BoardHistory hist(board, P_BLACK, Rules(), 0);
    bool legal = true;
    for(int i = 0; i < n && legal; i++) {
      int action;
      head >> action;
      Loc loc = Board::moveLocOfAction(action);
      Player pla = hist.presumedNextMovePla;
      legal = hist.isLegal(board, loc, pla);
      if(legal)
        hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    }
    if(legal && !hist.isGameFinished) {
      boards.push_back(board);
      hists.push_back(hist);
    }
  }
  const size_t P = boards.size();
  MiscNNInputParams params;
  vector<float> row(S * S * C, 0.0f), global(G);
  vector<int> set;
  double checksum = 0;
  size_t entries = 0;

  for(bool nhwc : {true, false}) {
    for(int mode = 0; mode < 2; mode++) {
      std::fill(row.begin(), row.end(), 0.0f);
      set.clear();
      // One round untimed, to warm the caches and the feature code's thread-local buffers.
      for(int r = -1; r < rounds; r++) {
        auto t0 = chrono::steady_clock::now();
        for(size_t i = 0; i < P; i++) {
          NNInputs::fillRowV8(boards[i], hists[i], hists[i].presumedNextMovePla, params, S, S, nhwc, row.data(), global.data(),
                              mode == 1 ? &set : NULL);
          checksum += row[(i * 7919) % row.size()];
          if(mode == 1 && r == 0)
            entries += set.size();
        }
        auto t1 = chrono::steady_clock::now();
        static double total[2][2] = {{0, 0}, {0, 0}};
        if(r >= 0)
          total[nhwc][mode] += chrono::duration<double>(t1 - t0).count();
        if(r == rounds - 1)
          printf("%s %-28s %8.2f us per fill (%zu positions x %d rounds)\n", nhwc ? "NHWC" : "NCHW",
                 mode == 0 ? "whole row cleared:" : "only the last entries cleared:", 1e6 * total[nhwc][mode] / (P * rounds),
                 P, rounds);
      }
    }
  }
  if(argc > 4) {
    const int nrows = atoi(argv[3]), nthreads = atoi(argv[4]);
    for(int mode = 0; mode < 2; mode++) {
      vector<vector<float>> rows(nrows, vector<float>(S * S * C, 0.0f));
      vector<vector<int>> sets(nrows);
      std::atomic<double> sink(0);
      auto work = [&](int t) {
        vector<float> g(G);
        double local = 0;
        const int per = nrows / nthreads;
        for(int r = 0; r < rounds; r++)
          for(size_t i = 0; i < P; i++) {
            const int k = t * per + int(i % per);
            NNInputs::fillRowV8(boards[i], hists[i], hists[i].presumedNextMovePla, params, S, S, true, rows[k].data(), g.data(),
                                mode == 1 ? &sets[k] : NULL);
            local += rows[k][(i * 7919) % rows[k].size()];
          }
        sink = sink + local;
      };
      auto t0 = chrono::steady_clock::now();
      vector<std::thread> threads;
      for(int t = 0; t < nthreads; t++)
        threads.emplace_back(work, t);
      for(auto& th : threads)
        th.join();
      auto t1 = chrono::steady_clock::now();
      printf("%d rows, %d threads, %-30s %8.2f us wall per fill, %.0f fills/s (sink %.0f)\n", nrows, nthreads,
             mode == 0 ? "whole row cleared:" : "only the last entries cleared:",
             1e6 * chrono::duration<double>(t1 - t0).count() / (double(P) * rounds * nthreads),
             double(P) * rounds * nthreads / chrono::duration<double>(t1 - t0).count(), double(sink));
    }
    return 0;
  }
  printf("set entries per fill: %.0f of %d; checksum %.0f\n", double(entries) / P / 2, S * S * C, checksum);
  // The clearing alone: the whole row, against the last position's entries.
  const int reps = 20000;
  auto t0 = chrono::steady_clock::now();
  for(int r = 0; r < reps; r++) {
    std::fill(row.begin(), row.end(), 0.0f);
    checksum += row[r % row.size()];
  }
  auto t1 = chrono::steady_clock::now();
  for(int r = 0; r < reps; r++) {
    for(int i : set)
      row[i] = 0.0f;
    checksum += row[r % row.size()];
  }
  auto t2 = chrono::steady_clock::now();
  printf("clearing alone: whole row %.2f us, %zu entries %.2f us (checksum %.0f)\n",
         1e6 * chrono::duration<double>(t1 - t0).count() / reps, set.size(),
         1e6 * chrono::duration<double>(t2 - t1).count() / reps, checksum);
  return 0;
}
