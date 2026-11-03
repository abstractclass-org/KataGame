// Blueprints (abscls-gml5.3): the CPU's cost of the reach of the stars (game/blueprints/starreach.h, both players)
// beside the network's input fill of the inputs versions 14 and 15 (NNInputs), on the same positions, on one thread.
// Version 15 (abscls-gml5.4) is version 14 with the planes of reach, which the reach computes.
//
//   benchstarreach STATES [PASSES]
//
// STATES: a position's state a line, after its source's name and a tab, as the engine repository's
// games/blueprints/tools/star_reach_check.py --states writes them (GameRules::state_text). The positions keep their
// views' orientations (the setting viewOrientation=randomGame keeps each state's).
//
// Each position holds rules boards of 511 x 511 points on the heap (its board, and a history's boards), so the
// positions go in chunks of CHUNK. For each chunk: one pass of each function untimed (the scratch storage and the
// caches), then PASSES passes (by default 5) of the reach over the chunk's positions, then PASSES passes of each fill.
// Each call is timed alone, and a position's time is its least over the passes. Between two calls on one position, a
// pass calls the function on the chunk's other positions, as a search reads another position each time.
//
// The functions: the reach of both players' stars (star_reach, with its budget), the packed fill that each network
// evaluation calls (NNInputs::fillRowV14Packed and fillRowV15Packed, nneval.cpp) and the float fill that the training
// rows call (NNInputs::fillRowV14 and fillRowV15, trainingwrite.cpp), at the identity. It prints, for each source and
// for all the positions: their count, then for each function the mean, the median, the 99th percentile and the most
// microseconds a position, the mean of the reach over the mean of version 14's packed fill, each fill of version 15
// over version 14's (means), and the mean and the 99th percentile of version 15's packed fill less version 14's, from
// each position's least times. Then the reach's values (all its stars) and
// its work units (both players' sum: the mean and the most), and a sum of the results, so that the compiler keeps the
// work. Built by tests/enclosure/build.sh with TESTS=blueprints/benchstarreach, against a Blueprints build.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "../../game/blueprints/starreach.h"
#include "../../game/boardhistory.h"
#include "../../neuralnet/nninputs.h"

using namespace std;
using Clock = chrono::steady_clock;

namespace {

constexpr int CHUNK = 64;

struct Times {
  vector<double> reach, packed, floats, packed15, floats15, extra;  // microseconds, one a position
};

double percentile(vector<double> v, double p) {
  sort(v.begin(), v.end());
  const size_t i = min(v.size() - 1, size_t(p * double(v.size() - 1) + 0.5));
  return v[i];
}

double mean(const vector<double>& v) {
  double sum = 0;
  for(double x : v)
    sum += x;
  return sum / double(v.size());
}

void show(const string& name, const vector<double>& v) {
  printf("  %-36s mean %7.2f  median %7.2f  p99 %8.2f  max %8.2f\n", name.c_str(), mean(v), percentile(v, 0.5),
         percentile(v, 0.99), *max_element(v.begin(), v.end()));
}

}  // namespace

int main(int argc, char** argv) {
  if(argc < 2) {
    fprintf(stderr, "usage: benchstarreach STATES [PASSES]\n");
    return 2;
  }
  const int passes = argc > 2 ? atoi(argv[2]) : 5;
  Board::initHash();
  ScoreValue::initTables();
  string why;
  if(!GameRules::set_setting("viewOrientation", "randomGame", why)) {
    fprintf(stderr, "%s\n", why.c_str());
    return 2;
  }
  ifstream in(argv[1]);
  if(!in) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  const int S = NNPos::MAX_BOARD_LEN;
  MiscNNInputParams params;
  vector<uint32_t> bits(NNInputs::gameSpatialWords(15));
  vector<float> floats(size_t(NNInputs::NUM_FEATURES_SPATIAL_V15) * S * S), global(NNInputs::NUM_FEATURES_GLOBAL_V15);
  blueprints::StarReach out;
  map<string, Times> bySource;
  map<int, long long> values;
  long long sumWork = 0, mostWork = 0, positions = 0;
  double sink = 0;

  vector<string> sources;
  vector<Board> boards;
  vector<BoardHistory> hists;
  boards.reserve(CHUNK);
  hists.reserve(CHUNK);
  auto timeChunk = [&]() {
    const size_t n = boards.size();
    vector<double> reach(n, 1e300), packed(n, 1e300), flt(n, 1e300), packed15(n, 1e300), flt15(n, 1e300);
    auto runReach = [&](size_t i) {
      blueprints::star_reach(boards[i].pos, out);
      for(int pl = 0; pl < 2; pl++)
        for(const blueprints::StarReach::Star& s : out.stars[pl])
          sink += s.k;
    };
    auto runPacked = [&](size_t i) {
      const Player pla = boards[i].nextPla();
      NNInputs::fillRowV14Packed(boards[i], hists[i], pla, params, S, S, false, bits.data(), global.data(), 0);
      sink += bits[i % bits.size()] & 1;
    };
    auto runFloats = [&](size_t i) {
      const Player pla = boards[i].nextPla();
      NNInputs::fillRowV14(boards[i], hists[i], pla, params, S, S, false, floats.data(), global.data());
      sink += floats[(i * 7919) % floats.size()];
    };
    auto runPacked15 = [&](size_t i) {
      const Player pla = boards[i].nextPla();
      NNInputs::fillRowV15Packed(boards[i], hists[i], pla, params, S, S, false, bits.data(), global.data(), 0);
      sink += bits[i % bits.size()] & 1;
    };
    auto runFloats15 = [&](size_t i) {
      const Player pla = boards[i].nextPla();
      NNInputs::fillRowV15(boards[i], hists[i], pla, params, S, S, false, floats.data(), global.data());
      sink += floats[(i * 7919) % floats.size()];
    };
    auto timeAll = [&](auto&& f, vector<double>& best) {
      for(size_t i = 0; i < n; i++)
        f(i);  // untimed
      for(int pass = 0; pass < passes; pass++)
        for(size_t i = 0; i < n; i++) {
          const auto t0 = Clock::now();
          f(i);
          const double us = chrono::duration<double, micro>(Clock::now() - t0).count();
          best[i] = min(best[i], us);
        }
    };
    timeAll(runReach, reach);
    timeAll(runPacked, packed);
    timeAll(runPacked15, packed15);
    timeAll(runFloats, flt);
    timeAll(runFloats15, flt15);
    for(size_t i = 0; i < n; i++) {
      blueprints::star_reach(boards[i].pos, out);
      long long work = 0;
      for(int pl = 0; pl < 2; pl++) {
        work += out.work[pl];
        for(const blueprints::StarReach::Star& s : out.stars[pl])
          values[s.k]++;
      }
      sumWork += work;
      mostWork = max(mostWork, work);
      for(const string& name : {sources[i], string("all")}) {
        Times& t = bySource[name];
        t.reach.push_back(reach[i]);
        t.packed.push_back(packed[i]);
        t.floats.push_back(flt[i]);
        t.packed15.push_back(packed15[i]);
        t.floats15.push_back(flt15[i]);
        t.extra.push_back(packed15[i] - packed[i]);
      }
    }
    positions += (long long)n;
    sources.clear();
    boards.clear();
    hists.clear();
  };

  string line;
  long long lineNumber = 0;
  while(getline(in, line)) {
    lineNumber++;
    const size_t tab = line.find('\t');
    if(tab == string::npos) {
      fprintf(stderr, "line %lld: no source's name before a tab\n", lineNumber);
      return 2;
    }
    GameRules::Position p;
    if(!GameRules::parse_state(line.substr(tab + 1), p, why)) {
      fprintf(stderr, "line %lld: %s\n", lineNumber, why.c_str());
      return 2;
    }
    sources.push_back(line.substr(0, tab));
    boards.emplace_back();
    boards.back().setPosition(p);
    hists.emplace_back(boards.back(), boards.back().nextPla(), Rules(), 0);
    if((int)boards.size() == CHUNK)
      timeChunk();
  }
  if(!boards.empty())
    timeChunk();
  if(positions == 0) {
    fprintf(stderr, "no position\n");
    return 2;
  }

  printf("%lld positions, the least of %d passes a position, microseconds on one thread\n", positions, passes);
  for(const auto& [name, t] : bySource) {
    printf("%s: %zu positions\n", name.c_str(), t.reach.size());
    show("the reach, both players", t.reach);
    show("the packed fill (fillRowV14Packed)", t.packed);
    show("the packed fill (fillRowV15Packed)", t.packed15);
    show("the float fill (fillRowV14)", t.floats);
    show("the float fill (fillRowV15)", t.floats15);
    printf("  the reach / the packed fill of version 14: %.2f (means)\n", mean(t.reach) / mean(t.packed));
    printf("  version 15 / version 14: packed %.2f, floats %.2f (means)\n", mean(t.packed15) / mean(t.packed),
           mean(t.floats15) / mean(t.floats));
    printf("  version 15's packed fill less version 14's: mean %.2f, p99 %.2f\n", mean(t.extra),
           percentile(t.extra, 0.99));
  }
  printf("The reach's values:");
  for(const auto& [k, count] : values)
    printf(" %d: %lld", k, count);
  printf("\nThe work units of a position (both players): mean %.0f, most %lld (a player's budget %lld)\n",
         double(sumWork) / double(positions), mostWork, (long long)blueprints::StarReach::BUDGET);
  printf("sink %.3f\n", sink);
  return 0;
}
