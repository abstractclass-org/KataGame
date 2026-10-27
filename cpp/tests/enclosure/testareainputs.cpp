// KataGame fork (2026-10-27, abscls-luzl.14): the planes of closed areas of the inputs version 12 (NNInputs::fillRowV12
// and fillRowV12Packed) against a fixture of the game's reference, position by position, on a build of a game with
// such planes (GameInterface::areaSpatialInputs: Blueprints).
//
//   testareainputs FIXTURE [ROUNDS] [SETTING=VALUE ...]
//
// SETTING=VALUE: the game's settings (GameInterface::setSetting): viewOrientation=randomGame reads the fixture's texts
// in their own orientations. The fixture (the engine repository's games/blueprints/tools/area_fixture.py) has a line
// "position TEXT" for each position (GameInterface::stateText), then a line "area PLANE N POINT ..." for each plane of
// closed areas (0 the first), with the N points y * MAX_BOARD_LEN + x of the view where the reference sets it. At each
// position, at each allowed symmetry:
// - the planes of closed areas of fillRowV12Packed are the fixture's, moved as pixels (GameInterface::symPoint);
// - the planes before them are V11's (fillRowV11Packed), and the global inputs too;
// - fillRowV12's floats, in NCHW and NHWC, are the packed row's entries (at the identity).
// Then, over ROUNDS rounds (default 20) of the fixture's positions, each round V11's pass then V12's, the time of a
// packed fill of V11 and of V12 on one thread: the planes' cost.

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../../game/boardhistory.h"
#include "../../neuralnet/nninputs.h"

using namespace std;

static int failures = 0;
static void check(bool ok, const string& what) {
  if(!ok) {
    if(failures < 20)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  }
}

struct Sample {
  Board board;
  vector<vector<int>> planes;  //each plane of closed areas: its points y * len + x
};

static const int LEN = NNPos::MAX_BOARD_LEN, AREA = LEN * LEN;
static const int V11 = NNInputs::NUM_FEATURES_SPATIAL_V11, V12 = NNInputs::NUM_FEATURES_SPATIAL_V12;
static const int G = NNInputs::NUM_FEATURES_GLOBAL_V12;

static void fillPacked(const Board& board, int version, int symmetry, vector<uint32_t>& bits, vector<float>& global) {
  BoardHistory hist(board, board.nextPla(), Rules(), 0);
  MiscNNInputParams params;
  bits.assign(NNInputs::NUM_SPATIAL_WORDS_V12, 0u);
  global.assign(G, 0.0f);
  if(version == 12)
    NNInputs::fillRowV12Packed(board, hist, board.nextPla(), params, LEN, LEN, false, bits.data(), global.data(),
                               symmetry);
  else
    NNInputs::fillRowV11Packed(board, hist, board.nextPla(), params, LEN, LEN, false, bits.data(), global.data(),
                               symmetry);
}

static bool bit(const vector<uint32_t>& bits, int i) { return (bits[i >> 5] >> (i & 31)) & 1u; }

int main(int argc, char** argv) {
  static_assert(GameInterface::areaSpatialInputs<GameEncoding>() == V12 - V11, "V12 adds the planes of closed areas");
  if(argc < 2) {
    fprintf(stderr, "usage: testareainputs FIXTURE [ROUNDS] [SETTING=VALUE ...]\n");
    return 2;
  }
  Board::initHash();
  int rounds = 20;
  for(int i = 2; i < argc; i++) {
    const string arg = argv[i];
    const size_t eq = arg.find('=');
    string why;
    if(eq == string::npos)
      rounds = atoi(arg.c_str());
    else if(!GameInterface::setSetting<GameRules>(arg.substr(0, eq), arg.substr(eq + 1), why)) {
      fprintf(stderr, "%s\n", why.c_str());
      return 2;
    }
  }
  const int numAreas = V12 - V11;
  if(numAreas == 0) {
    printf("the game has no planes of closed areas: V12 is V11\n");
    return 0;
  }

  vector<Sample> samples;
  ifstream in(argv[1]);
  if(!in) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  string line;
  while(getline(in, line)) {
    if(line.rfind("position ", 0) == 0) {
      GameRules::Position p;
      string why;
      if(!GameInterface::parseState<GameRules>(line.substr(9), p, why)) {
        check(false, "a position does not parse: " + why + ": " + line.substr(0, 80));
        samples.push_back(Sample());
        continue;
      }
      Sample s;
      s.board.setPosition(p);
      samples.push_back(s);
    }
    else if(line.rfind("area ", 0) == 0) {
      istringstream row(line.substr(5));
      int plane, n;
      row >> plane >> n;
      vector<int> points(n);
      for(int& point : points)
        row >> point;
      testAssert(!samples.empty() && plane == (int)samples.back().planes.size());
      samples.back().planes.push_back(points);
    }
  }

  vector<uint32_t> bits, bitsV11;
  vector<float> global, globalV11;
  int64_t positions = 0, setPoints = 0, symmetryChecks = 0;
  for(size_t k = 0; k < samples.size(); k++) {
    const Sample& sample = samples[k];
    if((int)sample.planes.size() != numAreas) {
      check(false, "position " + to_string(k) + ": " + to_string(sample.planes.size()) + " planes in the fixture");
      continue;
    }
    positions++;
    for(int s : SymmetryHelpers::ALLOWED) {
      const string where = "position " + to_string(k) + " at symmetry " + to_string(s);
      fillPacked(sample.board, 12, s, bits, global);
      fillPacked(sample.board, 11, s, bitsV11, globalV11);
      bool same = global == globalV11;
      for(int i = 0; i < V11 * AREA && same; i++)
        same = bit(bits, i) == bit(bitsV11, i);
      check(same, where + ": the planes before the closed areas and the global inputs are V11's");
      for(int a = 0; a < numAreas; a++) {
        vector<char> expected(AREA, 0);
        for(int point : sample.planes[a]) {
          int x = point % LEN, y = point / LEN;
          GameInterface::symPoint(x, y, s, LEN, LEN);
          expected[y * LEN + x] = 1;
        }
        int differ = 0;
        for(int i = 0; i < AREA; i++)
          differ += bit(bits, (V11 + a) * AREA + i) != (expected[i] != 0);
        check(differ == 0, where + ": the plane of closed areas " + to_string(a) + " differs at " + to_string(differ) +
                             " points");
        if(s == 0)
          setPoints += (int64_t)sample.planes[a].size();
      }
      for(int i = V12 * AREA; i < NNInputs::NUM_SPATIAL_WORDS_V12 * 32; i++)
        check(!bit(bits, i), where + ": a padding bit is set");
      symmetryChecks++;
    }
    //The floats, NCHW and NHWC, against the packed row at the identity.
    fillPacked(sample.board, 12, 0, bits, global);
    BoardHistory hist(sample.board, sample.board.nextPla(), Rules(), 0);
    MiscNNInputParams params;
    for(bool nhwc : {false, true}) {
      vector<float> row(V12 * AREA, 0.0f), rowGlobal(G, 0.0f);
      NNInputs::fillRowV12(sample.board, hist, sample.board.nextPla(), params, LEN, LEN, nhwc, row.data(),
                           rowGlobal.data());
      int differ = 0;
      for(int c = 0; c < V12; c++)
        for(int i = 0; i < AREA; i++)
          differ += (row[nhwc ? i * V12 + c : c * AREA + i] != 0.0f) != bit(bits, c * AREA + i);
      check(differ == 0 && rowGlobal == global,
            "position " + to_string(k) + ": fillRowV12's floats (" + (nhwc ? "NHWC" : "NCHW") +
              ") are the packed row's");
    }
  }
  printf("%lld positions, %lld points set in the planes of closed areas, %lld checks at the symmetries\n",
         (long long)positions, (long long)setPoints, (long long)symmetryChecks);

  //The time of a packed fill of V11 and of V12, on one thread over the positions: in each round V11's pass, then V12's,
  //so that a change of the machine's load or clock in the run falls on both.
  if(rounds > 0 && positions > 0) {
    vector<BoardHistory> hists;
    for(const Sample& sample : samples)
      hists.push_back(BoardHistory(sample.board, sample.board.nextPla(), Rules(), 0));
    MiscNNInputParams params;
    bits.assign(NNInputs::NUM_SPATIAL_WORDS_V12, 0u);
    global.assign(G, 0.0f);
    double seconds[2] = {0.0, 0.0};
    for(int r = 0; r < rounds; r++)
      for(int v = 0; v < 2; v++) {
        const auto start = chrono::steady_clock::now();
        for(size_t k = 0; k < samples.size(); k++) {
          if(v == 0)
            NNInputs::fillRowV11Packed(samples[k].board, hists[k], samples[k].board.nextPla(), params, LEN, LEN, false,
                                       bits.data(), global.data());
          else
            NNInputs::fillRowV12Packed(samples[k].board, hists[k], samples[k].board.nextPla(), params, LEN, LEN, false,
                                       bits.data(), global.data());
        }
        seconds[v] += chrono::duration<double>(chrono::steady_clock::now() - start).count();
      }
    const double fills = (double)rounds * samples.size();
    printf("a packed fill: V11 %.1f us, V12 %.1f us (+%.1f us), over %d rounds of %zu positions\n",
           1e6 * seconds[0] / fills, 1e6 * seconds[1] / fills, 1e6 * (seconds[1] - seconds[0]) / fills, rounds,
           samples.size());
  }
  printf("testareainputs: %d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
