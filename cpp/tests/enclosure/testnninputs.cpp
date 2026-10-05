// Enclosure fork: the network inputs (NNInputs::fillRowV8) against the
// Enclosure repository's net.input_planes, position by position.
//
//   testnninputs tests/data/enclosure_inputs.txt
//
// The file comes from the Enclosure repository's tools/export_katago_inputs.py
// (positions from archived and random games, with the input its Python
// reference builds). Python's planes are channels last over 219 planes: the
// observation's 192 edge planes, its 5 scalar planes, then the 22 feature
// planes. V8 puts the on-board plane first, then the edge planes and the
// feature planes, and the scalars in the global row. Checked in both layouts
// (NHWC and NCHW), and the NN hash is checked to see the history.

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

#include "../../neuralnet/nninputs.h"

using namespace std;

static const int PY_PLANES = 219;
static const int PY_EDGE_PLANES = 192;
static const int PY_FEATURES_START = 197;

//V8 channel of a Python plane (not one of the scalar planes).
static int v8Channel(int plane) {
  if(plane < PY_EDGE_PLANES)
    return NNInputs::V8_EDGE_PLANES_START + plane;
  assert(plane >= PY_FEATURES_START);
  return NNInputs::V8_FEATURE_PLANES_START + (plane - PY_FEATURES_START);
}

int main(int argc, char** argv) {
  if(argc != 2 && argc != 3) {
    fprintf(stderr, "usage: testnninputs INPUTS_FILE [TURN_SAFE_FILE]\n");
    return 2;
  }
  //With ENCLOSURE_TURN_SAFE, plane 7 of each player holds its turn-safe points, from TURN_SAFE_FILE (the Python
  //reference, tools/export_turn_safe_fixture.py): "turnsafe <label> <player> <count> <y * 19 + x> ...".
  std::map<string, vector<int>> turnSafe[2];
  if(enclosure::TURN_SAFE) {
    if(argc != 3) {
      fprintf(stderr, "ENCLOSURE_TURN_SAFE needs TURN_SAFE_FILE\n");
      return 2;
    }
    ifstream ts(argv[2]);
    string tsLine;
    while(getline(ts, tsLine)) {
      istringstream row(tsLine);
      string word, label;
      int player, count;
      if(!(row >> word >> label >> player >> count) || word != "turnsafe")
        continue;
      vector<int> points(count);
      for(int& q : points)
        row >> q;
      turnSafe[player][label] = points;
    }
  }
  Board::initHash();
  ifstream in(argv[1]);
  if(!in) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  static_assert(NNInputs::NUM_FEATURES_SPATIAL_V8 == 1 + PY_EDGE_PLANES + (PY_PLANES - PY_FEATURES_START), "");
  static_assert(NNInputs::NUM_FEATURES_GLOBAL_V8 == PY_FEATURES_START - PY_EDGE_PLANES, "");
  const int S = enclosure::SIZE;
  const int C = NNInputs::NUM_FEATURES_SPATIAL_V8;
  const int G = NNInputs::NUM_FEATURES_GLOBAL_V8;
  vector<float> nhwc(S * S * C), nchw(S * S * C), expected(S * S * C);
  vector<float> global(G), globalNchw(G);
  //Rows refilled position after position, clearing only the entries the last fill set (fillRowV8's setEntries, as
  //NNEvaluator::fillRowBufs does): each must equal the fresh fill.
  vector<float> nhwcReused(S * S * C, 0.0f), nchwReused(S * S * C, 0.0f), globalReused(G);
  vector<int> nhwcSet, nchwSet;
  int reusedChecks = 0;

  string line;
  int positions = 0, failures = 0, historyChecks = 0;
  auto fail = [&](const string& what) {
    if(failures < 20)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  };
  while(getline(in, line)) {
    istringstream head(line);
    string word, label;
    int n;
    if(!(head >> word >> label >> n) || word != "position")
      continue;
    positions++;
    Board board;
    BoardHistory hist(board, P_BLACK, Rules(), 0);
    for(int i = 0; i < n; i++) {
      int action;
      head >> action;
      Loc loc = Board::moveLocOfAction(action);
      Player pla = hist.presumedNextMovePla;
      if(!hist.isLegal(board, loc, pla)) {
        fail(label + ": illegal move " + Global::intToString(action));
        break;
      }
      hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    }

    vector<float> expectedGlobal(G);
    getline(in, line);
    istringstream globalRow(line);
    globalRow >> word;
    for(int i = 0; i < G; i++)
      globalRow >> expectedGlobal[i];
    getline(in, line);
    istringstream spatialRow(line);
    int count;
    spatialRow >> word >> count;
    std::fill(expected.begin(), expected.end(), 0.0f);
    for(int pos = 0; pos < S * S; pos++)
      expected[pos * C + 0] = 1.0f;
    for(int i = 0; i < count; i++) {
      int index;
      spatialRow >> index;
      const int plane = index % PY_PLANES;
      //With ENCLOSURE_NO_HELD_GROUND or ENCLOSURE_TURN_SAFE, held ground and its openings (each player's planes 7 and
      //8) are not computed.
      const int f = plane - PY_FEATURES_START;
      if((enclosure::NO_HELD_GROUND || enclosure::TURN_SAFE) && f >= 0 &&
         (f % enclosure::PER_PLAYER == 7 || f % enclosure::PER_PLAYER == 8) && f < 2 * enclosure::PER_PLAYER)
        continue;
      expected[(index / PY_PLANES) * C + v8Channel(plane)] = 1.0f;
      //With ENCLOSURE_EDGE_BOTH_ENDS, an edge plane's entry also at the edge's upper end, in the opposite offset.
      if(enclosure::EDGE_BOTH_ENDS && plane < 4 * enclosure::NOFF) {
        const int pos = index / PY_PLANES, group = plane / enclosure::NOFF, k = plane % enclosure::NOFF;
        const int dx = enclosure::OFFS[k].dx, dy = enclosure::OFFS[k].dy;
        const int upper = (pos / S + dy) * S + pos % S + dx;
        expected[upper * C + v8Channel(group * enclosure::NOFF + enclosure::OFF_INDEX[3 - dx][3 - dy])] = 1.0f;
      }
    }
    if(enclosure::TURN_SAFE)
      for(int player = 0; player < 2; player++) {
        auto it = turnSafe[player].find(label);
        if(it == turnSafe[player].end()) {
          fail(label + ": no turn-safe reference");
          continue;
        }
        for(int q : it->second)
          expected[q * C + NNInputs::V8_FEATURE_PLANES_START + player * enclosure::PER_PLAYER + 7] = 1.0f;
      }

    MiscNNInputParams params;
    Player pla = hist.presumedNextMovePla;
    NNInputs::fillRowV8(board, hist, pla, params, S, S, true, nhwc.data(), global.data());
    NNInputs::fillRowV8(board, hist, pla, params, S, S, false, nchw.data(), globalNchw.data());
    int wrong = 0;
    string firstWrong;
    for(int pos = 0; pos < S * S; pos++)
      for(int c = 0; c < C; c++) {
        float want = expected[pos * C + c];
        if(nhwc[pos * C + c] != want || nchw[c * S * S + pos] != want) {
          if(wrong++ == 0)
            firstWrong = "x " + Global::intToString(pos % S) + " y " + Global::intToString(pos / S) +
              " channel " + Global::intToString(c) + ": want " + Global::floatToString(want);
        }
      }
    if(wrong > 0)
      fail(label + ": " + Global::intToString(wrong) + " spatial entries differ, first at " + firstWrong);
    NNInputs::fillRowV8(board, hist, pla, params, S, S, true, nhwcReused.data(), globalReused.data(), &nhwcSet);
    if(nhwcReused != nhwc)
      fail(label + ": the NHWC row refilled over the previous position's differs from a fresh fill");
    NNInputs::fillRowV8(board, hist, pla, params, S, S, false, nchwReused.data(), globalReused.data(), &nchwSet);
    if(nchwReused != nchw)
      fail(label + ": the NCHW row refilled over the previous position's differs from a fresh fill");
    reusedChecks++;
    for(int i = 0; i < G; i++)
      if(global[i] != expectedGlobal[i] || globalNchw[i] != expectedGlobal[i])
        fail(label + ": global " + Global::intToString(i) + " is " + Global::floatToString(global[i]) +
             ", want " + Global::floatToString(expectedGlobal[i]));

    //The history planes enter the inputs, so they must enter the NN cache's hash too.
    bool hasHistory = false;
    for(int h = 0; h < enclosure::HISTORY_TURNS; h++)
      hasHistory = hasHistory || board.pos.history_n[h] > 0;
    if(hasHistory) {
      Board other = board;
      for(int h = 0; h < enclosure::HISTORY_TURNS; h++)
        other.pos.history_n[h] = 0;
      if(NNInputs::getHash(board, hist, pla, params) == NNInputs::getHash(other, hist, pla, params))
        fail(label + ": positions differing only in history hash alike");
      historyChecks++;
    }
  }
  printf("%d positions, %d with history, %d refilled, %d failures\n", positions, historyChecks, reusedChecks, failures);
  return failures == 0 && positions > 0 ? 0 : 1;
}
