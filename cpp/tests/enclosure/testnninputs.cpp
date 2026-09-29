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
  if(argc != 2) {
    fprintf(stderr, "usage: testnninputs INPUTS_FILE\n");
    return 2;
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
      expected[(index / PY_PLANES) * C + v8Channel(index % PY_PLANES)] = 1.0f;
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
  printf("%d positions, %d with history, %d failures\n", positions, historyChecks, failures);
  return failures == 0 && positions > 0 ? 0 : 1;
}
