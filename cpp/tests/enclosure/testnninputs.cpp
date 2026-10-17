// Enclosure fork: the network inputs (NNInputs::fillRowV8) against the
// Enclosure repository's net.input_planes, position by position.
//
//   testnninputs tests/data/enclosure_inputs.txt TURN_SAFE_FILE
//
// The file comes from the Enclosure repository's games/enclosure/tools/export_katago_inputs.py
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

//V8's planes: the front's on-board plane, then the game's (GameEncoding): the edge planes, then the feature planes.
static const int V8_EDGE_PLANES_START = NNInputs::V8_GAME_PLANES_START;
static const int V8_FEATURE_PLANES_START = V8_EDGE_PLANES_START + GameEncoding::EDGE_PLANES;

//V8 channel of a Python plane (not one of the scalar planes).
static int v8Channel(int plane) {
  if(plane < PY_EDGE_PLANES)
    return V8_EDGE_PLANES_START + plane;
  assert(plane >= PY_FEATURES_START);
  return V8_FEATURE_PLANES_START + (plane - PY_FEATURES_START);
}

int main(int argc, char** argv) {
  if(argc != 3) {
    fprintf(stderr, "usage: testnninputs INPUTS_FILE TURN_SAFE_FILE\n");
    return 2;
  }
  //Plane 7 of each player holds its turn-safe points, from TURN_SAFE_FILE (the Python reference,
  //tools/export_turn_safe_fixture.py): "turnsafe <label> <player> <count> <y * 19 + x> ...".
  std::map<string, vector<int>> turnSafe[2];
  {
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
  //Packed rows (fillRowV8Packed, as NNEvaluator::fillRowBufs fills them since 2026-10-06), refilled position after
  //position: each must unpack to the float row exactly, with the padding bits 0.
  const int W = NNInputs::NUM_SPATIAL_WORDS_V8;
  static_assert(NNInputs::NUM_SPATIAL_ELTS_V8 == enclosure::SIZE * enclosure::SIZE * NNInputs::NUM_FEATURES_SPATIAL_V8, "");
  vector<uint32_t> nhwcBits(W, 0xFFFFFFFFu), nchwBits(W, 0xFFFFFFFFu);
  vector<float> unpacked(S * S * C), globalPacked(G);
  int packedChecks = 0;
  vector<float> symRow(S * S * C), symGlobal(G);
  int symChecks = 0;
  const int G9 = NNInputs::NUM_FEATURES_GLOBAL_V9;
  static_assert(NNInputs::NUM_FEATURES_SPATIAL_V9 == NNInputs::NUM_FEATURES_SPATIAL_V8, "");
  static_assert(NNInputs::NUM_FEATURES_GLOBAL_V9 == NNInputs::NUM_FEATURES_GLOBAL_V8 + 2, "");
  vector<float> v9Row(S * S * C), v9Global(G9), v9GlobalPacked(G9), v8Global(G);
  int v9Checks = 0;

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
      //Held ground and its openings (Python's planes 7 and 8 of each player) are not computed: plane 7 holds the
      //turn-safe points (below) and plane 8 stays zero.
      const int f = plane - PY_FEATURES_START;
      if(f >= 0 && (f % enclosure::PER_PLAYER == 7 || f % enclosure::PER_PLAYER == 8) && f < 2 * enclosure::PER_PLAYER)
        continue;
      expected[(index / PY_PLANES) * C + v8Channel(plane)] = 1.0f;
      //An edge plane's entry also at the edge's upper end, in the opposite offset.
      if(plane < 4 * enclosure::NOFF) {
        const int pos = index / PY_PLANES, group = plane / enclosure::NOFF, k = plane % enclosure::NOFF;
        const int dx = enclosure::OFFS[k].dx, dy = enclosure::OFFS[k].dy;
        const int upper = (pos / S + dy) * S + pos % S + dx;
        expected[upper * C + v8Channel(group * enclosure::NOFF + enclosure::OFF_INDEX[3 - dx][3 - dy])] = 1.0f;
      }
    }
    for(int player = 0; player < 2; player++) {
      auto it = turnSafe[player].find(label);
      if(it == turnSafe[player].end()) {
        fail(label + ": no turn-safe reference");
        continue;
      }
      for(int q : it->second)
        expected[q * C + V8_FEATURE_PLANES_START + player * enclosure::PER_PLAYER + 7] = 1.0f;
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
    for(int layout = 0; layout < 2; layout++) {
      const bool useNHWC = layout == 0;
      vector<uint32_t>& bits = useNHWC ? nhwcBits : nchwBits;
      NNInputs::fillRowV8Packed(board, hist, pla, params, S, S, useNHWC, bits.data(), globalPacked.data());
      NNInputs::unpackSpatialBits(bits.data(), unpacked.data(), S * S * C);
      if(unpacked != (useNHWC ? nhwc : nchw))
        fail(label + ": the packed " + (useNHWC ? "NHWC" : "NCHW") + " row differs from the float row");
      if(globalPacked != (useNHWC ? global : globalNchw))
        fail(label + ": the packed fill's global inputs differ");
      if((bits[W - 1] >> ((S * S * C) % 32)) != 0 && (S * S * C) % 32 != 0)
        fail(label + ": the packed row's padding bits are not 0");
    }
    packedChecks++;
    //The image's row written from board (fillRowV8Packed's symmetry) equals the row of the image board, at every
    //symmetry and in both layouts. (testgamelayer and testgameinterface check the map of the legal moves.)
    for(int sym = 1; sym < SymmetryHelpers::NUM_SYMMETRIES; sym++) {
      const Board symBoard = SymmetryHelpers::getSymBoard(board, sym);
      for(int layout = 0; layout < 2; layout++) {
        const bool useNHWC = layout == 0;
        NNInputs::fillRowV8(symBoard, hist, pla, params, S, S, useNHWC, symRow.data(), symGlobal.data());
        NNInputs::fillRowV8Packed(board, hist, pla, params, S, S, useNHWC, nhwcBits.data(), globalPacked.data(), sym);
        NNInputs::unpackSpatialBits(nhwcBits.data(), unpacked.data(), S * S * C);
        if(unpacked != symRow)
          fail(label + ": symmetry " + Global::intToString(sym) + (useNHWC ? " NHWC" : " NCHW") +
               ": the row written from the board differs from the image board's");
        if(globalPacked != symGlobal)
          fail(label + ": symmetry " + Global::intToString(sym) + ": the global inputs differ");
      }
      symChecks++;
    }
    for(int i = 0; i < G; i++)
      if(global[i] != expectedGlobal[i] || globalNchw[i] != expectedGlobal[i])
        fail(label + ": global " + Global::intToString(i) + " is " + Global::floatToString(global[i]) +
             ", want " + Global::floatToString(expectedGlobal[i]));

    //V9 (2026-10-13): V8's row and its global inputs, then the playout doubling advantage's two: 0 and 0 without an
    //advantage, 1 and half the advantage with one; the float and packed fills agree. V8 ignores the advantage, and the
    //advantage enters the NN cache's hash.
    for(double advantage : {0.0, 1.5, -2.25}) {
      MiscNNInputParams v9Params;
      v9Params.playoutDoublingAdvantage = advantage;
      std::fill(v9Global.begin(), v9Global.end(), -1.0f);
      NNInputs::fillRowV9(board, hist, pla, v9Params, S, S, true, v9Row.data(), v9Global.data());
      const string what = label + ": V9 with advantage " + Global::doubleToString(advantage);
      if(v9Row != nhwc)
        fail(what + ": the spatial row differs from V8's");
      for(int i = 0; i < G; i++)
        if(v9Global[i] != global[i])
          fail(what + ": global " + Global::intToString(i) + " differs from V8's");
      const float wantFlag = advantage != 0 ? 1.0f : 0.0f, wantValue = (float)(0.5 * advantage);
      if(v9Global[G] != wantFlag || v9Global[G + 1] != wantValue)
        fail(what + ": the advantage's inputs are " + Global::floatToString(v9Global[G]) + " and " +
             Global::floatToString(v9Global[G + 1]));
      std::fill(v9GlobalPacked.begin(), v9GlobalPacked.end(), -1.0f);
      NNInputs::fillRowV9Packed(board, hist, pla, v9Params, S, S, true, nhwcBits.data(), v9GlobalPacked.data());
      NNInputs::unpackSpatialBits(nhwcBits.data(), unpacked.data(), S * S * C);
      if(unpacked != nhwc || v9GlobalPacked != v9Global)
        fail(what + ": the packed fill differs from the float fill");
      NNInputs::fillRowV8(board, hist, pla, v9Params, S, S, true, v9Row.data(), v8Global.data());
      if(v8Global != global)
        fail(what + ": V8's global inputs depend on the advantage");
      if(advantage != 0 && NNInputs::getHash(board, hist, pla, v9Params) == NNInputs::getHash(board, hist, pla, params))
        fail(what + ": the advantage does not enter the hash");
      v9Checks++;
    }

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
  printf("%d positions, %d with history, %d refilled, %d packed, %d symmetric, %d V9, %d failures\n", positions,
         historyChecks, reusedChecks, packedChecks, symChecks, v9Checks, failures);
  return failures == 0 && positions > 0 ? 0 : 1;
}
