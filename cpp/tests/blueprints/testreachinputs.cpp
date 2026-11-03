// Blueprints (the engine repository's abscls-gml5.4): the network's inputs version 15 (NNInputs V15), version 14's
// with the planes of reach after its planes (game/blueprints/gameencoding.h, write_reach):
//   - the packed row of version 15 at each allowed symmetry holds the packed row of version 14 at that symmetry in its
//     first planes, the planes of reach after them and 0 after those, and the same global inputs;
//   - the planes of reach at each symmetry are the reach kernel's (starreach.h) on the image of the position
//     (GameEncoding::sym_position), whose hands are the images of the hands: each star of the image's view on the
//     levels that its value passes (k > REACH_LEVELS[i]), the mover's on MY_REACH and the opponent's on THEIR_REACH,
//     and a star whose value is UNKNOWN or UNAVAILABLE on REACH_UNKNOWN alone;
//   - the float rows of version 15 (the training rows', NCHW and NHWC) hold the packed row's entries at the identity.
//
//   testreachinputs STATES [EVERY]
//
// STATES: a position's state a line, after its source's name and a tab, as the engine repository's
// games/blueprints/tools/star_reach_check.py --states writes them (GameRules::state_text), as benchstarreach reads
// them. The positions keep their views' orientations. EVERY: the test reads every EVERY-th line (default 1). It prints
// the positions, the rows and each plane of reach's marks. Exit code 0 when every check holds. Built by
// tests/enclosure/build.sh with TESTS=blueprints/testreachinputs, against a Blueprints build.

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "../../core/test.h"
#include "../../game/blueprints/starreach.h"
#include "../../game/boardhistory.h"
#include "../../neuralnet/nninputs.h"

using namespace std;

namespace {

const int S = NNPos::MAX_BOARD_LEN;
const int AREA = S * S;
const int V14 = NNInputs::NUM_FEATURES_SPATIAL_V14, V15 = NNInputs::NUM_FEATURES_SPATIAL_V15;
const int REACH = GameEncoding::NUM_REACH_SPATIAL_INPUTS;

bool bit(const vector<uint32_t>& bits, int i) { return (bits[size_t(i) >> 5] >> (i & 31)) & 1u; }

// The planes of reach of p's image under the symmetry, from the kernel on the image: plane c at (x, y) is
// planes[c * AREA + y * S + x].
vector<char> reference(const GameRules::Position& p, int symmetry) {
  const GameRules::Position image = GameEncoding::sym_position(p, symmetry);
  blueprints::StarReach reach;
  blueprints::star_reach(image, reach);
  const int mover = GameRules::next_player(image);
  const blueprints::View view = blueprints::view_of(image);
  vector<char> planes(size_t(REACH) * AREA, 0);
  for(int player = 0; player < 2; player++) {
    const int levels = (player == mover ? GameEncoding::MY_REACH : GameEncoding::THEIR_REACH) - GameEncoding::MY_REACH;
    for(const blueprints::StarReach::Star& s : reach.stars[player]) {
      int x, y;
      view.view_point(s.x, s.y, x, y);
      if(!blueprints::on_view(x, y))
        continue;
      const size_t point = size_t(y) * S + x;
      if(s.k == blueprints::StarReach::UNKNOWN || s.k == blueprints::StarReach::UNAVAILABLE) {
        planes[size_t(GameEncoding::REACH_UNKNOWN - GameEncoding::MY_REACH) * AREA + point] = 1;
        continue;
      }
      for(int i = 0; i < GameEncoding::NUM_REACH_LEVELS; i++)
        if(s.k > GameEncoding::REACH_LEVELS[i])
          planes[size_t(levels + i) * AREA + point] = 1;
    }
  }
  return planes;
}

}  // namespace

int main(int argc, char** argv) {
  if(argc < 2 || argc > 3) {
    fprintf(stderr, "usage: testreachinputs STATES [EVERY]\n");
    return 2;
  }
  const long long every = argc > 2 ? atoll(argv[2]) : 1;
  if(every < 1) {
    fprintf(stderr, "EVERY is 1 or more\n");
    return 2;
  }
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
  testAssert(REACH == 9 && V15 == V14 + REACH);
  testAssert(GameEncoding::MY_REACH + 1 == V14);  // the game's planes start after the front's on-board plane
  testAssert(NNInputs::NUM_FEATURES_GLOBAL_V15 == NNInputs::NUM_FEATURES_GLOBAL_V14);
  testAssert(NNInputs::gameSpatialFeatures(15) == V15);
  testAssert(NNInputs::gameSpatialWords(15) == NNInputs::NUM_SPATIAL_WORDS_V15);
  const int G = NNInputs::NUM_FEATURES_GLOBAL_V15;
  vector<uint32_t> bits14(NNInputs::gameSpatialWords(14)), bits15(NNInputs::NUM_SPATIAL_WORDS_V15);
  vector<float> glob14(G), glob15(G), nchw(size_t(V15) * AREA), nhwc(size_t(V15) * AREA);
  long long marks[REACH] = {}, positions = 0, rows = 0, lineNumber = 0;
  MiscNNInputParams params;
  string line;
  while(getline(in, line)) {
    if(lineNumber++ % every != 0)
      continue;
    const size_t tab = line.find('\t');
    GameRules::Position p;
    if(tab == string::npos || !GameRules::parse_state(line.substr(tab + 1), p, why)) {
      const string error = tab == string::npos ? "no source's name before a tab" : why;
      fprintf(stderr, "line %lld: %s\n", lineNumber, error.c_str());
      return 2;
    }
    Board board;
    board.setPosition(p);
    const BoardHistory hist(board, board.nextPla(), Rules(), 0);
    const Player pla = board.nextPla();
    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      if(!SymmetryHelpers::isAllowed(symmetry))
        continue;
      NNInputs::fillRowV14Packed(board, hist, pla, params, S, S, false, bits14.data(), glob14.data(), symmetry);
      NNInputs::fillRowV15Packed(board, hist, pla, params, S, S, false, bits15.data(), glob15.data(), symmetry);
      testAssert(glob15 == glob14);
      const int prefix = V14 * AREA;  // version 14's entries: whole words, then the bits of the last one
      for(int w = 0; w < prefix / 32; w++)
        testAssert(bits15[size_t(w)] == bits14[size_t(w)]);
      for(int i = prefix / 32 * 32; i < prefix; i++)
        testAssert(bit(bits15, i) == bit(bits14, i));
      const vector<char> expected = reference(board.pos, symmetry);
      for(int c = 0; c < REACH; c++)
        for(int point = 0; point < AREA; point++) {
          const bool got = bit(bits15, (V14 + c) * AREA + point);
          testAssert(got == (expected[size_t(c) * AREA + point] != 0));
          marks[c] += got;
        }
      for(int i = V15 * AREA; i < NNInputs::NUM_SPATIAL_WORDS_V15 * 32; i++)
        testAssert(!bit(bits15, i));
      rows++;
    }
    //The training rows' floats, NCHW and NHWC, at the identity.
    NNInputs::fillRowV15Packed(board, hist, pla, params, S, S, false, bits15.data(), glob15.data(), 0);
    NNInputs::fillRowV15(board, hist, pla, params, S, S, false, nchw.data(), glob14.data());
    testAssert(glob14 == glob15);
    NNInputs::fillRowV15(board, hist, pla, params, S, S, true, nhwc.data(), glob14.data());
    for(int c = 0; c < V15; c++)
      for(int point = 0; point < AREA; point++) {
        const float v = bit(bits15, c * AREA + point) ? 1.0f : 0.0f;
        testAssert(nchw[size_t(c) * AREA + point] == v);
        testAssert(nhwc[size_t(point) * V15 + c] == v);
      }
    positions++;
  }
  testAssert(positions > 0);
  printf("ok: %lld positions, %lld packed rows of version 15 against version 14's and the kernel on the image\n",
         positions, rows);
  printf("marks of each plane of reach (the mover's 4 levels, the opponent's 4, unknown):");
  for(int c = 0; c < REACH; c++)
    printf(" %lld", marks[c]);
  printf("\n");
  ScoreValue::freeTables();
  return 0;
}
