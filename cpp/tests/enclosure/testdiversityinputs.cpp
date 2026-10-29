// KataGame fork (2026-10-29): AZdb's latent (program/azdb.h) in the network's inputs and cache, for any game:
//   - NNInputs::getHash: member 0 keeps the hash of an evaluation without a member, and the members' hashes differ;
//   - the rows of the inputs version 14 are version 13's with the member's one-hot after version 13's global inputs,
//     the float rows and the packed rows at each allowed symmetry;
//   - the rows of version 13 do not read the member;
//   - the training rows: 80 global targets before version 14, and 84 in its rows (AZdb's 4 targets after them).
//
//   testdiversityinputs POSITIONS_FILE
//
// It names no game, so it builds against any game's build (tests/enclosure/build.sh BUILD_DIR, with
// TESTS=testdiversityinputs). A position is the game's actions from the start of its standard board, as testnnoutputs
// reads them (Blueprints': tests/data/blueprints_positions.txt). Exit code 0 when every check holds.

#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>

#include "../../core/test.h"
#include "../../dataio/trainingwrite.h"
#include "../../neuralnet/nninputs.h"

using namespace std;

int main(int argc, char** argv) {
  if(argc != 2) {
    fprintf(stderr, "usage: testdiversityinputs POSITIONS_FILE\n");
    return 2;
  }
  Board::initHash();
  ScoreValue::initTables();
  const int S = NNPos::MAX_BOARD_LEN;
  const int members = NNInputs::NUM_DIVERSITY_MEMBERS;
  const int n13 = NNInputs::NUM_FEATURES_GLOBAL_V13;
  testAssert(NNInputs::NUM_FEATURES_SPATIAL_V14 == NNInputs::NUM_FEATURES_SPATIAL_V13);
  testAssert(NNInputs::NUM_FEATURES_GLOBAL_V14 == n13 + members);
  testAssert(NNInputs::gameSpatialWords(14) == NNInputs::gameSpatialWords(13));
  const size_t spatial = (size_t)NNInputs::NUM_FEATURES_SPATIAL_V13 * S * S;
  const size_t words = NNInputs::gameSpatialWords(13);

  ifstream in(argv[1]);
  if(!in) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  string line;
  int positions = 0, rows = 0;
  vector<float> bin13(spatial), glob13(n13), bin(spatial), glob(NNInputs::NUM_FEATURES_GLOBAL_V14), globSym13(n13);
  vector<uint32_t> bits13(words), bits(words);
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
    if(!legal || hist.isGameFinished)
      continue;
    const Player pla = hist.presumedNextMovePla;

    //The cache's hash: member 0's is the default's, and the members' hashes differ.
    MiscNNInputParams params;
    const Hash128 none = NNInputs::getHash(board, hist, pla, params);
    set<Hash128> hashes;
    for(int m = 0; m < members; m++) {
      params.diversityMember = m;
      const Hash128 h = NNInputs::getHash(board, hist, pla, params);
      testAssert(m != 0 || h == none);
      hashes.insert(h);
    }
    testAssert((int)hashes.size() == members);

    //The rows: V14 is V13 with the member's one-hot, and V13 does not read the member.
    params.diversityMember = 0;
    NNInputs::fillRowV13(board, hist, pla, params, S, S, false, bin13.data(), glob13.data());
    for(int m = 0; m < members; m++) {
      params.diversityMember = m;
      NNInputs::fillRowV14(board, hist, pla, params, S, S, false, bin.data(), glob.data());
      testAssert(bin == bin13);
      testAssert(equal(glob13.begin(), glob13.end(), glob.begin()));
      for(int k = 0; k < members; k++)
        testAssert(glob[n13 + k] == (k == m ? 1.0f : 0.0f));
      NNInputs::fillRowV13(board, hist, pla, params, S, S, false, bin.data(), glob.data());
      testAssert(bin == bin13);
      testAssert(equal(glob13.begin(), glob13.end(), glob.begin()));
      for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
        if(!SymmetryHelpers::isAllowed(symmetry))
          continue;
        MiscNNInputParams zero;
        NNInputs::fillRowV13Packed(board, hist, pla, zero, S, S, false, bits13.data(), globSym13.data(), symmetry);
        NNInputs::fillRowV14Packed(board, hist, pla, params, S, S, false, bits.data(), glob.data(), symmetry);
        testAssert(bits == bits13);
        testAssert(equal(globSym13.begin(), globSym13.end(), glob.begin()));
        for(int k = 0; k < members; k++)
          testAssert(glob[n13 + k] == (k == m ? 1.0f : 0.0f));
        rows++;
      }
    }
    positions++;
  }
  testAssert(positions > 0);

  //The training rows' global targets: AZdb's 4 after the 80, in the rows of version 14 only.
  TrainingWriteBuffers v13(13, 1, NNInputs::NUM_FEATURES_SPATIAL_V13, n13, S, S, false);
  TrainingWriteBuffers v14(14, 1, NNInputs::NUM_FEATURES_SPATIAL_V14, NNInputs::NUM_FEATURES_GLOBAL_V14, S, S, false);
  testAssert(v13.numGlobalTargetChannels == 80 && v13.globalTargetsNC.shape[1] == 80);
  testAssert(v14.numGlobalTargetChannels == 84 && v14.globalTargetsNC.shape[1] == 84);

  ScoreValue::freeTables();
  printf("ok: %d positions, %d packed rows of version 14 against version 13's\n", positions, rows);
  return 0;
}
