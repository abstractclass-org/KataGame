#ifndef NEURALNET_NNINPUTS_H_
#define NEURALNET_NNINPUTS_H_

#include <memory>

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/rand.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/enclosure/features.h"
#include "../game/rules.h"

//Enclosure fork: the policy has one entry per edge, NUM_OFFSETS per point (the Enclosure action
//index (y * 19 + x) * 48 + offset), plus pass last. Spatial planes (inputs, ownership) still use
//xyToPos, one entry per point.
namespace NNPos {
  constexpr int MAX_BOARD_LEN = Board::MAX_LEN;
  constexpr int MAX_BOARD_AREA = MAX_BOARD_LEN * MAX_BOARD_LEN;
  constexpr int NUM_OFFSETS = enclosure::NOFF;
  //Policy output adds +1 for the pass move
  constexpr int MAX_NN_POLICY_SIZE = MAX_BOARD_AREA * NUM_OFFSETS + 1;
  static_assert(MAX_NN_POLICY_SIZE == enclosure::PASS + 1, "policy index = Enclosure action index");
  //Extra score distribution radius, used for writing score in data rows and for the neural net score belief output
  constexpr int EXTRA_SCORE_DISTR_RADIUS = 60;
  //Enclosure fork (owner): scores stay in points, and constants KataGo tuned for Go's point scale are scaled by
  //SCORE_SCALE, Enclosure's score spread against Go's (the D1 bin width: 12-point bins against Go's 1-point ones).
  //The score distribution keeps its 2 * (MAX_BOARD_AREA + EXTRA_SCORE_DISTR_RADIUS) bins, each SCORE_SCALE wide.
  constexpr double SCORE_SCALE = 12.0;
  //Score targets are capped at the distribution's range (KataGo: 361 + 60, "an upper bound for the plausible
  //final score difference").
  constexpr double SCORE_TARGET_CAP = (MAX_BOARD_AREA + EXTRA_SCORE_DISTR_RADIUS) * SCORE_SCALE;
  //Used various places we clip komi beyond board area.
  constexpr float KOMI_CLIP_RADIUS = 20.0f;

  inline int xyToPos(int x, int y, int nnXLen) { return y * nnXLen + x; }
  //A point's index in a spatial plane (KataGo's locToPos for points; locToPos is for moves here).
  inline int pointToPos(Loc loc, int boardXSize, int nnXLen) {
    assert(loc >= 0 && loc < Board::MAX_ARR_SIZE);
    return Location::getY(loc,boardXSize) * nnXLen + Location::getX(loc,boardXSize);
  }
  //Moves (Board::moveLocOfAction) and policy indices. The board and the net are both 19x19.
  inline int getPassPos(int nnXLen, int nnYLen) { return nnXLen * nnYLen * NUM_OFFSETS; }
  inline bool isPassPos(int pos, int nnXLen, int nnYLen) { return pos == getPassPos(nnXLen, nnYLen); }
  inline int getPolicySize(int nnXLen, int nnYLen) { return nnXLen * nnYLen * NUM_OFFSETS + 1; }
  inline int locToPos(Loc loc, int boardXSize, int nnXLen, int nnYLen) {
    assert(boardXSize == enclosure::SIZE && nnXLen == enclosure::SIZE && nnYLen == enclosure::SIZE);
    (void)boardXSize;
    if(loc == Board::PASS_LOC)
      return getPassPos(nnXLen, nnYLen);
    else if(loc == Board::NULL_LOC)
      return getPolicySize(nnXLen, nnYLen);
    assert(loc >= Board::MOVE_LOC_BASE && loc < Board::NUM_MOVE_LOCS);
    return Board::actionOfMoveLoc(loc);
  }
  inline Loc posToLoc(int pos, int boardXSize, int boardYSize, int nnXLen, int nnYLen) {
    assert(boardXSize == enclosure::SIZE && boardYSize == enclosure::SIZE && nnXLen == enclosure::SIZE && nnYLen == enclosure::SIZE);
    (void)boardXSize; (void)boardYSize;
    if(pos == getPassPos(nnXLen, nnYLen))
      return Board::PASS_LOC;
    if(pos < 0 || pos > getPassPos(nnXLen, nnYLen))
      return Board::NULL_LOC;
    return Board::moveLocOfAction(pos);
  }
}

namespace NNInputs {
  constexpr int SYMMETRY_NOTSPECIFIED = -1;
  constexpr int SYMMETRY_ALL = -2;
}

struct MiscNNInputParams {
  double drawEquivalentWinsForWhite = 0.5;
  bool conservativePassAndIsRoot = false;
  bool enablePassingHacks = false;
  double playoutDoublingAdvantage = 0.0;
  float nnPolicyTemperature = 1.0f;
  bool avoidMYTDaggerHack = false;
  // If no symmetry is specified, it will use default or random based on config, unless node is already cached.
  int symmetry = NNInputs::SYMMETRY_NOTSPECIFIED;
  double policyOptimism = 0.0;
  int maxHistory = 1000;
  // -1 = no override: pass-alive featurization follows hist.modes.alwaysComputePassAliveUnderSuicideRules.
  // 0/1 = override that flag for featurization (and its contribution to the nn cache hash). Used when
  // evaluating with a secondary net (e.g. a human SL profile net) whose declared featurization mode
  // differs from the mode the search itself is using.
  int passAliveSuicideRulesOverride = -1;
  // Same, but for hist.modes.excludeTerritoryAdjacentToAtari.
  int excludeTerritoryAdjAtariOverride = -1;

  // The effective value of modes.alwaysComputePassAliveUnderSuicideRules for featurization: the
  // override if one is set, else hist's own flag.
  bool getAlwaysComputePassAliveUnderSuicideRules(const BoardHistory& hist) const;
  // The effective value of modes.excludeTerritoryAdjacentToAtari for featurization: the override
  // if one is set, else hist's own flag.
  bool getExcludeTerritoryAdjacentToAtari(const BoardHistory& hist) const;
  // The effective full modes for featurization, taking both overrides into account.
  BoardHistoryModes getModes(const BoardHistory& hist) const;
  // The suicide legality to use for pass-alive area computations in featurization, taking the
  // override and hist.modes.alwaysComputePassAliveUnderSuicideRules into account.
  bool getSuicideLegalForPassAlive(const BoardHistory& hist) const;

  static const Hash128 ZOBRIST_CONSERVATIVE_PASS;
  static const Hash128 ZOBRIST_FRIENDLY_PASS;
  static const Hash128 ZOBRIST_PASSING_HACKS;
  static const Hash128 ZOBRIST_PLAYOUT_DOUBLINGS;
  static const Hash128 ZOBRIST_NN_POLICY_TEMP;
  static const Hash128 ZOBRIST_AVOID_MYTDAGGER_HACK;
  static const Hash128 ZOBRIST_POLICY_OPTIMISM;
  static const Hash128 ZOBRIST_ZERO_HISTORY;
};

namespace NNInputs {
  const int NUM_FEATURES_SPATIAL_V3 = 22;
  const int NUM_FEATURES_GLOBAL_V3 = 14;

  const int NUM_FEATURES_SPATIAL_V4 = 22;
  const int NUM_FEATURES_GLOBAL_V4 = 14;

  const int NUM_FEATURES_SPATIAL_V5 = 13;
  const int NUM_FEATURES_GLOBAL_V5 = 12;

  const int NUM_FEATURES_SPATIAL_V6 = 22;
  const int NUM_FEATURES_GLOBAL_V6 = 16;

  const int NUM_FEATURES_SPATIAL_V7 = 22;
  const int NUM_FEATURES_GLOBAL_V7 = 19;

  //Enclosure fork: V8, the Enclosure inputs (the Enclosure repository's net.input_planes, with its
  //scalars moved to the global inputs). Spatial: 0 on board (the backends' mask), then the
  //observation's 4 * 48 edge planes (each player's edges, then each player's protected or
  //just-placed edges, by source point and offset), then the feature planes (planes.feature_planes:
  //9 per player, then 4 history). Global: the observation's 5 scalars.
  //Absolute colours, as the Enclosure network: plane k is Blue's (P_BLACK) for k = 0 and Red's for 1.
  const int V8_EDGE_PLANES_START = 1;
  const int V8_FEATURE_PLANES_START = V8_EDGE_PLANES_START + 4 * enclosure::NOFF;
  const int NUM_FEATURES_SPATIAL_V8 = V8_FEATURE_PLANES_START + enclosure::FEATURE_PLANES;
  const int NUM_FEATURES_GLOBAL_V8 = enclosure::SCALARS;

  Hash128 getHash(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams
  );

  void fillRowV3(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV4(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV5(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV6(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV7(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  //Enclosure fork (owner, 2026-10-05): with setEntries, rowBin must hold exactly 1.0 at the entries listed there and
  //0 elsewhere (an empty list: all zeros), as this function leaves it; only those entries are cleared, not the whole
  //row (77,615 floats, about 8% of self-play's CPU), and the list is replaced by the entries set now. Without it,
  //the whole row is cleared.
  void fillRowV8(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal,
    std::vector<int>* setEntries = NULL
  );
  //Enclosure fork (owner, 2026-10-06; iviazovetskyi's packed inputs): the same row as fillRowV8, one bit per entry,
  //since every spatial entry is 0 or 1: bit i of the row (rowBits[i / 32] >> (i % 32) & 1) is fillRowV8's entry i, in
  //the same layout (useNHWC). The row is NUM_SPATIAL_WORDS_V8 words, cleared whole here; padding bits stay 0.
  //rowGlobal as fillRowV8. The GPU backends send these 9.7 KB instead of 310 KB of floats and unpack them there.
  const int NUM_SPATIAL_ELTS_V8 = NUM_FEATURES_SPATIAL_V8 * enclosure::SIZE * enclosure::SIZE;
  const int NUM_SPATIAL_WORDS_V8 = (NUM_SPATIAL_ELTS_V8 + 31) / 32;
  void fillRowV8Packed(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, uint32_t* rowBits, float* rowGlobal
  );
  //The packed row's first n entries as floats (0 or 1) into out.
  void unpackSpatialBits(const uint32_t* rowBits, float* out, int n);

  //If groupTax is specified, for each color region of area, reduce weight on empty spaces equally to reduce the total sum by 2.
  //(but should handle seki correctly)
  void fillScoring(
    const Board& board,
    const Color* area,
    bool groupTax,
    float* scoring
  );
}

struct NNOutput {
  Hash128 nnHash; //NNInputs - getHash

  //Initially from the perspective of the player to move at the time of the eval, fixed up later in nnEval.cpp
  //to be the value from white's perspective.
  //These three are categorial probabilities for each outcome.
  float whiteWinProb;
  float whiteLossProb;
  float whiteNoResultProb;

  //The first two moments of the believed distribution of the expected score at the end of the game, from white's perspective.
  float whiteScoreMean;
  float whiteScoreMeanSq;
  //Points to make game fair
  float whiteLead;
  //Expected arrival time of remaining game variance, in turns, weighted by variance, only when modelVersion >= 9
  float varTimeLeft;
  //A metric indicating the "typical" error in the winloss value or the score that the net expects, relative to the
  //short-term future MCTS value.
  float shorttermWinlossError;
  float shorttermScoreError;

  //Indexed by pos rather than loc
  //Values in here will be set to negative for illegal moves, including superko
  float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
  //Enclosure fork (2026-10-05): the positions with policyProbs >= 0 (the legal moves), ascending, as the evaluator set
  //them; root noise and temperature keep the sign, so they hold for getPolicyProbsMaybeNoised too. Empty when not
  //computed (averaged outputs, tests): the search then scans every position, as upstream. About 440 legal moves
  //against 17,330 positions, so the search's new-child scan reads the list (selectBestChildToDescend).
  std::vector<uint16_t> legalPos;
  //The optimism value used for this neural net evaluation.
  float policyOptimismUsed;

  int nnXLen;
  int nnYLen;
  //If not NULL, then this contains a nnXLen*nnYLen-sized map of expected ownership on the board.
  float* whiteOwnerMap;

  //If not NULL, then contains policy with dirichlet noise or any other noise adjustments for this node
  float* noisedPolicyProbs;

  //Enclosure fork (owner, 2026-10-04): RGSC's heads (ValueHeadDesc::regretHeads). hasRegret is false when the model
  //has none. The regret value is in [0, 4]; for the ranking score only the order matters.
  bool hasRegret;
  float regretValue;
  float regretRanking;

  //Enclosure fork (owner, 2026-10-04): the exported training-only outputs (ValueHeadDesc::auxVersion), when requested
  //(NNResultBuf::includeAuxOutputs) and the model has them, else NULL. Raw outputs (before any tanh, softmax or
  //multiplier), from the player to move's view, with the model's output scale applied.
  //auxMap: NUM_AUX_MAP_CHANNELS channels of nnXLen*nnYLen, mapped back to the board like whiteOwnerMap: 0-1 the
  //future positions, 2-3 the territory-income logits (mover's, opponent's), 4-7 the cut-edge logits, 8 the future
  //occupancy. auxValues: NUM_AUX_VALUES, the misc value outputs 4-9 then the more-misc 2-7 (TD values and scores).
  static constexpr int NUM_AUX_MAP_CHANNELS = 9;
  static constexpr int NUM_AUX_VALUES = 12;
  float* auxMap;
  float* auxValues;

  //Export stage 2 (owner, 2026-10-04), with the aux outputs above (NULL without them):
  //auxPolicy: NUM_AUX_POLICIES blocks of MAX_NN_POLICY_SIZE, policy outputs 1-4 of model_pytorch.PolicyHead (the
  //opponent's reply, the soft policy, the soft opponent reply, the long-term optimistic policy) as raw logits with the
  //output scale applied, laid out and mapped back like policyProbs; entries for illegal actions are 0.
  //scoreBelief: scoreBeliefLen probabilities over the final margin from the player to move's view, bin i centred
  //at (i - scoreBeliefLen/2 + 0.5) bins of the model's bin width (NNEvaluator::getScoreBinWidth).
  //scoreBeliefRaw: the backend's part of the score belief (s2, then the mixture logits), consumed by NNEvaluator.
  static constexpr int NUM_AUX_POLICIES = 4;
  float* auxPolicy;
  int scoreBeliefLen;
  float* scoreBelief;
  float* scoreBeliefRaw;

  NNOutput(); //Does NOT initialize values
  NNOutput(const NNOutput& other);
  ~NNOutput();

  //Averages the others. Others must be nonempty and share the same nnHash (i.e. be for the same board, except if somehow the hash collides)
  //Does NOT carry over noisedPolicyProbs.
  NNOutput(const std::vector<std::shared_ptr<NNOutput>>& others);

  NNOutput& operator=(const NNOutput&);

  inline float* getPolicyProbsMaybeNoised() { return noisedPolicyProbs != NULL ? noisedPolicyProbs : policyProbs; }
  inline const float* getPolicyProbsMaybeNoised() const { return noisedPolicyProbs != NULL ? noisedPolicyProbs : policyProbs; }
  void debugPrint(std::ostream& out, const Board& board) const;
  inline int getPos(Loc loc, const Board& board) const { return NNPos::locToPos(loc, board.x_size, nnXLen, nnYLen ); }
};

namespace SymmetryHelpers {
  //A symmetry is 3 bits flipY(bit 0), flipX(bit 1), transpose(bit 2). They are applied in that order.
  //The first four symmetries only reflect, and do not transpose X and Y.
  constexpr int NUM_SYMMETRIES = 8;
  constexpr int NUM_SYMMETRIES_WITHOUT_TRANSPOSE = 4;

  //These two IGNORE transpose if hSize and wSize do not match. So non-square transposes are disallowed.
  //copyOutputsWithSymmetry performs the inverse of symmetry.
  void copyInputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int cSize, bool useNHWC, int symmetry);
  void copyOutputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int symmetry);

  //Applies a symmetry to a location
  Loc getSymLoc(int x, int y, const Board& board, int symmetry);
  Loc getSymLoc(Loc loc, const Board& board, int symmetry);
  Loc getSymLoc(int x, int y, int xSize, int ySize, int symmetry);
  Loc getSymLoc(Loc loc, int xSize, int ySize, int symmetry);

  //Applies a symmetry to a board
  Board getSymBoard(const Board& board, int symmetry);

  //Enclosure fork: symmetries of edges, positions and moves (getSymLoc is for points only).
  //An edge maps to the edge between its endpoints' images, canonical again, so a flip can re-anchor
  //it at its other end: the input planes, which hold an edge at its lower end, are not a plain
  //permutation of pixels and channels. So the fork transforms positions, not input planes.
  enclosure::Edge getSymEdge(const enclosure::Edge& e, int symmetry);
  //Every edge (both players', protected, placed, history) mapped; scores and turn state kept.
  enclosure::Position getSymPosition(const enclosure::Position& p, int symmetry);
  //A move's policy index (Enclosure action) under a symmetry, as a move of symBoard, the image of the
  //board it is played on. Between two of the mover's nodes only one direction is listed, and the
  //image may be listed from its other end: then that one is returned.
  int getSymAction(int action, const Board& symBoard, int symmetry);
  Loc getSymMoveLoc(Loc moveLoc, const Board& symBoard, int symmetry);
  //A sequence of moves from board (a principal variation) mapped move by move, on the image boards.
  std::vector<Loc> getSymMoveLocs(const Board& board, const std::vector<Loc>& moves, int symmetry);

  //Get the inverse of the specified symmetry
  int invert(int symmetry);
  //Get the symmetry equivalent to first applying firstSymmetry and then applying nextSymmetry.
  int compose(int firstSymmetry, int nextSymmetry);
  int compose(int firstSymmetry, int nextSymmetry, int nextNextSymmetry);

  inline bool isTranspose(int symmetry) { return (symmetry & 0x4) != 0; }
  inline bool isFlipX(int symmetry) { return (symmetry & 0x2) != 0; }
  inline bool isFlipY(int symmetry) { return (symmetry & 0x1) != 0; }

  //Fill isSymDupLoc with true on all but one copy of each symmetrically equivalent move, and false everywhere else.
  //Enclosure fork: isSymDupLocs should be an array of size Board::NUM_MOVE_LOCS (move Locs)
  //If onlySymmetries is not NULL, will only consider the symmetries specified there.
  //validSymmetries will be filled with all symmetries of the current board, including using history for checking ko/superko and some encore-related state.
  //This implementation is dependent on specific order of the symmetries (i.e. transpose is coded as 0x4)
  //Will pretend moves that have a nonzero value in avoidMoves do not exist.
  void markDuplicateMoveLocs(
    const Board& board,
    const BoardHistory& hist,
    const std::vector<int>* onlySymmetries,
    const std::vector<int>& avoidMoves,
    bool* isSymDupLoc,
    std::vector<int>& validSymmetries
  );

// For each symmetry, return a metric about the "amount" of difference that board would have with other
// if symmetry were applied to board.
  void getSymmetryDifferences(
    const Board& board, const Board& other, double maxDifferenceToReport, double symmetryDifferences[NUM_SYMMETRIES]
  );
}

//Utility functions for computing the "scoreValue", the unscaled utility of various numbers of points, prior to multiplication by
//staticScoreUtilityFactor or dynamicScoreUtilityFactor (see searchparams.h)
namespace ScoreValue {
  //MUST BE CALLED AT PROGRAM START!
  void initTables();
  void freeTables();

  //The number of wins a game result should count as
  double whiteWinsOfWinner(Player winner, double drawEquivalentWinsForWhite);
  //The score difference that a game result should count as on average
  double whiteScoreDrawAdjust(double finalWhiteMinusBlackScore, double drawEquivalentWinsForWhite, const BoardHistory& hist);

  //The unscaled utility of achieving a certain score difference
  //Enclosure fork: the search's score utility passes Board::scoreUnit() as sqrtBoardArea (the unit the scale multiplies;
  //KataGo's is the board's side), owner 2026-10-03.
  double whiteScoreValueOfScoreSmooth(double finalWhiteMinusBlackScore, double center, double scale, double drawEquivalentWinsForWhite, double sqrtBoardArea, const BoardHistory& hist);
  double whiteScoreValueOfScoreSmoothNoDrawAdjust(double finalWhiteMinusBlackScore, double center, double scale, double sqrtBoardArea);
  //Approximately invert whiteScoreValueOfScoreSmooth
  double approxWhiteScoreOfScoreValueSmooth(double scoreValue, double center, double scale, double sqrtBoardArea);
  //The derviative of whiteScoreValueOfScoreSmoothNoDrawAdjust with respect to finalWhiteMinusBlackScore.
  double whiteDScoreValueDScoreSmoothNoDrawAdjust(double finalWhiteMinusBlackScore, double center, double scale, double sqrtBoardArea);

  //Compute what the scoreMeanSq should be for a final game result
  //It is NOT simply the same as finalWhiteMinusBlackScore^2 because for integer komi we model it as a distribution where with the appropriate probability
  //you gain or lose 0.5 point to achieve the desired drawEquivalentWinsForWhite, so it actually has some variance.
  double whiteScoreMeanSqOfScoreGridded(double finalWhiteMinusBlackScore, double drawEquivalentWinsForWhite);

  //The expected unscaled utility of the final score difference, given the mean and stdev of the distribution of that difference,
  //assuming roughly a normal distribution.
  double expectedWhiteScoreValue(double whiteScoreMean, double whiteScoreStdev, double center, double scale, double sqrtBoardArea);

  //Get the standard deviation of score given the E(score) and E(score^2)
  double getScoreStdev(double scoreMeanAvg, double scoreMeanSqAvg);
}

#endif  // NEURALNET_NNINPUTS_H_
