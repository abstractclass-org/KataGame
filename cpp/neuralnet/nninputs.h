#ifndef NEURALNET_NNINPUTS_H_
#define NEURALNET_NNINPUTS_H_

#include <memory>

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/rand.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/chosengame.h"
#include "../game/rules.h"

//Enclosure fork: the network's board, policy layout and score scale are the game's (game/gameinterface.h). The
//network's board is the game's largest board, which can be smaller than the build's (Board::MAX_LEN). The policy has
//POLICY_CHANNELS entries per point, plus pass last, and a policy index is an action. Spatial planes (inputs,
//ownership) still use xyToPos, one entry per point.
namespace NNPos {
  constexpr int MAX_BOARD_LEN = GameRules::MAX_BOARD_LEN;
  static_assert(MAX_BOARD_LEN <= Board::MAX_LEN, "the game's largest board fits in the build's");
  constexpr int MAX_BOARD_AREA = MAX_BOARD_LEN * MAX_BOARD_LEN;
  //KataGame fork: whether a board fills the network's board. A smaller board plays inside it (Connect-4's 7 x 6), so
  //the backends cannot skip the board's mask (requireExactNNLen).
  constexpr bool fillsNNBoard(int xSize, int ySize) { return xSize == MAX_BOARD_LEN && ySize == MAX_BOARD_LEN; }
  constexpr int POLICY_CHANNELS = GameEncoding::POLICY_CHANNELS;
  //Policy output adds +1 for the pass move: MAX_BOARD_AREA * POLICY_CHANNELS + 1 (the game interface checks it).
  constexpr int MAX_NN_POLICY_SIZE = GameRules::NUM_ACTIONS;
  constexpr int PASS_POS = GameRules::PASS_ACTION;
  //Extra score distribution radius, used for writing score in data rows and for the neural net score belief output
  constexpr int EXTRA_SCORE_DISTR_RADIUS = 60;
  //Enclosure fork: scores stay in points, and constants KataGo tuned for Go's point scale are scaled by SCORE_SCALE,
  //the game's score spread against Go's (game/gameinterface.h). The score distribution keeps its
  //2 * (MAX_BOARD_AREA + EXTRA_SCORE_DISTR_RADIUS) bins, each SCORE_SCALE wide.
  constexpr double SCORE_SCALE = GameEncoding::SCORE_SCALE;
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
  //Moves (Board::moveLocOfAction) and policy indices. The net is the game's largest board, so the index is the action.
  inline int getPassPos(int nnXLen, int nnYLen) { return nnXLen * nnYLen * POLICY_CHANNELS; }
  inline bool isPassPos(int pos, int nnXLen, int nnYLen) { return pos == getPassPos(nnXLen, nnYLen); }
  inline int getPolicySize(int nnXLen, int nnYLen) { return nnXLen * nnYLen * POLICY_CHANNELS + 1; }
  inline int locToPos(Loc loc, int boardXSize, int nnXLen, int nnYLen) {
    assert(boardXSize <= MAX_BOARD_LEN && nnXLen == MAX_BOARD_LEN && nnYLen == MAX_BOARD_LEN);
    (void)boardXSize;
    if(loc == Board::PASS_LOC)
      return getPassPos(nnXLen, nnYLen);
    else if(loc == Board::NULL_LOC)
      return getPolicySize(nnXLen, nnYLen);
    assert(loc >= Board::MOVE_LOC_BASE && loc < Board::NUM_MOVE_LOCS);
    return Board::actionOfMoveLoc(loc);
  }
  inline Loc posToLoc(int pos, int boardXSize, int boardYSize, int nnXLen, int nnYLen) {
    assert(boardXSize <= MAX_BOARD_LEN && boardYSize <= MAX_BOARD_LEN);
    assert(nnXLen == MAX_BOARD_LEN && nnYLen == MAX_BOARD_LEN);
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

  //KataGame fork: V8, the game's inputs (GameEncoding, game/gameinterface.h). Spatial: 0 on board (the backends'
  //mask, written by the front), then the game's NUM_SPATIAL_INPUTS planes (spatial_inputs). Global: the game's
  //NUM_GLOBAL_INPUTS (global_inputs). Enclosure's are its observation and feature planes
  //(game/enclosure/gameencoding.h).
  const int V8_GAME_PLANES_START = 1;
  const int NUM_FEATURES_SPATIAL_V8 = V8_GAME_PLANES_START + GameEncoding::NUM_SPATIAL_INPUTS;
  const int NUM_FEATURES_GLOBAL_V8 = GameEncoding::NUM_GLOBAL_INPUTS;
  //Enclosure fork (2026-10-13): V9, V8 with KataGo's playout doubling advantage as two more global inputs (V7's
  //inputs 15 and 16): 1 when the advantage is nonzero, and half the advantage, from the player to move's side. Self-play
  //gives one side more visits in some games (normalAsymmetricPlayoutProb), and its rows record the advantage. Model
  //version 18.
  const int NUM_FEATURES_SPATIAL_V9 = NUM_FEATURES_SPATIAL_V8;
  const int NUM_FEATURES_GLOBAL_V9 = NUM_FEATURES_GLOBAL_V8 + 2;
  //KataGame fork (2026-10-26): V10, V9 with the game's extended inputs (GameInterface: NUM_EXTENDED_SPATIAL_INPUTS and
  //NUM_EXTENDED_GLOBAL_INPUTS): its extended planes after its planes, and its extended global inputs after V9's. So a
  //new run reads more inputs, and the rows and the networks of V9 stay as they are. Model version 19.
  const int NUM_FEATURES_SPATIAL_V10 = NUM_FEATURES_SPATIAL_V9 + GameInterface::extendedSpatialInputs<GameEncoding>();
  const int NUM_FEATURES_GLOBAL_V10 = NUM_FEATURES_GLOBAL_V9 + GameInterface::extendedGlobalInputs<GameEncoding>();

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
  const int NUM_SPATIAL_ELTS_V8 = NUM_FEATURES_SPATIAL_V8 * NNPos::MAX_BOARD_AREA;
  const int NUM_SPATIAL_WORDS_V8 = (NUM_SPATIAL_ELTS_V8 + 31) / 32;
  //symmetry: the row of the image of board under that allowed symmetry, written from board itself (the game's
  //spatial_inputs at options.symmetry), equal to fillRowV8Packed(getSymBoard(board, symmetry), ...) without building
  //the image board.
  void fillRowV8Packed(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, uint32_t* rowBits, float* rowGlobal,
    int symmetry = 0
  );
  //V9: the V8 rows, with rowGlobal holding NUM_FEATURES_GLOBAL_V9 values.
  void fillRowV9(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal,
    std::vector<int>* setEntries = NULL
  );
  void fillRowV9Packed(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, uint32_t* rowBits, float* rowGlobal,
    int symmetry = 0
  );
  //V10: the rows of NUM_FEATURES_SPATIAL_V10 planes (NUM_SPATIAL_WORDS_V10 words packed), with rowGlobal holding
  //NUM_FEATURES_GLOBAL_V10 values.
  const int NUM_SPATIAL_ELTS_V10 = NUM_FEATURES_SPATIAL_V10 * NNPos::MAX_BOARD_AREA;
  const int NUM_SPATIAL_WORDS_V10 = (NUM_SPATIAL_ELTS_V10 + 31) / 32;
  void fillRowV10(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal,
    std::vector<int>* setEntries = NULL
  );
  void fillRowV10Packed(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, uint32_t* rowBits, float* rowGlobal,
    int symmetry = 0
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

//Enclosure fork (owner, 2026-10-05/06: compact policy storage): a policy read by position, as upstream reads its dense
//array: view[pos] is the probability of the move at pos, negative (-1) for an illegal move. Either dense
//(MAX_NN_POLICY_SIZE entries: root noise) or compact, read through a rank bitmap: per point, mask has bit `channel` set
//when that action is legal and start the slot of the point's first legal action, so action point * POLICY_CHANNELS +
//channel is at slot start[point] + popcount(mask[point] & ((1 << channel) - 1)) of prob; pass (the last position) at
//passSlot, -1 when illegal. Both list the legal positions ascending in pos[0, n): legalPos(k), probAt(k) walk them.
static_assert(NNPos::POLICY_CHANNELS <= 64, "a point's legal channels fit in one mask word");
static_assert(NNPos::MAX_NN_POLICY_SIZE <= 65536, "a policy index and a slot fit in 16 bits");
struct PolicyView {
  const float* dense;
  const uint64_t* mask;
  const uint16_t* start;
  const float* prob;
  int passSlot;
  const uint16_t* pos;
  int n;

  inline float operator[](int p) const {
    if(dense != NULL)
      return dense[p];
    if(p == NNPos::PASS_POS)
      return passSlot >= 0 ? prob[passSlot] : -1.0f;
    const int point = p / NNPos::POLICY_CHANNELS, channel = p - point * NNPos::POLICY_CHANNELS;
    const uint64_t m = mask[point];
    if(((m >> channel) & 1) == 0)
      return -1.0f;
    return prob[start[point] + __builtin_popcountll(m & ((uint64_t(1) << channel) - 1))];
  }
  inline int legalPos(int k) const { return pos[k]; }
  inline float probAt(int k) const { return dense != NULL ? dense[pos[k]] : prob[k]; }
  //MAX_NN_POLICY_SIZE entries into out, -1 for illegal moves (the dense array upstream holds).
  void toDense(float* out) const;
};

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

  //Enclosure fork (owner, 2026-10-05/06: compact policy storage): the policy, indexed by pos rather than loc, is kept
  //for the legal moves only (upstream: a dense MAX_NN_POLICY_SIZE array, negative for illegal moves, 69 KB here for
  //about 440 legal moves out of 17,330, held by every cached output and every expanded search node): legalPos the
  //legal positions ascending, legalProbs their probabilities, and the rank bitmap (policyMask, policyStart,
  //policyPassSlot; PolicyView) that finds a position's slot. Read it through policy() or getPolicyProbsMaybeNoised():
  //view[pos] is the probability, -1 for an illegal move, as the dense array gave. Root noise and temperature keep the
  //sign, so legalPos holds for the noised policy too; the search's new-child scan walks it.
  std::vector<uint16_t> legalPos;
  std::vector<float> legalProbs;
  uint64_t policyMask[NNPos::MAX_BOARD_AREA];
  uint16_t policyStart[NNPos::MAX_BOARD_AREA];
  int policyPassSlot;
  //The dense policy while an evaluation is in flight: the backend writes it and the evaluator post-processes it (the
  //requesting NNResultBuf's policyScratch), then compactPolicy() keeps the legal entries and sets this to NULL.
  float* policyDense;
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
  //Enclosure fork (2026-10-11): RGSC's second source, the swing regret's two heads (the 3rd and 4th regret heads), in
  //the same ranges. hasSwing is false when the model has none.
  bool hasSwing;
  float swingValue;
  float swingRanking;

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

  inline PolicyView policy() const {
    return PolicyView{NULL, policyMask, policyStart, legalProbs.data(), policyPassSlot, legalPos.data(), (int)legalPos.size()};
  }
  inline PolicyView getPolicyProbsMaybeNoised() const {
    if(noisedPolicyProbs == NULL)
      return policy();
    return PolicyView{noisedPolicyProbs, NULL, NULL, NULL, -1, legalPos.data(), (int)legalPos.size()};
  }
  //Keeps the entries of dense (MAX_NN_POLICY_SIZE, negative for illegal moves) at or above 0 as the compact policy.
  void setPolicyFromDense(const float* dense);
  //The compact policy from n listed positions (ascending, pass last if listed) and their probabilities, keeping those
  //with legal[j] set: the same result as setPolicyFromDense on a dense array holding exactly those.
  void setPolicyFromLegal(const uint16_t* pos, const float* probs, const char* legal, int n);
  //The in-flight dense policy (policyDense) made compact; policyDense is NULL after.
  void compactPolicy();
  void debugPrint(std::ostream& out, const Board& board) const;
  inline int getPos(Loc loc, const Board& board) const { return NNPos::locToPos(loc, board.x_size, nnXLen, nnYLen ); }
};

namespace SymmetryHelpers {
  //A symmetry is 3 bits flipY(bit 0), flipX(bit 1), transpose(bit 2). They are applied in that order.
  //The first four symmetries only reflect, and do not transpose X and Y.
  constexpr int NUM_SYMMETRIES = 8;
  constexpr int NUM_SYMMETRIES_WITHOUT_TRANSPOSE = 4;

  //KataGame fork: the game's allowed symmetries (GameEncoding::SYMMETRIES, bit s for symmetry s), the only ones that
  //the evaluator, the search and the positions' images use. The identity is always one. Enclosure allows all 8.
  constexpr unsigned ALLOWED_SYMMETRIES = GameEncoding::SYMMETRIES;
  constexpr int NUM_ALLOWED_SYMMETRIES = GameInterface::numSymmetries(ALLOWED_SYMMETRIES);
  inline bool isAllowed(int symmetry) { return GameInterface::hasSymmetry(ALLOWED_SYMMETRIES, symmetry); }
  //The allowed symmetries in ascending order. A random draw is an index into it: with all 8 allowed, the index i
  //is the symmetry i.
  constexpr std::array<int, NUM_ALLOWED_SYMMETRIES> ALLOWED =
    GameInterface::symmetryList<ALLOWED_SYMMETRIES>();

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
  //KataGame fork: an allowed symmetry. The image is the game's (GameEncoding::sym_position), and its legal moves are
  //the images of board's (SymActionMap).
  Board getSymBoard(const Board& board, int symmetry);

  //KataGame fork: the symmetries of moves (getSymLoc is for points only). A game's move need not map as its point
  //does (Enclosure's edge can be listed from its other end in the image), so the game maps it: the image of a legal
  //move of board, or pass, under an allowed symmetry, as a move of getSymBoard(board, symmetry), without building the
  //image board (GameEncoding::SymActionMap, built once for each board and symmetry).
  struct SymActionMap : GameEncoding::SymActionMap {
    SymActionMap(const Board& board, int sym) : GameEncoding::SymActionMap(board.pos, sym) {}
  };
  //The image of a legal move of board (a move Loc), pass or the null move, as SymActionMap maps it. board is the
  //board that the move is played on, as in getSymLoc.
  Loc getSymMoveLoc(Loc moveLoc, const Board& board, int symmetry);
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
  //KataGame fork: only the allowed symmetries.
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
