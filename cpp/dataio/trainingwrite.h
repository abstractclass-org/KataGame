#ifndef DATAIO_TRAINING_WRITE_H_
#define DATAIO_TRAINING_WRITE_H_

#include <atomic>

#include "../dataio/numpywrite.h"
#include "../neuralnet/nninputs.h"
#include "../program/azdb.h"
#include "../neuralnet/sgfmetadata.h"
#include "../neuralnet/nninterface.h"

STRUCT_NAMED_PAIR(Loc,loc,int16_t,policyTarget,PolicyTargetMove);
STRUCT_NAMED_PAIR(std::vector<PolicyTargetMove>*,policyTargets,int64_t,unreducedNumVisits,PolicyTarget);

STRUCT_NAMED_QUAD(Loc,loc,float,winLoss,float,score,int64_t,visits,QValueTargetMove);

struct QValueTargets {
  std::vector<QValueTargetMove> targets;
  QValueTargets() = default;
  ~QValueTargets() = default;
};

//Summary of value-head-related training targets for outputted data.
struct ValueTargets {
  //As usual, these are from the perspective of white.
  float win;
  float loss;
  float noResult;
  float score;
  bool hasLead;
  float lead;

  ValueTargets();
  ~ValueTargets();
};

//Some basic extra stats to record outputted data about the neural net's raw evaluation on the position.
struct NNRawStats {
  double whiteWinLoss;
  double whiteScoreMean;
  double policyEntropy;
};

//Info about the post-game reanalysis (full re-search) of a position that originally got only a cheap search.
struct ReanalysisData {
  bool wasReanalyzed = false;
  //If false, the targets derived from the final board and the actual game continuation (ownership, final
  //score, future board positions, and the next-move policy target) are not recorded for this row. The value
  //targets always come from the game's value target array, whose entry at this turn is the reanalysis
  //search's values but whose later entries and final outcome still reflect the actual game.
  bool usedOutcomeTargets = true;
  //The policy surprise and value surprise of the original cheap search on this turn, which are the stats
  //that drove the probability of this position being selected for reanalysis. The value surprise is computed
  //by whichever definition was configured (see PlaySettings::useSearchValueSurprise).
  float selectionPolicySurprise = 0.0f;
  float selectionValueSurprise = 0.0f;
  //The number of visits of the original cheap search, whose recorded targets were replaced.
  int64_t originalNumVisits = 0;
  //Number of neural net changes this game before this reanalysis search was performed.
  int numNeuralNetChangesSoFar = 0;
};

//A side position that was searched off the main line of the game, to give some data about an alternative move.
struct SidePosition {
  Board board;
  BoardHistory hist;
  Player pla;
  int64_t unreducedNumVisits;
  std::vector<PolicyTargetMove> policyTarget;
  double policySurprise;
  double policyEntropy;
  double searchEntropy;
  ValueTargets whiteValueTargets;
  QValueTargets whiteQValueTargets;
  NNRawStats nnRawStats;
  float targetWeight;
  float targetWeightUnrounded;
  int numNeuralNetChangesSoFar; //Number of neural net changes this game before the creation of this side position
  Player playoutDoublingAdvantagePla;
  double playoutDoublingAdvantage;

  SidePosition();
  SidePosition(const Board& board, const BoardHistory& hist, Player pla, int numNeuralNetChangesSoFar);
  ~SidePosition();
};

STRUCT_NAMED_PAIR(std::string,name,int,turnIdx,ChangedNeuralNet);

struct FinishedGameData {
  std::string bName;
  std::string wName;
  int bIdx;
  int wIdx;

  Board startBoard; //Board as of the end of startHist, beginning of training period
  BoardHistory startHist; //Board history as of start of training period
  BoardHistory endHist; //Board history as of end of training period
  Player startPla; //Player to move as of end of startHist.
  Hash128 gameHash;

  double drawEquivalentWinsForWhite;
  Player playoutDoublingAdvantagePla;
  double playoutDoublingAdvantage;
  bool hitTurnLimit;

  //Metadata about how the game was initialized
  int numExtraBlack;
  int mode;
  int beganInEncorePhase;
  int usedInitialPosition;
  //This differs from numExtraBlack in that numExtraBlack counts number of extra black stones
  //played following the start of startHist, whereas handicapForSgf counts from startBoard.
  //So on things like forked handicap games this one will be larger. Also this one does the
  //whole +1 thing, skipping 1H.
  int handicapForSgf;

  //If false, then we don't have these below vectors and ownership information
  bool hasFullData;
  std::vector<float> targetWeightByTurn;
  std::vector<float> targetWeightByTurnUnrounded;
  std::vector<PolicyTarget> policyTargetsByTurn;
  std::vector<double> policySurpriseByTurn;
  std::vector<double> policyEntropyByTurn;
  std::vector<double> searchEntropyByTurn;
  std::vector<ValueTargets> whiteValueTargetsByTurn; //Except this one, we may have some of
  std::vector<QValueTargets> whiteQValueTargetsByTurn;
  std::vector<NNRawStats> nnRawStatsByTurn;
  std::vector<ReanalysisData> reanalysisByTurn; //May be empty if no reanalysis was configured
  //Statistical info, not used for training targets. The value surprise of each turn as used for surprise-based
  //data weighting (if reanalysis happened, recomputed afterward, so reanalyzed turns reflect the reanalysis
  //search rather than the value surprise that drove their selection), and whether each turn's search during
  //the game was a cheap search.
  std::vector<double> valueSurpriseByTurn;
  std::vector<bool> wasCheapSearchByTurn;

  //Enclosure fork (owner, 2026-10-04): RGSC (program/rgsc.h). Per turn, the search value (white's win minus loss)
  //of the move played and the network's regret ranking of the position (NaN without the head); and, set after the
  //game by the self-play loop, each turn's regret target (Eq. 2), written to the rows as C70 (flag C71). Empty when
  //RGSC is off.
  std::vector<float> selectedWhiteWinLossByTurn;
  std::vector<float> regretRankingByTurn;
  std::vector<float> regretByTurn;
  //KataGame fork (2026-10-27): per turn, whether its action continues the move of the action before it
  //(GameInterface::continuesMove: Blueprints' drop), for RGSC's regret by kind of action (Rgsc::Buffer::addRegrets).
  std::vector<bool> continuesMoveByTurn;
  //Enclosure fork (2026-10-11): RGSC's second source, value swing. Per turn, the network's swing ranking of the
  //position (NaN without the swing heads); and, set after the game, each turn's swing regret target (C72, flag C73).
  //rgscSource: the buffer the game restarted from (Rgsc::Source; C74 of every row of the game).
  std::vector<float> swingRankingByTurn;
  std::vector<float> swingByTurn;
  int rgscSource = 0;
  //KataGame fork (2026-10-29): AZdb (program/azdb.h), for the rows of the inputs version 14. diversityMembers: the
  //member of each player, by the rules' player (Board::indexOfPlayer), whose latent the player's rows get.
  //diversityState: the game's copy of the members' state, taken at its start, from which the rows' rewards and the
  //targets of the intrinsic value head come (NULL: none, and those targets have weight 0). intrinsicAByTurn and
  //intrinsicBByTurn: per turn, the outputs A and B of the network's evaluation at the root of the search (NaN without
  //the head), which the targets read where their span ends.
  int diversityMembers[2] = {0, 0};
  std::shared_ptr<const AZdb::State<GameEncoding>> diversityState;
  //KataGame fork (2026-10-29): whether each player's rows train, by the rules' player, its main rows and its side
  //positions' rows. AZdb's matchmaker (program/azdbleague.h) trains only the chooser's rows, both players' in a game of
  //the chooser against itself. Both by default. Play::runGame sets it from its bots (MatchPairer::BotSpec::trains) and
  //gives the turns of a player that does not train the weight 0.
  bool diversityTrains[2] = {true, true};
  std::vector<float> intrinsicAByTurn;
  std::vector<float> intrinsicBByTurn;
  //KataGame fork (abscls-il4e.11): the view in which each move of endHist names its action (GameInterface::setView),
  //by the move's index in endHist.moveHistory: "" for the rules' view. Self-play's search of the root's views
  //(PlaySettings::rootViews) names each move in the view of the search that chose it, so the turn's policy target and
  //row are in that view too. Empty without it: every move is in the rules' view. Replays play each move in its view
  //(playMove), and the records name a move in a view as GTP does (MOVE/VIEW, WriteSgf).
  std::vector<std::string> moveViews;

  Color* finalFullArea;
  Color* finalOwnership;
  bool* finalSekiAreas;
  float* finalWhiteScoring;

  double trainingWeight;

  std::vector<SidePosition*> sidePositions;
  std::vector<ChangedNeuralNet*> changedNeuralNets;

  double bTimeUsed;
  double wTimeUsed;
  int bMoveCount;
  int wMoveCount;

  static constexpr int NUM_MODES = 8;
  static constexpr int MODE_NORMAL = 0;
  static constexpr int MODE_CLEANUP_TRAINING = 1;
  static constexpr int MODE_FORK = 2;
  static constexpr int MODE_HANDICAP = 3;
  static constexpr int MODE_SGFPOS = 4;
  static constexpr int MODE_HINTPOS = 5;
  static constexpr int MODE_HINTFORK = 6;
  static constexpr int MODE_ASYM = 7;

  FinishedGameData();
  ~FinishedGameData();

  void printDebug(std::ostream& out) const;

  //KataGame fork (abscls-il4e.11): the view of endHist's move moveIdx (moveViews), "" for the rules' view.
  const std::string& moveView(size_t moveIdx) const {
    static const std::string rulesView;
    return moveIdx < moveViews.size() ? moveViews[moveIdx] : rulesView;
  }
  //Sets the view of endHist's move moveIdx on board, the position before the move (setBoardView).
  void setMoveView(size_t moveIdx, Board& board) const;
  //Plays endHist's move moveIdx on board and hist, in its view.
  void playMove(size_t moveIdx, Board& board, BoardHistory& hist) const;
  //Sets a view of the game's on board (GameInterface::setView): "" leaves board as it is. The view must be one.
  static void setBoardView(Board& board, const std::string& view);
};

struct TrainingWriteBuffers {
  int inputsVersion;
  int maxRows;
  int numBinaryChannels;
  int numGlobalChannels;
  int numGlobalTargetChannels;  //80, or 84 from the inputs version 14 (globalTargetsNC's C80-83)
  int dataXLen;
  int dataYLen;
  int packedBoardArea;

  bool hasMetadataInput;

  int curRows;
  float* binaryInputNCHWUnpacked;

  //Input feature planes that have spatial extent, all of which happen to be binary.
  //Packed bitwise, with each (HW) zero-padded to a round byte.
  //Within each byte, bits are packed bigendianwise, since that's what numpy's unpackbits will expect.
  NumpyBuffer<uint8_t> binaryInputNCHWPacked;
  //Input features that are global.
  NumpyBuffer<float> globalInputNC;

  //Policy targets
  //Shape is [N,C,Pos]. Almost NCHW, except we have a Pos of length, e.g. 362, due to the pass input, instead of 19x19.
  //Contains number of visits, possibly with a subtraction.
  //Channel i will still be a dummy probability distribution (not all zero) if weight 0
  //C0: Policy target this turn.
  //C1: Policy target next turn.
  NumpyBuffer<int16_t> policyTargetsNCMove;

  //Value targets and other metadata, from the perspective of the player to move
  //C0-3: Categorial game result, win,loss,noresult, and also score. Draw is encoded as some blend of win and loss based on drawEquivalentWinsForWhite.
  //C4-7: MCTS win-loss-noresult estimate td-like target, lambda = 1 - 1/(1 + boardArea * 0.176)
  //C8-11: MCTS win-loss-noresult estimate td-like target, lambda = 1 - 1/(1 + boardArea * 0.056)
  //C12-15: MCTS win-loss-noresult estimate td-like target, lambda = 1 - 1/(1 + boardArea * 0.016)
  //KataGame fork: boardArea is the game's look-ahead area (Board::lookaheadArea).
  //C16-19: MCTS win-loss-noresult estimate td-like target, lambda = 0 (so, actually just the immediate MCTS result).

  //C20: Actual final score, from the perspective of the player to move, adjusted for draw utility, zero if C27 is zero.
  //C21: Lead in points, number of points to make the game fair, zero if C29 is zero.
  //C22: Expected arrival time of WL variance.
  //C23: Unused
  //C24: 1.0 minus weight assigned to td value targets

  //C25 Weight multiplier for row as a whole

  //C26: Weight assigned to the policy target
  //C27: Weight assigned to the final board ownership target and score distr targets. Most training rows will have this be 1, some will be 0.
  //C28: Weight assigned to the next move policy target
  //C29: Weight assigned to the lead target
  //C30: Policy Surprise (for statistical purposes)
  //C31: Policy Entropy (for statistical purposes)
  //C32: Search Entropy (for statistical purposes)
  //C33: Weight assigned to the future position targets valueTargetsNCHW C1-C2
  //C34: Weight assigned to the area/territory target valueTargetsNCHW C4
  //     KataGame fork: valueTargetWeight when C4 holds the game's target (GameInterface::TARGET_SCORING), else 0.
  //C35: 1.0 minus weight assigned to value targets

  //C36-40: Precomputed mask values indicating if we should use historical moves 1-5, if we desire random history masking.
  //1 means use, 0 means don't use.

  //C41-46: 128-bit hash identifying the game, different rows from the same game share the same value.
  //Split into chunks of 22, 22, 20, 22, 22, 20 bits, little-endian style (since floats have > 22 bits of precision).

  //C47: Komi, adjusted for draw utility and points costed or paid so far, from the perspective of the player to move.
  //C48: 1 if we're in an area-scoring-like phase of the game (area scoring or second encore territory scoring)

  //C49: 1 if an earlier neural net started this game, compared to the latest in this data file.
  //C50: If positive, an earlier neural net was playing this specific move, compared to the latest in this data file.

  //C51: Turn idx of the game right now, zero-indexed. Starts at 0 even for sgfposes.
  //C52: Did this game end via hitting turn limit?
  //C53: First turn of this game that was selfplay for training rather than initialization (e.g. handicap stones, random init of the starting board pos)
  //C54: Number of extra moves black got at the start (i.e. handicap games)

  //C55: Game type, how the game was initialized
  //0 = normal self-play game.
  //1 = cleanup-phase-training game.
  //2 = fork from another self-play game.
  //3 = handicap game
  //4 = sampled from an external SGF position (e.g. human data or other bots).
  //5 = sampled from a hint position (e.g. blindspot training).
  //6 = forked from a hint position (e.g. blindspot training).
  //7 = asymmetric playouts game (nonzero "PDA"). Note that this might actually get overwritten by modes 2,4,5,6.

  //C56: Initial turn number - the turn number that corresponds to turn idx 0, such as for sgfposes.
  //C57: Raw winloss from neural net
  //C58: Raw scoremean from neural net
  //C59: Policy prior entropy
  //C60: Number of visits in the search generating this row, prior to any reduction.
  //C61: Number of bonus points the player to move will get onward from this point in the game. Reliable only if C27 and/or C62 (V2 and later), otherwise may make no sense.
  //C62: V1: unused. V2 and later: 1 if the game was finished and not a side position.
  //C63: Data format version, currently always equals 3. Version 2 data had only 64 channels, ending here.

  //C64: 1 if this position originally got only a cheap search during the game and was reanalyzed with a full search
  //after the game ended, generating this row. 0 for ordinary full-search rows and side position rows.
  //C65: If C64, the policy surprise of the original cheap search (the reanalysis search's own policy surprise is in C30).
  //C66: If C64, the value surprise of the original cheap search, by whichever value surprise definition was
  //configured (see PlaySettings::useSearchValueSurprise). Both C65 and C66 are the stats that drove the
  //probability of this position being selected for reanalysis, recorded for statistical purposes.
  //C67: If C64, the number of visits of the original cheap search (the reanalysis search's visits are in C60).
  //C68: Enclosure fork: 1 if valueTargetsNCHW C1 holds the game's target (GameInterface::TARGET_SEKI, Enclosure's
  //cut-edge target). Rows before 2026-10-03 have 0: no target. In rows with C27 0, KataGo's flag of the pass-alive
  //mode (BoardHistoryModes), which C27 cancels.
  //C69: Enclosure fork: 1 if valueTargetsNCHW C5-6 hold the game's target (GameInterface::TARGET_INCOME, Enclosure's
  //territory income). 0 in rows written before it, which the shuffle pads to 7 channels, and in rows with no turn end
  //left.
  //C70: Enclosure fork (owner, 2026-10-04): RGSC's regret target (Eq. 2), if C71.
  //C71: Enclosure fork: 1 if C70 holds the regret target (0 in side positions and when RGSC is off).
  //C72: Enclosure fork (2026-10-11): RGSC's swing regret target (value swing, scaled to the regret's range and capped),
  //if C73.
  //C73: Enclosure fork: 1 if C72 holds the swing regret target (0 in side positions, when RGSC is off, and in rows
  //written before 2026-10-11).
  //C74: Enclosure fork (2026-10-11): the RGSC buffer that the row's game restarted from: 0 none (a normal start, a
  //fork, or RGSC off), 1 the vanilla buffer, 2 the swing buffer. Side positions carry their game's.
  //C75: Enclosure fork (2026-10-14): 1 if valueTargetsNCHW C7 holds the next mover's nodes (0 in rows with no next
  //move, and in rows written before it, which the shuffle pads to 8 channels).
  //C76-79: KataGame fork: the game's own global targets (GameInterface::TargetValues), 0 where the game writes none.
  //Blueprints: C76 who makes the next move (+1 the mover, -1 the opponent, 0 none), C77 the version of its targets.
  //C80-83: KataGame fork (2026-10-29), in the rows of the inputs version 14 only (numGlobalTargetChannels): AZdb's
  //targets of the intrinsic value head (program/azdb.h, rowTargets): output A's target, its weight, output B's target,
  //its weight. The rows of earlier versions end at C79, as before.

  NumpyBuffer<float> globalTargetsNC;

  //Score target
  //Indices correspond to scores, from (-dataXLen*dataYLen-EXTRA_SCORE_DISTR_RADIUS)-0.5 to (dataXLen*dataYLen+EXTRA_SCORE_DISTR_RADIUS)+0.5,
  //making 2*dataXLen*dataYLen+2*EXTRA_SCORE_DISTR_RADIUS indices in total.
  //Index of the actual score is labeled with 100, the rest labeled with 0, from the perspective of the player to move.
  //Except in case of integer komi, the value can be split between two adjacent labels based on value of draw.
  //Arbitrary if C27 has weight 0.
  NumpyBuffer<int8_t> scoreDistrN;

  //Spatial value-related targets
  //KataGame fork: C0, C1, C4 and C5-6 hold the game's own targets (GameEncoding::TARGETS and training_targets, stored
  //as GameInterface::TargetValues gives), all 0 in a slot that the game does not write. C2-3 and C7 are the front's,
  //from the game's pieces, unless the game computes them itself (GameInterface::ownTargets: Blueprints' C2-3).
  //Enclosure's meanings follow each slot (ENCLOSURE.md, "Spatial targets").
  //C0: Final board ownership [-1,1], from the perspective of the player to move. All 0 if C27 has weight 0.
  //    Enclosure fork: the future occupancy (the mean over the turn ends after the row of +1 inside the mover's
  //    regions, -1 inside the opponent's), x 120 ([-120,120]).
  //C1: Difference between ownership and naive area (such as due to seki). All 0 if C27 has weight 0.
  //    Enclosure fork: edges that get cut (+1 the mover's, -1 the opponent's, 0 not cut, 2 no edge), if C68.
  //C2-3: Future board position a certain number of turns in the future. All 0 if C33 has weight 0.
  //C4: Final board area/territory [-120,120]. All 0 if C34 has weight 0. Unlike ownership, takes into account group tax and scoring rules.
  //    Enclosure fork: the region occupancy 4 rounds later (at the 8th turn end after the row), x 120.
  //C5-6: Enclosure fork (owner, 2026-10-03; iviazovetskyi's ownership target): territory income of the player to move
  //(C5) and of the opponent (C6). Per lattice point, its enclosed area inside the unit square around the point,
  //averaged over the turn ends after this row's position, x 120 ([0,120]). All 0 unless C69.
  //C7: Enclosure fork (2026-10-14): not a target. 1 on each node of the next mover (the player of policy channel 1)
  //in the position after this row's move, 0 elsewhere. The trainer's symmetries list policy channel 1's moves
  //between two of that player's nodes from their lower end, as C++ does (games/enclosure/docs/decisions/build.md, "The
  //training's symmetry handling"). All 0 unless C75.
  NumpyBuffer<int8_t> valueTargetsNCHW;

  //Spatial q-value targets, from the perspective of the player to move.
  //C0: winloss * 32000
  //C1: score * 60
  //C2: visits
  NumpyBuffer<int16_t> qValueTargetsNCMove;

  NumpyBuffer<float> metadataInputNC;

  TrainingWriteBuffers(
    int inputsVersion,
    int maxRows,
    int numBinaryChannels,
    int numGlobalChannels,
    int dataXLen,
    int dataYLen,
    bool hasMetadataInput
  );
  ~TrainingWriteBuffers();

  TrainingWriteBuffers(const TrainingWriteBuffers&) = delete;
  TrainingWriteBuffers& operator=(const TrainingWriteBuffers&) = delete;

  void clear();

  //Enclosure fork (owner, 2026-10-04): the regret target of the row just added (C70, flag C71).
  void setLastRowRegret(float regret);
  //Enclosure fork (2026-10-11): the swing regret target of the row just added (C72, flag C73), and its game's restart
  //source (C74).
  void setLastRowSwing(float swing);
  void setLastRowRgscSource(int source);
  //KataGame fork (2026-10-29): AZdb's targets of the intrinsic value head of the row just added (C80-83), in the rows
  //of the inputs version 14.
  void setLastRowIntrinsic(float aTarget, float aWeight, float bTarget, float bWeight);

  void addRow(
    const Board& board, const BoardHistory& hist, Player nextPlayer,
    const BoardHistory& startHist,
    const BoardHistory& actualGameEndHist,
    int turnAfterStart,
    float targetWeight,
    int64_t unreducedNumVisits,
    const std::vector<PolicyTargetMove>* policyTarget0, //can be null
    const std::vector<PolicyTargetMove>* policyTarget1, //can be null
    double policySurprise,
    double policyEntropy,
    double searchEntropy,
    const std::vector<ValueTargets>& whiteValueTargets,
    const std::vector<QValueTargets>& whiteQValueTargets,
    int whiteValueTargetsIdx, //index in whiteValueTargets corresponding to this turn.
    float valueTargetWeight,
    float tdValueTargetWeight,
    float leadTargetWeightFactor,
    const NNRawStats& nnRawStats,
    const Board* finalBoard,
    Color* finalFullArea,
    Color* finalOwnership,
    float* finalWhiteScoring,
    const std::vector<Board>* posHistForFutureBoards, //can be null
    bool isSidePosition,
    int numNeuralNetsBehindLatest,
    double drawEquivalentWinsForWhite,
    Player playoutDoublingAdvantagePla,
    double playoutDoublingAdvantage,
    Hash128 gameHash,
    const std::vector<ChangedNeuralNet*>& changedNeuralNets,
    bool hitTurnLimit,
    int numExtraBlack,
    int mode,
    SGFMetadata* sgfMeta,
    Rand& rand,
    const ReanalysisData& reanalysisData = ReanalysisData(),
    int diversityMember = 0,  //KataGame fork: the member of the row's player, its latent from the inputs version 14
    //KataGame fork (abscls-il4e.11): the position after the row's move in the view of policyTarget1 when it is not
    //posHistForFutureBoards' (the root's views, TrainingDataWriter::writeGame), else NULL. board, the row's position,
    //can hold a view too: the game's targets read both in their views (GameInterface::trainingTargets).
    const Board* nextFrame = NULL
  );

  void writeToZipFile(const std::string& fileName);
  void writeToTextOstream(std::ostream& out);

};

class TrainingDataWriter {
 public:
  TrainingDataWriter(const std::string& outputDir, int inputsVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, const std::string& randSeed);
  TrainingDataWriter(std::ostream* debugOut, int inputsVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, int onlyWriteEvery, const std::string& randSeed);
  TrainingDataWriter(const std::string& outputDir, std::ostream* debugOut, int inputsVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, int onlyWriteEvery, const std::string& randSeed);
  ~TrainingDataWriter();

  void writeGame(const FinishedGameData& data);
  void flushIfNonempty();
  bool flushIfNonempty(std::string& resultingFilename);

  bool isEmpty() const;
  int64_t numRowsInBuffer() const;
  // Cumulative training rows added by this writer, including rows still in the buffer not yet flushed to file.
  // Safe to call concurrently from other threads (e.g. for stats logging).
  int64_t numRowsWritten() const;
  // KataGame fork (abscls-il4e.11): of those rows, the game's rows whose second policy target went from the next
  // search's view into the view that the row gives it (writeGame, FinishedGameData::moveViews), and the rows where no
  // move of it stayed, which then have no second policy target. Safe to call as numRowsWritten.
  int64_t numRowsNextPolicyTranslated() const;
  int64_t numRowsNextPolicyLost() const;

 private:
  std::string outputDir;
  int inputsVersion;
  Rand rand;
  TrainingWriteBuffers* writeBuffers;

  std::ostream* debugOut;
  int debugOnlyWriteEvery;
  // Atomic only so that numRowsWritten() can be read for stats from other threads;
  // all mutation happens on the single thread using this writer.
  std::atomic<int64_t> rowCount;
  std::atomic<int64_t> nextPolicyTranslatedCount;
  std::atomic<int64_t> nextPolicyLostCount;

  bool isFirstFile;
  int firstFileMaxRows;

  void writeAndClearIfFull();

};


#endif  // DATAIO_TRAININGWRITE_H_
