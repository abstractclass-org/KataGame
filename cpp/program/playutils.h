#ifndef PROGRAM_PLAY_UTILS_H_
#define PROGRAM_PLAY_UTILS_H_

#include <functional>

#include "../core/config_parser.h"
#include "../program/play.h"
#include "../search/asyncbot.h"

//This is a grab-bag of various useful higher-level functions that select moves or evaluate the board in various ways.

namespace PlayUtils {
  //Use the given bot to play free handicap stones, modifying the board and hist in the process and setting the bot's position to it.
  //Does NOT switch the initial player of the board history to white
  void playExtraBlack(
    Search* bot,
    int numExtraBlack,
    Board& board,
    BoardHistory& hist,
    double temperature,
    Rand& gameRand
  );

  //Set board to empty and place fixed handicap stones, raising an exception if invalid
  void placeFixedHandicap(Board& board, int n);

  ExtraBlackAndKomi chooseExtraBlackAndKomi(
    float base, float stdev, double allowIntegerProb,
    double handicapProb, int numExtraBlackFixed,
    double bigStdevProb, float bigStdev,
    double biggerStdevProb, float biggerStdev,
    const Board& board, Rand& rand  //KataGame fork: the board, for its komi area (Board::komiArea)
  );
  void setKomiWithoutNoise(const ExtraBlackAndKomi& extraBlackAndKomi, BoardHistory& hist); //Also ignores allowInteger
  void setKomiWithNoise(const ExtraBlackAndKomi& extraBlackAndKomi, BoardHistory& hist, Rand& rand);

  ReportedSearchValues getWhiteScoreValues(
    Search* bot,
    const Board& board,
    const BoardHistory& hist,
    Player pla,
    int64_t numVisits,
    const OtherGameProperties& otherGameProps
  );

  Loc chooseRandomLegalMove(const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc banMove);
  int chooseRandomLegalMoves(const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc* buf, int len);

  Loc chooseRandomPolicyMove(
    const NNOutput* nnOutput,
    const Board& board,
    const BoardHistory& hist,
    Player pla,
    Rand& gameRand,
    double temperature,
    bool allowPass,
    Loc banMove
  );

  Loc getGameInitializationMove(
    Search* botB, Search* botW, const Board& board, const BoardHistory& hist, Player pla, NNResultBuf& buf,
    Rand& gameRand, double temperature
  );
  void initializeGameUsingPolicy(
    Search* botB, Search* botW, Board& board, BoardHistory& hist, Player& pla,
    Rand& gameRand, bool doEndGameIfAllPassAlive,
    double proportionOfBoardArea, double policyInitGammaShape, double temperature
  );

  float roundAndClipKomi(double unrounded, const Board& board);

  //KataGame fork: the last stage of KataGo's even-komi search (adjustKomiToEven, computeLead): the offset delta from
  //the search's guess where White's win-loss value winLossAt(offset) is 0. It brackets the sign change from the guess
  //(offset 0) on the side that the value at 0 points to, with the offsets step, 2 * step and so on up to limit, and
  //bisects the bracket on the half-point grid to a window of half a point. Then it interpolates in that window.
  //Without a sign change up to limit, it returns false at once, with delta the far end, unless endIsEven: KataGo's
  //rule, under which the bisection ends at the far end, and that end counts as the even komi. Else it returns true.
  //KataGo's bracket is step 1 and limit 32, with its rule (the game's: GameInterface::komiBracketStep,
  //komiBracketLimit and komiBracketEndIsEven).
  bool evenKomiOffset(
    const std::function<double(double)>& winLossAt, double step, double limit, bool endIsEven, double& delta
  );

  //KataGame fork: returns false when the search finds no even komi (evenKomiOffset), and then leaves the komi as it was.
  bool adjustKomiToEven(
    Search* botB,
    Search* botW, //can be NULL if only one bot
    const Board& board,
    BoardHistory& hist,
    Player pla,
    int64_t numVisits,
    const OtherGameProperties& otherGameProps,
    Rand& rand
  );

  //Lead from WHITE's perspective
  //KataGame fork: found, when given, tells whether the search found an even komi (evenKomiOffset). Without one, the
  //lead is the bracket's far end, not a measurement.
  float computeLead(
    Search* botB,
    Search* botW, //can be NULL if only one bot
    const Board& board,
    BoardHistory& hist,
    Player pla,
    int64_t numVisits,
    const OtherGameProperties& otherGameProps,
    bool* found = NULL
  );

  double getSearchFactor(
    double searchFactorWhenWinningThreshold,
    double searchFactorWhenWinning,
    const SearchParams& params,
    const std::vector<double>& recentWinLossValues,
    Player pla
  );

  double getHackedLCBForWinrate(const Search* search, const AnalysisData& data, Player pla);

  std::vector<double> computeOwnership(
    Search* bot,
    const Board& board,
    const BoardHistory& hist,
    Player pla,
    int64_t numVisits
  );

  //Determine all living and dead stones, if the game were terminated right now and
  //the rules were interpreted naively and directly.
  //Returns a vector indexed by board Loc (length Board::MAX_ARR_SIZE).
  std::vector<bool> computeAnticipatedStatusesSimple(
    const Board& board,
    const BoardHistory& hist
  );

  //Determine all living and dead stones, trying to be clever and use the ownership prediction
  //of the neural net.
  //Returns a vector indexed by board Loc (length Board::MAX_ARR_SIZE).
  std::vector<bool> computeAnticipatedStatusesWithOwnership(
    Search* bot,
    const Board& board,
    const BoardHistory& hist,
    Player pla,
    int64_t numVisits,
    std::vector<double>& ownershipsBuf
  );


  struct BenchmarkResults {
    int numThreads = 0;
    int totalPositionsSearched = 0;
    int totalPositions = 0;
    int64_t totalVisits = 0;
    double totalSeconds = 0;
    int64_t numNNEvals = 0;
    int64_t numNNBatches = 0;
    double avgBatchSize = 0;

    std::string toStringNotDone() const;
    std::string toString() const;
    std::string toStringWithElo(const BenchmarkResults* baseline, double secondsPerGameMove) const;

    double computeEloEffect(double secondsPerGameMove) const;

    static void printEloComparison(const std::vector<BenchmarkResults>& results, double secondsPerGameMove);
  };

  //Run benchmark on sgf positions. ALSO prints to stdout the ongoing result as it benchmarks.
  BenchmarkResults benchmarkSearchOnPositionsAndPrint(
    const SearchParams& params,
    const CompactSgf& sgf,
    int numPositionsToUse,
    NNEvaluator* nnEval,
    const BenchmarkResults* baseline,
    double secondsPerGameMove,
    bool printElo
  );

  void printGenmoveLog(
    std::ostream& out,
    const Search* search,
    const NNEvaluator* nnEval,
    Loc moveLoc,
    double timeTaken,
    Player perspective,
    bool logSearchInfoForChosenMove,
    const PrintTreeOptions& treeOptions
  );

  Rules genRandomRules(Rand& rand);

  Loc maybeCleanupBeforePass(
    enabled_t cleanupBeforePass,
    enabled_t friendlyPass,
    const Player pla,
    Loc moveLoc,
    const AsyncBot* bot
  );

  Loc maybeFriendlyPass(
    enabled_t cleanupBeforePass,
    enabled_t friendlyPass,
    const Player pla,
    Loc moveLoc,
    Search* bot,
    int64_t numVisits
  );

  std::shared_ptr<NNOutput> getFullSymmetryNNOutput(
    const Board& board, const BoardHistory& hist, Player pla, bool includeOwnerMap, const SGFMetadata* sgfMeta, NNEvaluator* nnEval
  );

}


#endif //PROGRAM_PLAY_UTILS_H_
