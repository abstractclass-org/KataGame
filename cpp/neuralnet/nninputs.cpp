#include "../neuralnet/nninputs.h"

#include "../core/test.h"
#include "../core/bitscan.h"

using namespace std;

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

const Hash128 MiscNNInputParams::ZOBRIST_CONSERVATIVE_PASS =
  Hash128(0x0c2b96f4b8ae2da9ULL, 0x5a14dee208fec0edULL);
const Hash128 MiscNNInputParams::ZOBRIST_FRIENDLY_PASS =
  Hash128(0xe750505a66f7c5c2ULL, 0x7a83139bf632d6c4ULL);
const Hash128 MiscNNInputParams::ZOBRIST_PASSING_HACKS =
  Hash128(0x9c89f4fd3ce5a92cULL, 0x268c9aff79c64d00ULL);
const Hash128 MiscNNInputParams::ZOBRIST_PLAYOUT_DOUBLINGS =
  Hash128(0xa5e6114d380bfc1dULL, 0x4160557f1222f4adULL);
const Hash128 MiscNNInputParams::ZOBRIST_NN_POLICY_TEMP =
  Hash128(0xebcbdfeec6f4334bULL, 0xb85e43ee243b5ad2ULL);
const Hash128 MiscNNInputParams::ZOBRIST_AVOID_MYTDAGGER_HACK =
  Hash128(0x612d22ec402ce054ULL, 0x0db915c49de527aeULL);
const Hash128 MiscNNInputParams::ZOBRIST_POLICY_OPTIMISM =
  Hash128(0x88415c85c2801955ULL, 0x39bdf76b2aaa5eb1ULL);
const Hash128 MiscNNInputParams::ZOBRIST_ZERO_HISTORY =
  Hash128(0x78f02afdd1aa4910ULL, 0xda78d550486fe978ULL);

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

double ScoreValue::whiteWinsOfWinner(Player winner, double drawEquivalentWinsForWhite) {
  if(winner == P_WHITE)
    return 1.0;
  else if(winner == P_BLACK)
    return 0.0;

  testAssert(winner == C_EMPTY);
  return drawEquivalentWinsForWhite;
}

static const double twoOverPi = 0.63661977236758134308;
static const double piOverTwo = 1.57079632679489661923;

double ScoreValue::whiteScoreDrawAdjust(double finalWhiteMinusBlackScore, double drawEquivalentWinsForWhite, const BoardHistory& hist) {
  return finalWhiteMinusBlackScore + hist.whiteKomiAdjustmentForDraws(drawEquivalentWinsForWhite);
}

double ScoreValue::whiteScoreValueOfScoreSmooth(
  double finalWhiteMinusBlackScore,
  double center,
  double scale,
  double drawEquivalentWinsForWhite,
  double sqrtBoardArea,
  const BoardHistory& hist)
{
  double adjustedScore = finalWhiteMinusBlackScore + hist.whiteKomiAdjustmentForDraws(drawEquivalentWinsForWhite) - center;
  return atan(adjustedScore / (scale * sqrtBoardArea)) * twoOverPi;
}

double ScoreValue::whiteScoreValueOfScoreSmoothNoDrawAdjust(double finalWhiteMinusBlackScore, double center, double scale, double sqrtBoardArea) {
  double adjustedScore = finalWhiteMinusBlackScore - center;
  return atan(adjustedScore / (scale * sqrtBoardArea)) * twoOverPi;
}

double ScoreValue::whiteDScoreValueDScoreSmoothNoDrawAdjust(double finalWhiteMinusBlackScore, double center, double scale, double sqrtBoardArea) {
  double adjustedScore = finalWhiteMinusBlackScore - center;
  double scaleFactor;
  scaleFactor = scale * sqrtBoardArea;
  return scaleFactor / (scaleFactor * scaleFactor + adjustedScore * adjustedScore) * twoOverPi;
}

static double inverse_atan(double x) {
  if(x >= piOverTwo - 1e-6) return 1e6;
  if(x <= -piOverTwo + 1e-6) return -1e6;
  return tan(x);
}

double ScoreValue::approxWhiteScoreOfScoreValueSmooth(double scoreValue, double center, double scale, double sqrtBoardArea) {
  testAssert(scoreValue >= -1 && scoreValue <= 1);
  double scoreUnscaled = inverse_atan(scoreValue * piOverTwo);
  return scoreUnscaled * (scale * sqrtBoardArea) + center;
}

double ScoreValue::whiteScoreMeanSqOfScoreGridded(double finalWhiteMinusBlackScore, double drawEquivalentWinsForWhite) {
  //Enclosure fork: scores are exact areas, not on Go's half-point grid, so a final score is a point mass
  //(no draw smearing over +-0.5; the score bins are Enclosure's, departure D1).
  (void)drawEquivalentWinsForWhite;
  return finalWhiteMinusBlackScore * finalWhiteMinusBlackScore;
  testAssert((int)(finalWhiteMinusBlackScore * 2) == finalWhiteMinusBlackScore * 2);
  bool finalScoreIsInteger = ((int)finalWhiteMinusBlackScore == finalWhiteMinusBlackScore);
  if(!finalScoreIsInteger)
    return finalWhiteMinusBlackScore * finalWhiteMinusBlackScore;

  double lower = finalWhiteMinusBlackScore - 0.5;
  double upper = finalWhiteMinusBlackScore + 0.5;
  double lowerSq = lower * lower;
  double upperSq = upper * upper;

  return lowerSq + (upperSq - lowerSq) * drawEquivalentWinsForWhite;
}


static bool scoreValueTablesInitialized = false;
static double* expectedSVTable = NULL;
static const int svTableAssumedBSize = NNPos::MAX_BOARD_LEN;
static const int svTableMeanRadius = svTableAssumedBSize*svTableAssumedBSize + NNPos::EXTRA_SCORE_DISTR_RADIUS;
static const int svTableMeanLen = svTableMeanRadius*2;
static const int svTableStdevLen = svTableAssumedBSize*svTableAssumedBSize + NNPos::EXTRA_SCORE_DISTR_RADIUS;

void ScoreValue::freeTables() {
  if(scoreValueTablesInitialized) {
    delete[] expectedSVTable;
    expectedSVTable = NULL;
    scoreValueTablesInitialized = false;
  }
}

void ScoreValue::initTables() {
  testAssert(!scoreValueTablesInitialized);
  expectedSVTable = new double[svTableMeanLen*svTableStdevLen];

  //Precompute normal PDF
  const int stepsPerUnit = 10; //Must be divisible by 2. This is both the number of segments that we divide points into, and that we divide stdevs into
  const int boundStdevs = 5;
  int minStdevSteps = -boundStdevs*stepsPerUnit;
  int maxStdevSteps = boundStdevs*stepsPerUnit;
  double* normalPDF = new double[(maxStdevSteps-minStdevSteps)+1];
  for(int i = minStdevSteps; i <= maxStdevSteps; i++) {
    double xInStdevs = (double)i / stepsPerUnit;
    double w = exp(-0.5 * xInStdevs * xInStdevs);
    normalPDF[i-minStdevSteps] = w;
  }
  //Precompute scorevalue at increments of 1/stepsPerUnit points
  int minSVSteps = - (svTableMeanRadius*stepsPerUnit + stepsPerUnit/2 + boundStdevs * svTableStdevLen * stepsPerUnit);
  int maxSVSteps = -minSVSteps;
  double* svPrecomp = new double[(maxSVSteps-minSVSteps)+1];
  for(int i = minSVSteps; i <= maxSVSteps; i++) {
    double mean = (double)i / stepsPerUnit;
    double sv = whiteScoreValueOfScoreSmoothNoDrawAdjust(mean, 0.0, 1.0, svTableAssumedBSize);
    svPrecomp[i-minSVSteps] = sv;
  }

  //Perform numeric integration
  for(int meanIdx = 0; meanIdx < svTableMeanLen; meanIdx++) {
    int meanSteps = (meanIdx - svTableMeanRadius) * stepsPerUnit - stepsPerUnit/2;
    for(int stdevIdx = 0; stdevIdx < svTableStdevLen; stdevIdx++) {
      double wSum = 0.0;
      double wsvSum = 0.0;
      for(int i = minStdevSteps; i <= maxStdevSteps; i++) {
        int xSteps = meanSteps + stdevIdx * i;
        double w = normalPDF[i-minStdevSteps];
        assert(xSteps >= minSVSteps && xSteps <= maxSVSteps);
        double sv = svPrecomp[xSteps-minSVSteps];
        wSum += w;
        wsvSum += w*sv;
      }
      expectedSVTable[meanIdx*svTableStdevLen + stdevIdx] = wsvSum / wSum;
    }
  }

  delete[] normalPDF;
  delete[] svPrecomp;
  scoreValueTablesInitialized = true;
}

double ScoreValue::expectedWhiteScoreValue(double whiteScoreMean, double whiteScoreStdev, double center, double scale, double sqrtBoardArea) {
  testAssert(scoreValueTablesInitialized);

  double scaleFactor = (double)svTableAssumedBSize / (scale * sqrtBoardArea);

  double meanScaled = (whiteScoreMean - center) * scaleFactor;
  double stdevScaled = whiteScoreStdev * scaleFactor;

  double meanRounded = round(meanScaled);
  double stdevFloored = floor(stdevScaled);
  int meanIdx0 = (int)meanRounded + svTableMeanRadius;
  int stdevIdx0 = (int)stdevFloored;
  int meanIdx1 = meanIdx0+1;
  int stdevIdx1 = stdevIdx0+1;

  if(meanIdx0 < 0) { meanIdx0 = 0; meanIdx1 = 0; }
  if(meanIdx1 >= svTableMeanLen) { meanIdx0 = svTableMeanLen-1; meanIdx1 = svTableMeanLen-1; }
  testAssert(stdevIdx0 >= 0);
  if(stdevIdx1 >= svTableStdevLen) { stdevIdx0 = svTableStdevLen-1; stdevIdx1 = svTableStdevLen-1; }

  double lambdaMean = meanScaled - meanRounded + 0.5;
  double lambdaStdev = stdevScaled - stdevFloored;

  double a00 = expectedSVTable[meanIdx0*svTableStdevLen + stdevIdx0];
  double a01 = expectedSVTable[meanIdx0*svTableStdevLen + stdevIdx1];
  double a10 = expectedSVTable[meanIdx1*svTableStdevLen + stdevIdx0];
  double a11 = expectedSVTable[meanIdx1*svTableStdevLen + stdevIdx1];

  double b0 = a00 + lambdaStdev*(a01-a00);
  double b1 = a10 + lambdaStdev*(a11-a10);
  return b0 + lambdaMean*(b1-b0);
}

double ScoreValue::getScoreStdev(double scoreMean, double scoreMeanSq) {
  double variance = scoreMeanSq - scoreMean * scoreMean;
  if(variance <= 0.0)
    return 0.0;
  return sqrt(variance);
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

void NNInputs::fillScoring(
  const Board& board,
  const Color* area,
  bool groupTax,
  float* scoring
) {
  if(!groupTax) {
    std::fill(scoring, scoring + Board::MAX_ARR_SIZE, 0.0f);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        Color areaColor = area[loc];
        if(areaColor == P_BLACK)
          scoring[loc] = -1.0f;
        else if(areaColor == P_WHITE)
          scoring[loc] = 1.0f;
        else {
          assert(areaColor == C_EMPTY);
          scoring[loc] = 0;
        }
      }
    }
  }
  else {
    bool visited[Board::MAX_ARR_SIZE];
    Loc queue[Board::MAX_ARR_SIZE];

    std::fill(visited, visited + Board::MAX_ARR_SIZE, false);
    std::fill(scoring, scoring + Board::MAX_ARR_SIZE, 0.0f);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(visited[loc])
          continue;
        Color areaColor = area[loc];
        if(areaColor == P_BLACK || areaColor == P_WHITE) {
          float fullValue = areaColor == P_WHITE ? 1.0f : -1.0f;
          int queueHead = 0;
          int queueTail = 1;
          queue[0] = loc;
          visited[loc] = true;

          //First, count how many empty or opp locations there are
          int territoryCount = 0;
          while(queueHead < queueTail) {
            Loc next = queue[queueHead];
            queueHead++;
            if(board.colors[next] != areaColor)
              territoryCount++;
            //Push adjacent locations on to queue
            for(int i = 0; i<4; i++) {
              Loc adj = next + board.adj_offsets[i];
              if(area[adj] == areaColor && !visited[adj]) {
                queue[queueTail] = adj;
                queueTail++;
                visited[adj] = true;
              }
            }
          }

          //Then, actually fill values
          float territoryValue = territoryCount <= 2 ? 0.0f : fullValue * (territoryCount - 2.0f) / territoryCount;
          for(int j = 0; j<queueTail; j++) {
            Loc next = queue[j];
            queueHead++;
            if(board.colors[next] != areaColor)
              scoring[next] = territoryValue;
            else
              scoring[next] = fullValue;
          }
        }
        else {
          assert(areaColor == C_EMPTY);
          scoring[loc] = 0;
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------


NNOutput::NNOutput()
  :policyPassSlot(-1),policyDense(NULL),whiteOwnerMap(NULL),noisedPolicyProbs(NULL),hasRegret(false),regretValue(0.0f),
   regretRanking(0.0f),hasSwing(false),swingValue(0.0f),swingRanking(0.0f),auxMap(NULL),auxValues(NULL),auxPolicy(NULL),
   scoreBeliefLen(0),scoreBelief(NULL),scoreBeliefRaw(NULL)
{
  std::fill(policyMask, policyMask + NNPos::MAX_BOARD_AREA, uint64_t(0));
}

//Enclosure fork (owner, 2026-10-05/06: compact policy storage).
void PolicyView::toDense(float* out) const {
  if(dense != NULL) {
    std::copy(dense, dense + NNPos::MAX_NN_POLICY_SIZE, out);
    return;
  }
  std::fill(out, out + NNPos::MAX_NN_POLICY_SIZE, -1.0f);
  for(int k = 0; k<n; k++)
    out[pos[k]] = prob[k];
}

void NNOutput::setPolicyFromDense(const float* dense) {
  legalPos.clear();
  legalProbs.clear();
  std::fill(policyMask, policyMask + NNPos::MAX_BOARD_AREA, uint64_t(0));
  policyPassSlot = -1;
  const int passPos = NNPos::PASS_POS;
  for(int point = 0; point < NNPos::MAX_BOARD_AREA; point++) {
    policyStart[point] = (uint16_t)legalPos.size();
    for(int channel = 0; channel < NNPos::POLICY_CHANNELS; channel++) {
      const int p = point * NNPos::POLICY_CHANNELS + channel;
      if(dense[p] >= 0) {
        policyMask[point] |= uint64_t(1) << channel;
        legalPos.push_back((uint16_t)p);
        legalProbs.push_back(dense[p]);
      }
    }
  }
  if(dense[passPos] >= 0) {
    policyPassSlot = (int)legalPos.size();
    legalPos.push_back((uint16_t)passPos);
    legalProbs.push_back(dense[passPos]);
  }
  legalPos.shrink_to_fit();
  legalProbs.shrink_to_fit();
}

void NNOutput::setPolicyFromLegal(const uint16_t* pos, const float* probs, const char* legal, int n) {
  const int passPos = NNPos::PASS_POS;
  int count = 0;
  for(int j = 0; j<n; j++)
    count += legal[j] ? 1 : 0;
  legalPos.clear();
  legalProbs.clear();
  legalPos.reserve(count);
  legalProbs.reserve(count);
  std::fill(policyMask, policyMask + NNPos::MAX_BOARD_AREA, uint64_t(0));
  policyPassSlot = -1;
  //As setPolicyFromDense: a point's start is the number of legal entries before it, pass (last) after every point.
  int point = 0;
  for(int j = 0; j<n; j++) {
    if(!legal[j])
      continue;
    const int p = pos[j];
    if(p == passPos) {
      while(point < NNPos::MAX_BOARD_AREA)
        policyStart[point++] = (uint16_t)legalPos.size();
      policyPassSlot = (int)legalPos.size();
    }
    else {
      const int pt = p / NNPos::POLICY_CHANNELS;
      while(point <= pt)
        policyStart[point++] = (uint16_t)legalPos.size();
      policyMask[pt] |= uint64_t(1) << (p - pt * NNPos::POLICY_CHANNELS);
    }
    legalPos.push_back((uint16_t)p);
    legalProbs.push_back(probs[j]);
  }
  while(point < NNPos::MAX_BOARD_AREA)
    policyStart[point++] = (uint16_t)legalPos.size();
}

void NNOutput::compactPolicy() {
  testAssert(policyDense != NULL);
  setPolicyFromDense(policyDense);
  policyDense = NULL;
}

//The compact policy's fields, copied.
static void copyCompactPolicy(NNOutput& to, const NNOutput& from) {
  to.legalPos = from.legalPos;
  to.legalProbs = from.legalProbs;
  std::copy(from.policyMask, from.policyMask + NNPos::MAX_BOARD_AREA, to.policyMask);
  std::copy(from.policyStart, from.policyStart + NNPos::MAX_BOARD_AREA, to.policyStart);
  to.policyPassSlot = from.policyPassSlot;
  to.policyDense = NULL;
}

//Enclosure fork: a copy of an optional array, or NULL.
static float* copyOrNull(const float* src, size_t n) {
  if(src == NULL)
    return NULL;
  float* dst = new float[n];
  std::copy(src, src + n, dst);
  return dst;
}
NNOutput::NNOutput(const NNOutput& other) {
  nnHash = other.nnHash;
  whiteWinProb = other.whiteWinProb;
  whiteLossProb = other.whiteLossProb;
  whiteNoResultProb = other.whiteNoResultProb;
  whiteScoreMean = other.whiteScoreMean;
  whiteScoreMeanSq = other.whiteScoreMeanSq;
  whiteLead = other.whiteLead;
  varTimeLeft = other.varTimeLeft;
  shorttermWinlossError = other.shorttermWinlossError;
  shorttermScoreError = other.shorttermScoreError;

  nnXLen = other.nnXLen;
  nnYLen = other.nnYLen;
  if(other.whiteOwnerMap != NULL) {
    whiteOwnerMap = new float[nnXLen * nnYLen];
    std::copy(other.whiteOwnerMap, other.whiteOwnerMap + nnXLen * nnYLen, whiteOwnerMap);
  }
  else
    whiteOwnerMap = NULL;

  if(other.noisedPolicyProbs != NULL) {
    noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
    std::copy(other.noisedPolicyProbs, other.noisedPolicyProbs + NNPos::MAX_NN_POLICY_SIZE, noisedPolicyProbs);
  }
  else
    noisedPolicyProbs = NULL;

  copyCompactPolicy(*this, other);  //Enclosure fork: the compact policy
  policyOptimismUsed = other.policyOptimismUsed;

  hasRegret = other.hasRegret;
  regretValue = other.regretValue;
  regretRanking = other.regretRanking;
  hasSwing = other.hasSwing;
  swingValue = other.swingValue;
  swingRanking = other.swingRanking;
  auxMap = copyOrNull(other.auxMap, (size_t)NUM_AUX_MAP_CHANNELS * nnXLen * nnYLen);
  auxValues = copyOrNull(other.auxValues, NUM_AUX_VALUES);
  auxPolicy = copyOrNull(other.auxPolicy, (size_t)NUM_AUX_POLICIES * NNPos::MAX_NN_POLICY_SIZE);
  scoreBeliefLen = other.scoreBeliefLen;
  scoreBelief = copyOrNull(other.scoreBelief, scoreBeliefLen);
  scoreBeliefRaw = NULL;  //transient: only between the backend and NNEvaluator
}

NNOutput::NNOutput(const vector<shared_ptr<NNOutput>>& others) {
  testAssert(others.size() < 1000000);
  int len = (int)others.size();
  float floatLen = (float)len;
  testAssert(len > 0);
  for(int i = 1; i<len; i++) {
    testAssert(others[i]->nnHash == others[0]->nnHash);
  }
  nnHash = others[0]->nnHash;

  whiteWinProb = 0.0f;
  whiteLossProb = 0.0f;
  whiteNoResultProb = 0.0f;
  whiteScoreMean = 0.0f;
  whiteScoreMeanSq = 0.0f;
  whiteLead = 0.0f;
  varTimeLeft = 0.0f;
  shorttermWinlossError = 0.0f;
  shorttermScoreError = 0.0f;
  for(int i = 0; i<len; i++) {
    const NNOutput& other = *(others[i]);
    whiteWinProb += other.whiteWinProb;
    whiteLossProb += other.whiteLossProb;
    whiteNoResultProb += other.whiteNoResultProb;
    whiteScoreMean += other.whiteScoreMean;
    whiteScoreMeanSq += other.whiteScoreMeanSq;
    whiteLead += other.whiteLead;
    varTimeLeft += other.varTimeLeft;
    shorttermWinlossError += other.shorttermWinlossError;
    shorttermScoreError += other.shorttermScoreError;
  }
  whiteWinProb /= floatLen;
  whiteLossProb /= floatLen;
  whiteNoResultProb /= floatLen;
  whiteScoreMean /= floatLen;
  whiteScoreMeanSq /= floatLen;
  whiteLead /= floatLen;
  varTimeLeft /= floatLen;
  shorttermWinlossError /= floatLen;
  shorttermScoreError /= floatLen;

  nnXLen = others[0]->nnXLen;
  nnYLen = others[0]->nnYLen;

  {
    float whiteOwnerMapCount = 0.0f;
    whiteOwnerMap = NULL;
    for(int i = 0; i<len; i++) {
      const NNOutput& other = *(others[i]);
      if(other.whiteOwnerMap != NULL) {
        if(whiteOwnerMap == NULL) {
          whiteOwnerMap = new float[nnXLen * nnYLen];
          std::fill(whiteOwnerMap, whiteOwnerMap + nnXLen * nnYLen, 0.0f);
        }
        whiteOwnerMapCount += 1.0f;
        for(int pos = 0; pos<nnXLen*nnYLen; pos++)
          whiteOwnerMap[pos] += other.whiteOwnerMap[pos];
      }
    }
    if(whiteOwnerMap != NULL) {
      testAssert(whiteOwnerMapCount > 0);
      for(int pos = 0; pos<nnXLen*nnYLen; pos++)
        whiteOwnerMap[pos] /= whiteOwnerMapCount;
    }
  }

  noisedPolicyProbs = NULL;

  //Enclosure fork: RGSC's heads and the aux outputs, averaged like the rest (the aux arrays over the outputs that
  //have them).
  hasRegret = others[0]->hasRegret;
  regretValue = 0.0f;
  regretRanking = 0.0f;
  hasSwing = others[0]->hasSwing;
  swingValue = 0.0f;
  swingRanking = 0.0f;
  for(int i = 0; i<len; i++) {
    regretValue += others[i]->regretValue / floatLen;
    regretRanking += others[i]->regretRanking / floatLen;
    swingValue += others[i]->swingValue / floatLen;
    swingRanking += others[i]->swingRanking / floatLen;
  }
  auto averageOf = [&](float* NNOutput::* field, size_t n) {
    float* avg = NULL;
    float count = 0.0f;
    for(int i = 0; i<len; i++) {
      const float* src = (*others[i]).*field;
      if(src == NULL)
        continue;
      if(avg == NULL) {
        avg = new float[n];
        std::fill(avg, avg + n, 0.0f);
      }
      count += 1.0f;
      for(size_t j = 0; j<n; j++)
        avg[j] += src[j];
    }
    if(avg != NULL)
      for(size_t j = 0; j<n; j++)
        avg[j] /= count;
    return avg;
  };
  auxMap = averageOf(&NNOutput::auxMap, (size_t)NUM_AUX_MAP_CHANNELS * nnXLen * nnYLen);
  auxValues = averageOf(&NNOutput::auxValues, NUM_AUX_VALUES);
  auxPolicy = averageOf(&NNOutput::auxPolicy, (size_t)NUM_AUX_POLICIES * NNPos::MAX_NN_POLICY_SIZE);
  scoreBeliefLen = others[0]->scoreBeliefLen;
  scoreBelief = averageOf(&NNOutput::scoreBelief, scoreBeliefLen);  //a mixture of the symmetries' distributions
  scoreBeliefRaw = NULL;

  //For technical correctness in case of impossibly rare hash collisions:
  //Just give up if they don't all match in move legality
  //Enclosure fork (compact policy): the same on dense copies, then kept compact.
  {
    policyDense = NULL;
    bool mismatch = false;
    std::vector<float> policyProbs(NNPos::MAX_NN_POLICY_SIZE, 0.0f);
    std::vector<float> otherProbs(NNPos::MAX_NN_POLICY_SIZE);
    for(int i = 0; i<len; i++) {
      const NNOutput& other = *(others[i]);
      other.policy().toDense(otherProbs.data());
      for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++) {
        if(i > 0 && (policyProbs[pos] < 0) != (otherProbs[pos] < 0))
          mismatch = true;
        policyProbs[pos] += otherProbs[pos];
      }
    }
    //In case of mismatch, just take the first one
    //This should basically never happen, only on true hash collisions
    if(mismatch) {
      const NNOutput& other = *(others[0]);
      other.policy().toDense(policyProbs.data());
    }
    else {
      for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++)
        policyProbs[pos] /= floatLen;
    }
    setPolicyFromDense(policyProbs.data());
  }
  {
    bool allOptimismsMatch = true;
    for(int i = 1; i<len; i++) {
      if(others[i]->policyOptimismUsed != others[0]->policyOptimismUsed) {
        allOptimismsMatch = false;
        break;
      }
    }
    if(allOptimismsMatch) {
      policyOptimismUsed = others[0]->policyOptimismUsed;
    }
    else {
      policyOptimismUsed = 0.0;
      for(int i = 0; i<len; i++) {
        policyOptimismUsed += others[i]->policyOptimismUsed / (float)len;
      }
    }
  }
}

NNOutput& NNOutput::operator=(const NNOutput& other) {
  if(&other == this)
    return *this;
  nnHash = other.nnHash;
  whiteWinProb = other.whiteWinProb;
  whiteLossProb = other.whiteLossProb;
  whiteNoResultProb = other.whiteNoResultProb;
  whiteScoreMean = other.whiteScoreMean;
  whiteScoreMeanSq = other.whiteScoreMeanSq;
  whiteLead = other.whiteLead;
  varTimeLeft = other.varTimeLeft;
  shorttermWinlossError = other.shorttermWinlossError;
  shorttermScoreError = other.shorttermScoreError;

  nnXLen = other.nnXLen;
  nnYLen = other.nnYLen;
  if(whiteOwnerMap != NULL)
    delete[] whiteOwnerMap;
  if(other.whiteOwnerMap != NULL) {
    whiteOwnerMap = new float[nnXLen * nnYLen];
    std::copy(other.whiteOwnerMap, other.whiteOwnerMap + nnXLen * nnYLen, whiteOwnerMap);
  }
  else
    whiteOwnerMap = NULL;
  if(noisedPolicyProbs != NULL)
    delete[] noisedPolicyProbs;
  if(other.noisedPolicyProbs != NULL) {
    noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
    std::copy(other.noisedPolicyProbs, other.noisedPolicyProbs + NNPos::MAX_NN_POLICY_SIZE, noisedPolicyProbs);
  }
  else
    noisedPolicyProbs = NULL;

  copyCompactPolicy(*this, other);  //Enclosure fork: the compact policy
  policyOptimismUsed = other.policyOptimismUsed;

  hasRegret = other.hasRegret;
  regretValue = other.regretValue;
  regretRanking = other.regretRanking;
  hasSwing = other.hasSwing;
  swingValue = other.swingValue;
  swingRanking = other.swingRanking;
  delete[] auxMap;
  auxMap = copyOrNull(other.auxMap, (size_t)NUM_AUX_MAP_CHANNELS * nnXLen * nnYLen);
  delete[] auxValues;
  auxValues = copyOrNull(other.auxValues, NUM_AUX_VALUES);
  delete[] auxPolicy;
  auxPolicy = copyOrNull(other.auxPolicy, (size_t)NUM_AUX_POLICIES * NNPos::MAX_NN_POLICY_SIZE);
  scoreBeliefLen = other.scoreBeliefLen;
  delete[] scoreBelief;
  scoreBelief = copyOrNull(other.scoreBelief, scoreBeliefLen);
  delete[] scoreBeliefRaw;
  scoreBeliefRaw = NULL;

  return *this;
}


NNOutput::~NNOutput() {
  if(whiteOwnerMap != NULL) {
    delete[] whiteOwnerMap;
    whiteOwnerMap = NULL;
  }
  if(noisedPolicyProbs != NULL) {
    delete[] noisedPolicyProbs;
    noisedPolicyProbs = NULL;
  }
  delete[] auxMap;
  auxMap = NULL;
  delete[] auxValues;
  auxValues = NULL;
  delete[] auxPolicy;
  auxPolicy = NULL;
  delete[] scoreBelief;
  scoreBelief = NULL;
  delete[] scoreBeliefRaw;
  scoreBeliefRaw = NULL;
}


void NNOutput::debugPrint(ostream& out, const Board& board) const {
  out << "Win " << Global::strprintf("%.2fc",whiteWinProb*100) << endl;
  out << "Loss " << Global::strprintf("%.2fc",whiteLossProb*100) << endl;
  out << "NoResult " << Global::strprintf("%.2fc",whiteNoResultProb*100) << endl;
  out << "ScoreMean " << Global::strprintf("%.2f",whiteScoreMean) << endl;
  out << "ScoreMeanSq " << Global::strprintf("%.1f",whiteScoreMeanSq) << endl;
  out << "Lead " << Global::strprintf("%.2f",whiteLead) << endl;
  out << "VarTimeLeft " << Global::strprintf("%.1f",varTimeLeft) << endl;
  out << "STWinlossError " << Global::strprintf("%.2fc",shorttermWinlossError*100) << endl;
  out << "STScoreError " << Global::strprintf("%.2f",shorttermScoreError) << endl;
  out << "OptimismUsed " << Global::strprintf("%.2f",policyOptimismUsed) << endl;

  out << "Policy" << endl;
  const PolicyView policyProbs = policy();
  out << "Pass" << Global::strprintf("%4d ", (int)round(policyProbs[NNPos::getPassPos(nnXLen,nnYLen)] * 1000)) << endl;
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      float prob = policyProbs[pos];
      if(prob < 0)
        out << "   - ";
      else
        out << Global::strprintf("%4d ", (int)round(prob * 1000));
    }
    out << endl;
  }

  if(whiteOwnerMap != NULL) {
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,nnXLen);
        float whiteOwn = whiteOwnerMap[pos];
        out << Global::strprintf("%5d ", (int)round(whiteOwn * 1000));
      }
      out << endl;
    }
    out << endl;
  }
}

//-------------------------------------------------------------------------------------------------------------

static void copyWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int cSize, bool useNHWC, int symmetry, bool reverse) {
  //Enclosure fork: the identity is a plain copy (the fork applies symmetries to the position, so the backends
  //always pass 0 here; element by element, this copy of 215 planes per row was about 7% of self-play's CPU).
  if(symmetry == 0) {
    std::copy(src, src + (size_t)nSize * hSize * wSize * cSize, dst);
    return;
  }
  bool transpose = (symmetry & 0x4) != 0 && hSize == wSize;
  bool flipX = (symmetry & 0x2) != 0;
  bool flipY = (symmetry & 0x1) != 0;
  if(transpose && !reverse)
    std::swap(flipX,flipY);
  if(useNHWC) {
    int nStride = hSize * wSize * cSize;
    int hStride = wSize * cSize;
    int wStride = cSize;
    int hBaseNew = 0; int hStrideNew = hStride;
    int wBaseNew = 0; int wStrideNew = wStride;

    if(flipY) { hBaseNew = (hSize-1) * hStrideNew; hStrideNew = -hStrideNew; }
    if(flipX) { wBaseNew = (wSize-1) * wStrideNew; wStrideNew = -wStrideNew; }

    if(transpose)
      std::swap(hStrideNew,wStrideNew);

    for(int n = 0; n<nSize; n++) {
      for(int h = 0; h<hSize; h++) {
        int nhOld = n * nStride + h*hStride;
        int nhNew = n * nStride + hBaseNew + h*hStrideNew;
        for(int w = 0; w<wSize; w++) {
          int nhwOld = nhOld + w*wStride;
          int nhwNew = nhNew + wBaseNew + w*wStrideNew;
          for(int c = 0; c<cSize; c++) {
            dst[nhwNew + c] = src[nhwOld + c];
          }
        }
      }
    }
  }
  else {
    int ncSize = nSize * cSize;
    int ncStride = hSize * wSize;
    int hStride = wSize;
    int wStride = 1;
    int hBaseNew = 0; int hStrideNew = hStride;
    int wBaseNew = 0; int wStrideNew = wStride;

    if(flipY) { hBaseNew = (hSize-1) * hStrideNew; hStrideNew = -hStrideNew; }
    if(flipX) { wBaseNew = (wSize-1) * wStrideNew; wStrideNew = -wStrideNew; }

    if(transpose)
      std::swap(hStrideNew,wStrideNew);

    for(int nc = 0; nc<ncSize; nc++) {
      for(int h = 0; h<hSize; h++) {
        int nchOld = nc * ncStride + h*hStride;
        int nchNew = nc * ncStride + hBaseNew + h*hStrideNew;
        for(int w = 0; w<wSize; w++) {
          int nchwOld = nchOld + w*wStride;
          int nchwNew = nchNew + wBaseNew + w*wStrideNew;
          dst[nchwNew] = src[nchwOld];
        }
      }
    }
  }
}


void SymmetryHelpers::copyInputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int cSize, bool useNHWC, int symmetry) {
  copyWithSymmetry(src, dst, nSize, hSize, wSize, cSize, useNHWC, symmetry, false);
}

void SymmetryHelpers::copyOutputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int symmetry) {
  copyWithSymmetry(src, dst, nSize, hSize, wSize, 1, false, symmetry, true);
}

int SymmetryHelpers::invert(int symmetry) {
  if(symmetry == 5)
    return 6;
  if(symmetry == 6)
    return 5;
  return symmetry;
}

int SymmetryHelpers::compose(int firstSymmetry, int nextSymmetry) {
  if(isTranspose(firstSymmetry))
    nextSymmetry = (nextSymmetry & 0x4) | ((nextSymmetry & 0x2) >> 1) | ((nextSymmetry & 0x1) << 1);
  return firstSymmetry ^ nextSymmetry;
}

int SymmetryHelpers::compose(int firstSymmetry, int nextSymmetry, int nextNextSymmetry) {
  return compose(compose(firstSymmetry,nextSymmetry),nextNextSymmetry);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, int xSize, int ySize, int symmetry) {
  bool transpose = (symmetry & 0x4) != 0;
  bool flipX = (symmetry & 0x2) != 0;
  bool flipY = (symmetry & 0x1) != 0;
  if(flipX) { x = xSize - x - 1; }
  if(flipY) { y = ySize - y - 1; }

  if(transpose)
    std::swap(x,y);
  return Location::getLoc(x,y,transpose ? ySize : xSize);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, const Board& board, int symmetry) {
  return getSymLoc(x,y,board.x_size,board.y_size,symmetry);
}

Loc SymmetryHelpers::getSymLoc(Loc loc, const Board& board, int symmetry) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getSymLoc(Location::getX(loc,board.x_size), Location::getY(loc,board.x_size), board, symmetry);
}

Loc SymmetryHelpers::getSymLoc(Loc loc, int xSize, int ySize, int symmetry) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getSymLoc(Location::getX(loc,xSize), Location::getY(loc,xSize), xSize, ySize, symmetry);
}


//KataGame fork: the image of the game's position (Go's stones, ko and chains do not exist).
Board SymmetryHelpers::getSymBoard(const Board& board, int symmetry) {
  //An allowed symmetry. A game allows a transpose only on square boards (GameEncoding::SYMMETRIES).
  assert(isAllowed(symmetry) && (!isTranspose(symmetry) || board.x_size == board.y_size));
  Board symBoard(board.x_size, board.y_size);
  symBoard.setPosition(GameEncoding::sym_position(board.pos, symmetry));
  //The image's legal moves are the images of the board's (an allowed symmetry keeps the rules), so they are mapped
  //instead of listed again (Enclosure: about 440 moves, against 24 us to list them; most evaluations use a random
  //symmetry).
  const std::bitset<GameRules::NUM_ACTIONS>& bits = board.legalActions();
  std::bitset<GameRules::NUM_ACTIONS> image;
  if(bits.test(GameRules::PASS_ACTION))
    image.set(GameRules::PASS_ACTION);
  const SymActionMap symAction(board, symmetry);
  for(size_t a = bitsFirst(bits); a < (size_t)GameRules::PASS_ACTION; a = bitsNext(bits, a))
    image.set(symAction((int)a));
  symBoard.setLegalActions(image);
  return symBoard;
}

Loc SymmetryHelpers::getSymMoveLoc(Loc moveLoc, const Board& board, int symmetry) {
  if(moveLoc == Board::NULL_LOC || moveLoc == Board::PASS_LOC)
    return moveLoc;
  return Board::moveLocOfAction(SymActionMap(board, symmetry)(Board::actionOfMoveLoc(moveLoc)));
}

//KataGame fork: an allowed symmetry is valid when the position is its own image: the image has the position's hash
//(GameRules::hash covers the whole position). Earlier turns that only the inputs see (GameEncoding::input_hash, such
//as Enclosure's history planes) are not part of the game, as Go's move history is not. isSymDupLoc has
//Board::NUM_MOVE_LOCS entries and avoidMoves is indexed by move Loc. Of each set of equivalent legal moves, the one
//with the lowest action index is kept.
void SymmetryHelpers::markDuplicateMoveLocs(
  const Board& board,
  const BoardHistory& hist,
  const std::vector<int>* onlySymmetries,
  const std::vector<int>& avoidMoves,
  bool* isSymDupLoc,
  std::vector<int>& validSymmetries
) {
  (void)hist;
  std::fill(isSymDupLoc, isSymDupLoc + Board::NUM_MOVE_LOCS, false);
  validSymmetries.clear();
  validSymmetries.reserve(SymmetryHelpers::NUM_SYMMETRIES);
  validSymmetries.push_back(0);

  const Hash128 hash = GameRules::hash(board.pos);
  for(int symmetry = 1; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
    if(!isAllowed(symmetry))
      continue;
    if(onlySymmetries != NULL && !contains(*onlySymmetries,symmetry))
      continue;
    if(GameRules::hash(GameEncoding::sym_position(board.pos, symmetry)) == hash)
      validSymmetries.push_back(symmetry);
  }
  if(validSymmetries.size() <= 1)
    return;

  //The board is its own image, so the image of a move is a move of the same board.
  std::vector<SymActionMap> symActions;
  for(int symmetry: validSymmetries)
    if(symmetry != 0)
      symActions.emplace_back(board, symmetry);
  std::vector<Loc> legal;
  board.getLegalMoveLocs(legal);
  for(Loc loc : legal) {
    if(loc == Board::PASS_LOC)
      continue;
    if(avoidMoves.size() > 0 && avoidMoves[loc] > 0)
      continue;
    for(const SymActionMap& symAction: symActions) {
      Loc symLoc = Board::moveLocOfAction(symAction(Board::actionOfMoveLoc(loc)));
      if(!isSymDupLoc[loc] && loc != symLoc)
        isSymDupLoc[symLoc] = true;
    }
  }
}

std::vector<Loc> SymmetryHelpers::getSymMoveLocs(const Board& board, const std::vector<Loc>& moves, int symmetry) {
  std::vector<Loc> out;
  Board b = board;
  Board image = getSymBoard(board, symmetry);
  for(Loc loc : moves) {
    Player pla = b.nextPla();
    //KataGame fork: the game maps the legal moves only. A move that is not legal on b has no image (the null move).
    const bool legal = !b.isGameOver() && b.isLegal(loc, pla, false);
    Loc symLoc = legal || loc == Board::PASS_LOC ? getSymMoveLoc(loc, b, symmetry) : Board::NULL_LOC;
    out.push_back(symLoc);
    if(!legal || !image.isLegal(symLoc, pla, false))
      continue;
    b.playMoveAssumeLegal(loc, pla);
    image.playMoveAssumeLegal(symLoc, pla);
  }
  return out;
}

static double getSymmetryDifference(const Board& board, const Board& other, int symmetry, double maxDifferenceToReport) {
  double thisDifference = 0.0;
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      Loc symLoc = SymmetryHelpers::getSymLoc(x, y, board, symmetry);
      // Difference!
      if(board.colors[loc] != other.colors[symLoc]) {
        // One of them was empty, the other was a stone
        if(board.colors[loc] == C_EMPTY || other.colors[symLoc] == C_EMPTY)
          thisDifference += 1.0;
        // Differing stones - triple the penalty
        else
          thisDifference += 3.0;

        if(thisDifference > maxDifferenceToReport)
          return maxDifferenceToReport;
      }
    }
  }
  return thisDifference;
}


// For each symmetry, return a metric about the "amount" of difference that board would have with other
// if symmetry were applied to board.
void SymmetryHelpers::getSymmetryDifferences(
  const Board& board, const Board& other, double maxDifferenceToReport, double symmetryDifferences[SymmetryHelpers::NUM_SYMMETRIES]
) {
  for(int symmetry = 0; symmetry<SymmetryHelpers::NUM_SYMMETRIES; symmetry++)
    symmetryDifferences[symmetry] = maxDifferenceToReport;

  // Don't bother handling ultra-fancy transpose logic
  if(board.x_size != other.x_size || board.y_size != other.y_size)
    return;

  int numSymmetries = SymmetryHelpers::NUM_SYMMETRIES;
  if(board.x_size != board.y_size)
    numSymmetries = SymmetryHelpers::NUM_SYMMETRIES_WITHOUT_TRANSPOSE;

  for(int symmetry = 0; symmetry<numSymmetries; symmetry++) {
    symmetryDifferences[symmetry] = getSymmetryDifference(board, other, symmetry, maxDifferenceToReport);
  }
}


//-------------------------------------------------------------------------------------------------------------

static void setRowBin(float* rowBin, int pos, int feature, float value, int posStride, int featureStride) {
  rowBin[pos * posStride + feature * featureStride] = value;
}

//Calls f on each location that is part of an inescapable atari, or a group that can be put into inescapable atari
static void iterLadders(const Board& board, int nnXLen, const std::function<void(Loc,int,const vector<Loc>&)>& f) {
  int xSize = board.x_size;
  int ySize = board.y_size;

  Loc chainHeadsSolved[Board::MAX_PLAY_SIZE];
  bool chainHeadsSolvedValue[Board::MAX_PLAY_SIZE];
  int numChainHeadsSolved = 0;
  Board copy(board);
  vector<Loc> buf;
  vector<Loc> workingMoves;

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      Loc loc = Location::getLoc(x,y,xSize);
      Color stone = board.colors[loc];
      if(stone == P_BLACK || stone == P_WHITE) {
        int libs = board.getNumLiberties(loc);
        if(libs == 1 || libs == 2) {
          bool alreadySolved = false;
          int pos = NNPos::xyToPos(x,y,nnXLen);
          Loc head = board.chain_head[loc];
          for(int i = 0; i<numChainHeadsSolved; i++) {
            if(chainHeadsSolved[i] == head) {
              alreadySolved = true;
              if(chainHeadsSolvedValue[i]) {
                workingMoves.clear();
                f(loc,pos,workingMoves);
              }
              break;
            }
          }
          if(!alreadySolved) {
            //Perform search on copy so as not to mess up tracking of solved heads
            bool laddered;
            if(libs == 1)
              laddered = copy.searchIsLadderCaptured(loc,true,buf);
            else {
              workingMoves.clear();
              laddered = copy.searchIsLadderCapturedAttackerFirst2Libs(loc,buf,workingMoves);
            }

            chainHeadsSolved[numChainHeadsSolved] = head;
            chainHeadsSolvedValue[numChainHeadsSolved] = laddered;
            numChainHeadsSolved++;
            if(laddered)
              f(loc,pos,workingMoves);
          }
        }
      }
    }
  }
}

//Currently does NOT depend on history (except for marking ko-illegal spots)
Hash128 NNInputs::getHash(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams
) {
  //Hash using the effective BoardHistoryModes for this eval, which are normally hist's own
  //but may be overridden per-query (e.g. for a secondary net whose declared featurization differs).
  Hash128 hash = BoardHistory::getSituationRulesAndKoHash(
    board, hist, nextPlayer, nnInputParams.drawEquivalentWinsForWhite,
    nnInputParams.getModes(hist)
  );

  //Fold in whether a pass ends this phase.
  if(hist.passWouldEndPhase(board,nextPlayer)) {
    hash ^= Board::ZOBRIST_PASS_ENDS_PHASE;
    //Technically some of the below only apply when passing ends the game, but it's pretty harmless to use the more
    //conservative hashing including them when the phase would end too.

    //And in the case that a pass ends the phase, conservativePass also affects the result for the root node
    if(nnInputParams.conservativePassAndIsRoot)
      hash ^= MiscNNInputParams::ZOBRIST_CONSERVATIVE_PASS;

    //If we're in a ruleset where passing without capturing all the stones is okay, and as a result are suppressing
    //the game end effect of a pass during search, hash this in.
    if(hist.shouldSuppressEndGameFromFriendlyPass(board,nextPlayer))
      hash ^= MiscNNInputParams::ZOBRIST_FRIENDLY_PASS;

    //Passing hacks can also affect things at game or phase end.
    if(nnInputParams.enablePassingHacks)
      hash ^= MiscNNInputParams::ZOBRIST_PASSING_HACKS;
  }
  //Fold in whether the game is over or not, since this affects how we compute input features
  //but is not a function necessarily of previous hashed values.
  //If the history is in a weird prolonged state, also treat it similarly.
  if(hist.isGameFinished || hist.isPastNormalPhaseEnd)
    hash ^= Board::ZOBRIST_GAME_IS_OVER;

  //Distinguish between forcing the history to always be empty and allow any nonempty amount
  //We already tolerate caching and reusing evals across distinct transpositions with different
  //history, for performance reasons, but the case of no history at all is probably distinct
  //enough that we should distinguish it.
  if(nnInputParams.maxHistory <= 0) {
    hash.hash0 += MiscNNInputParams::ZOBRIST_ZERO_HISTORY.hash0;
    hash.hash1 += MiscNNInputParams::ZOBRIST_ZERO_HISTORY.hash1;
  }

  //Fold in asymmetric playout indicator
  if(nnInputParams.playoutDoublingAdvantage != 0) {
    int64_t playoutDoublingsDiscretized = (int64_t)(nnInputParams.playoutDoublingAdvantage*256.0f);
    hash.hash0 += Hash::splitMix64((uint64_t)playoutDoublingsDiscretized);
    hash.hash1 += Hash::basicLCong((uint64_t)playoutDoublingsDiscretized);
    hash ^= MiscNNInputParams::ZOBRIST_PLAYOUT_DOUBLINGS;
  }

  //Fold in policy temperature
  if(nnInputParams.nnPolicyTemperature != 1.0f) {
    int64_t nnPolicyTemperatureDiscretized = (int64_t)(nnInputParams.nnPolicyTemperature*2048.0f);
    hash.hash0 ^= Hash::basicLCong2((uint64_t)nnPolicyTemperatureDiscretized);
    hash.hash1 = Hash::splitMix64(hash.hash1 + (uint64_t)nnPolicyTemperatureDiscretized);
    hash.hash0 += hash.hash1;
    hash ^= MiscNNInputParams::ZOBRIST_NN_POLICY_TEMP;
  }

  if(nnInputParams.avoidMYTDaggerHack)
    hash ^= MiscNNInputParams::ZOBRIST_AVOID_MYTDAGGER_HACK;

  //Fold in policy optimism
  if(nnInputParams.policyOptimism > 0) {
    hash ^= MiscNNInputParams::ZOBRIST_POLICY_OPTIMISM;
    int64_t policyOptimismDiscretized = (int64_t)(nnInputParams.policyOptimism*1024.0);
    hash.hash0 = Hash::rrmxmx(Hash::splitMix64(hash.hash0) + (uint64_t)policyOptimismDiscretized);
    hash.hash1 = Hash::rrmxmx(hash.hash1 + hash.hash0 + (uint64_t)policyOptimismDiscretized);
  }

  return hash;
}

bool MiscNNInputParams::getAlwaysComputePassAliveUnderSuicideRules(const BoardHistory& hist) const {
  if(passAliveSuicideRulesOverride >= 0)
    return passAliveSuicideRulesOverride != 0;
  return hist.modes.alwaysComputePassAliveUnderSuicideRules;
}

bool MiscNNInputParams::getExcludeTerritoryAdjacentToAtari(const BoardHistory& hist) const {
  if(excludeTerritoryAdjAtariOverride >= 0)
    return excludeTerritoryAdjAtariOverride != 0;
  return hist.modes.excludeTerritoryAdjacentToAtari;
}

BoardHistoryModes MiscNNInputParams::getModes(const BoardHistory& hist) const {
  return BoardHistoryModes(
    getAlwaysComputePassAliveUnderSuicideRules(hist),
    getExcludeTerritoryAdjacentToAtari(hist)
  );
}

bool MiscNNInputParams::getSuicideLegalForPassAlive(const BoardHistory& hist) const {
  return hist.rules.multiStoneSuicideLegal || getAlwaysComputePassAliveUnderSuicideRules(hist);
}

//===========================================================================================
//INPUTSVERSION 3
//===========================================================================================

void NNInputs::fillRowV3(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  assert(nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= nnXLen);
  assert(board.y_size <= nnYLen);
  std::fill(rowBin,rowBin+NUM_FEATURES_SPATIAL_V3*nnXLen*nnYLen,false);
  std::fill(rowGlobal,rowGlobal+NUM_FEATURES_GLOBAL_V3,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_SPATIAL_V3;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowBin(rowBin,pos,0, 1.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 - 1,2,3 libs
      if(stone == pla)
        setRowBin(rowBin,pos,1, 1.0f, posStride, featureStride);
      else if(stone == opp)
        setRowBin(rowBin,pos,2, 1.0f, posStride, featureStride);

      if(stone == pla || stone == opp) {
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowBin(rowBin,pos,3, 1.0f, posStride, featureStride);
        else if(libs == 2) setRowBin(rowBin,pos,4, 1.0f, posStride, featureStride);
        else if(libs == 3) setRowBin(rowBin,pos,5, 1.0f, posStride, featureStride);
      }
    }
  }

  //Feature 6 - ko-ban locations, including possibly superko.
  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(board.ko_loc,xSize,nnXLen,nnYLen);
      setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc) {
          int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
          setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
        }
      }
    }
  }
  else {
    //Feature 6,7,8 - in the encore, no-second-ko-capture locations, encore ko prohibitions where we have to pass for ko
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.superKoBanned[loc])
          setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
        if(hist.koRecapBlocked[loc])
          setRowBin(rowBin,pos,7, 1.0f, posStride, featureStride);
      }
    }
  }

  //Hide history from the net if a pass would end things and we're behaving as if a pass won't.
  //Or if the game is in fact over right now!
  bool hideHistory =
    hist.isGameFinished ||
    hist.isPastNormalPhaseEnd ||
    (hist.passWouldEndGame(board,nextPlayer) && (
      nnInputParams.conservativePassAndIsRoot ||
      hist.shouldSuppressEndGameFromFriendlyPass(board,nextPlayer)
    ));
  int numTurnsOfHistoryIncluded = 0;

  //Features 9,10,11,12,13
  if(!hideHistory) {
    const vector<Move>& moveHistory = hist.moveHistory;
    size_t moveHistoryLen = moveHistory.size();
    if(moveHistoryLen >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
      Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
      numTurnsOfHistoryIncluded = 1;
      if(prev1Loc == Board::PASS_LOC)
        rowGlobal[0] = 1.0;
      else if(prev1Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev1Loc,xSize,nnXLen,nnYLen);
        setRowBin(rowBin,pos,9, 1.0f, posStride, featureStride);
      }
      if(moveHistoryLen >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
        Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
        numTurnsOfHistoryIncluded = 2;
        if(prev2Loc == Board::PASS_LOC)
          rowGlobal[1] = 1.0;
        else if(prev2Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev2Loc,xSize,nnXLen,nnYLen);
          setRowBin(rowBin,pos,10, 1.0f, posStride, featureStride);
        }
        if(moveHistoryLen >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
          Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
          numTurnsOfHistoryIncluded = 3;
          if(prev3Loc == Board::PASS_LOC)
            rowGlobal[2] = 1.0;
          else if(prev3Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev3Loc,xSize,nnXLen,nnYLen);
            setRowBin(rowBin,pos,11, 1.0f, posStride, featureStride);
          }
          if(moveHistoryLen >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
            Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
            numTurnsOfHistoryIncluded = 4;
            if(prev4Loc == Board::PASS_LOC)
              rowGlobal[3] = 1.0;
            else if(prev4Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev4Loc,xSize,nnXLen,nnYLen);
              setRowBin(rowBin,pos,12, 1.0f, posStride, featureStride);
            }
            if(moveHistoryLen >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
              Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
              numTurnsOfHistoryIncluded = 5;
              if(prev5Loc == Board::PASS_LOC)
                rowGlobal[4] = 1.0;
              else if(prev5Loc != Board::NULL_LOC) {
                int pos = NNPos::locToPos(prev5Loc,xSize,nnXLen,nnYLen);
                setRowBin(rowBin,pos,13, 1.0f, posStride, featureStride);
              }
            }
          }
        }
      }
    }
  }

  //Ladder features 14,15,16,17
  auto addLadderFeature = [&board,xSize,nnXLen,nnYLen,posStride,featureStride,rowBin,opp](Loc loc, int pos, const vector<Loc>& workingMoves)  noexcept {
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBin(rowBin,pos,14, 1.0f, posStride, featureStride);
    if(board.colors[loc] == opp && board.getNumLiberties(loc) > 1) {
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = NNPos::locToPos(workingMoves[j],xSize,nnXLen,nnYLen);
        setRowBin(rowBin,workingPos,17, 1.0f, posStride, featureStride);
      }
    }
  };

  iterLadders(board, nnXLen, addLadderFeature);

  const Board& prevBoard = (hideHistory || numTurnsOfHistoryIncluded < 1) ? board : hist.getRecentBoard(1);
  auto addPrevLadderFeature = [&prevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves) noexcept {
    (void)workingMoves;
    (void)loc;
    assert(prevBoard.colors[loc] == P_BLACK || prevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBin(rowBin,pos,15, 1.0f, posStride, featureStride);
  };
  iterLadders(prevBoard, nnXLen, addPrevLadderFeature);

  const Board& prevPrevBoard = (hideHistory || numTurnsOfHistoryIncluded < 2) ? prevBoard : hist.getRecentBoard(2);
  auto addPrevPrevLadderFeature = [&prevPrevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves) noexcept {
    (void)workingMoves;
    (void)loc;
    assert(prevPrevBoard.colors[loc] == P_BLACK || prevPrevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBin(rowBin,pos,16, 1.0f, posStride, featureStride);
  };
  iterLadders(prevPrevBoard, nnXLen, addPrevPrevLadderFeature);

  //Features 18,19 - current territory
  Color area[Board::MAX_ARR_SIZE];
  bool nonPassAliveStones;
  bool safeBigTerritories;
  bool unsafeBigTerritories;
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {
    nonPassAliveStones = true;
    safeBigTerritories = true;
    unsafeBigTerritories = true;
  }
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY) {
    nonPassAliveStones = false;
    safeBigTerritories = true;
    unsafeBigTerritories = false;
  }
  else {
    ASSERT_UNREACHABLE;
  }
  board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,nnInputParams.getSuicideLegalForPassAlive(hist));

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      Loc loc = Location::getLoc(x,y,xSize);
      int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
      if(area[loc] == pla)
        setRowBin(rowBin,pos,18, 1.0f, posStride, featureStride);
      else if(area[loc] == opp)
        setRowBin(rowBin,pos,19, 1.0f, posStride, featureStride);
    }
  }

  //Features 20, 21 - second encore starting stones
  if(hist.encorePhase >= 2) {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.secondEncoreStartColors[loc] == pla)
          setRowBin(rowBin,pos,20, 1.0f, posStride, featureStride);
        else if(hist.secondEncoreStartColors[loc] == opp)
          setRowBin(rowBin,pos,21, 1.0f, posStride, featureStride);
      }
    }
  }


  //Global features.
  //The first 5 of them were set already above to flag which of the past 5 moves were passes.

  //Komi and any score adjustments
  float selfKomi = hist.currentSelfKomi(nextPlayer,nnInputParams.drawEquivalentWinsForWhite);
  float bArea = (float)(xSize * ySize);
  //Bound komi just in case
  if(selfKomi > bArea+1.0f)
    selfKomi = bArea+1.0f;
  if(selfKomi < -bArea-1.0f)
    selfKomi = -bArea-1.0f;
  rowGlobal[5] = selfKomi/15.0f;

  //Ko rule
  if(hist.rules.koRule == Rules::KO_SIMPLE) {}
  else if(hist.rules.koRule == Rules::KO_POSITIONAL || hist.rules.koRule == Rules::KO_SPIGHT) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = 0.5f;
  }
  else if(hist.rules.koRule == Rules::KO_SITUATIONAL) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = -0.5f;
  }
  else
    ASSERT_UNREACHABLE;

  //Suicide
  if(hist.rules.multiStoneSuicideLegal)
    rowGlobal[8] = 1.0f;

  //Scoring
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {}
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY)
    rowGlobal[9] = 1.0f;
  else
    ASSERT_UNREACHABLE;

  //Encore phase
  if(hist.encorePhase > 0)
    rowGlobal[10] = 1.0f;
  if(hist.encorePhase > 1)
    rowGlobal[11] = 1.0f;

  //Does a pass end the current phase given the ruleset and history?
  bool passWouldEndPhase = hideHistory ? false : hist.passWouldEndPhase(board,nextPlayer);
  rowGlobal[12] = passWouldEndPhase ? 1.0f : 0.0f;

  //Provide parity information about the board size and komi
  //This comes from the following observation:
  //From white's perspective:
  //Komi = 0.0 - Draw possible
  //Komi = 0.5 - Win the games we would have drawn with komi 0.0
  //Komi = 1.0 - Usually no difference from komi 0.5
  //Komi = 1.5 - Usually no difference from komi 0.5
  //Komi = 2.0 - Draw possible
  //If we were to assign an "effective goodness" to these komis in order it would look like
  //0 1 1 1 2 3 3 3 4 5 5 5 6 ...
  //since when away from the right parity, increasing the komi doesn't help us except in cases of seki with odd numbers of dame.
  //If we were to add 0.5 times a vector like:
  //0 -1 0 1 0 -1 0 1 0 -1 0 ...
  //Then this would become a linear function and hopefully easier for a neural net to learn.
  //We expect that this is hard for a neural net to learn since it depends on the parity of the board size
  //and is very "xor"like.
  //So we provide it as an input.
  //Since we are using a model where games are jittered by 0.5 (see BoardHistory::whiteKomiAdjustmentForDraws)
  //in theory right thing to first order to provide should be a triangular wave with a period of 2 komi points:
  //  ../\........
  //  ./..\.......
  //  /....\..../.
  //  ......\../..
  //  .......\/...
  //The upsloping part of the wave is centered around the komi value where you could draw
  //since komi is extra valuable when it turns losses into draws into wins, peaking at the komi value where you could draw + 0.5.
  //It's downsloping around the komi value where you can't draw, since the marginal komi there is nearly useless, not causing you to win
  //more games except in case of odd-dame seki.

  if(hist.rules.scoringRule == Rules::SCORING_AREA || hist.encorePhase >= 2) {
    bool boardAreaIsEven = (xSize*ySize) % 2 == 0;

    //What is the parity of the komi values that can produce jigos?
    bool drawableKomisAreEven = boardAreaIsEven;

    //Find the difference between the komi viewed from our perspective and the nearest drawable komi below it.
    float komiFloor;
    if(drawableKomisAreEven)
      komiFloor = floor(selfKomi / 2.0f) * 2.0f;
    else
      komiFloor = floor((selfKomi-1.0f) / 2.0f) * 2.0f + 1.0f;

    //Cap just in case we have floating point weirdness
    float delta = selfKomi - komiFloor;
    assert(delta >= -0.0001f);
    assert(delta <= 2.0001f);
    if(delta < 0.0f)
      delta = 0.0f;
    if(delta > 2.0f)
      delta = 2.0f;

    //Create the triangle wave based on the difference
    float wave;
    if(delta < 0.5f)
      wave = delta;
    else if(delta < 1.5f)
      wave = 1.0f-delta;
    else
      wave = delta-2.0f;

    //NOTE: If ever changing which feature this is, must also update index in model.py where we multiply it into the scorebelief parity vector
    rowGlobal[13] = wave;
  }

}


//===========================================================================================
//INPUTSVERSION 4
//===========================================================================================

void NNInputs::fillRowV4(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  assert(nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= nnXLen);
  assert(board.y_size <= nnYLen);
  std::fill(rowBin,rowBin+NUM_FEATURES_SPATIAL_V4*nnXLen*nnYLen,false);
  std::fill(rowGlobal,rowGlobal+NUM_FEATURES_GLOBAL_V4,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_SPATIAL_V4;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowBin(rowBin,pos,0, 1.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 - 1,2,3 libs
      if(stone == pla)
        setRowBin(rowBin,pos,1, 1.0f, posStride, featureStride);
      else if(stone == opp)
        setRowBin(rowBin,pos,2, 1.0f, posStride, featureStride);

      if(stone == pla || stone == opp) {
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowBin(rowBin,pos,3, 1.0f, posStride, featureStride);
        else if(libs == 2) setRowBin(rowBin,pos,4, 1.0f, posStride, featureStride);
        else if(libs == 3) setRowBin(rowBin,pos,5, 1.0f, posStride, featureStride);
      }
    }
  }

  //Feature 6 - ko-ban locations, including possibly superko.
  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(board.ko_loc,xSize,nnXLen,nnYLen);
      setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc) {
          int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
          setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
        }
      }
    }
  }
  else {
    //Feature 6,7,8 - in the encore, no-second-ko-capture locations, encore ko prohibitions where we have to pass for ko
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.superKoBanned[loc])
          setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
        if(hist.koRecapBlocked[loc])
          setRowBin(rowBin,pos,7, 1.0f, posStride, featureStride);
      }
    }
  }

  //Hide history from the net if a pass would end things and we're behaving as if a pass won't.
  //Or if the game is in fact over right now!
  bool hideHistory =
    hist.isGameFinished ||
    hist.isPastNormalPhaseEnd ||
    (hist.passWouldEndGame(board,nextPlayer) && (
      nnInputParams.conservativePassAndIsRoot ||
      hist.shouldSuppressEndGameFromFriendlyPass(board,nextPlayer)
    ));
  int numTurnsOfHistoryIncluded = 0;

  //Features 9,10,11,12,13
  if(!hideHistory) {
    const vector<Move>& moveHistory = hist.moveHistory;
    size_t moveHistoryLen = moveHistory.size();
    if(moveHistoryLen >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
      Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
      numTurnsOfHistoryIncluded = 1;
      if(prev1Loc == Board::PASS_LOC)
        rowGlobal[0] = 1.0;
      else if(prev1Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev1Loc,xSize,nnXLen,nnYLen);
        setRowBin(rowBin,pos,9, 1.0f, posStride, featureStride);
      }
      if(moveHistoryLen >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
        Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
        numTurnsOfHistoryIncluded = 2;
        if(prev2Loc == Board::PASS_LOC)
          rowGlobal[1] = 1.0;
        else if(prev2Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev2Loc,xSize,nnXLen,nnYLen);
          setRowBin(rowBin,pos,10, 1.0f, posStride, featureStride);
        }
        if(moveHistoryLen >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
          Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
          numTurnsOfHistoryIncluded = 3;
          if(prev3Loc == Board::PASS_LOC)
            rowGlobal[2] = 1.0;
          else if(prev3Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev3Loc,xSize,nnXLen,nnYLen);
            setRowBin(rowBin,pos,11, 1.0f, posStride, featureStride);
          }
          if(moveHistoryLen >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
            Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
            numTurnsOfHistoryIncluded = 4;
            if(prev4Loc == Board::PASS_LOC)
              rowGlobal[3] = 1.0;
            else if(prev4Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev4Loc,xSize,nnXLen,nnYLen);
              setRowBin(rowBin,pos,12, 1.0f, posStride, featureStride);
            }
            if(moveHistoryLen >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
              Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
              numTurnsOfHistoryIncluded = 5;
              if(prev5Loc == Board::PASS_LOC)
                rowGlobal[4] = 1.0;
              else if(prev5Loc != Board::NULL_LOC) {
                int pos = NNPos::locToPos(prev5Loc,xSize,nnXLen,nnYLen);
                setRowBin(rowBin,pos,13, 1.0f, posStride, featureStride);
              }
            }
          }
        }
      }
    }
  }

  //Ladder features 14,15,16,17
  auto addLadderFeature = [&board,xSize,nnXLen,nnYLen,posStride,featureStride,rowBin,opp](Loc loc, int pos, const vector<Loc>& workingMoves) noexcept {
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBin(rowBin,pos,14, 1.0f, posStride, featureStride);
    if(board.colors[loc] == opp && board.getNumLiberties(loc) > 1) {
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = NNPos::locToPos(workingMoves[j],xSize,nnXLen,nnYLen);
        setRowBin(rowBin,workingPos,17, 1.0f, posStride, featureStride);
      }
    }
  };

  iterLadders(board, nnXLen, addLadderFeature);

  const Board& prevBoard = (hideHistory || numTurnsOfHistoryIncluded < 1) ? board : hist.getRecentBoard(1);
  auto addPrevLadderFeature = [&prevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves) noexcept {
    (void)workingMoves;
    (void)loc;
    assert(prevBoard.colors[loc] == P_BLACK || prevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBin(rowBin,pos,15, 1.0f, posStride, featureStride);
  };
  iterLadders(prevBoard, nnXLen, addPrevLadderFeature);

  const Board& prevPrevBoard = (hideHistory || numTurnsOfHistoryIncluded < 2) ? prevBoard : hist.getRecentBoard(2);
  auto addPrevPrevLadderFeature = [&prevPrevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves) noexcept {
    (void)workingMoves;
    (void)loc;
    assert(prevPrevBoard.colors[loc] == P_BLACK || prevPrevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBin(rowBin,pos,16, 1.0f, posStride, featureStride);
  };
  iterLadders(prevPrevBoard, nnXLen, addPrevPrevLadderFeature);

  //Features 18,19 - pass alive territory and stones
  Color area[Board::MAX_ARR_SIZE];
  {
    bool nonPassAliveStones = false;
    bool safeBigTerritories = true;
    bool unsafeBigTerritories = false;
    board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,nnInputParams.getSuicideLegalForPassAlive(hist));
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      Loc loc = Location::getLoc(x,y,xSize);
      int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
      if(area[loc] == pla)
        setRowBin(rowBin,pos,18, 1.0f, posStride, featureStride);
      else if(area[loc] == opp)
        setRowBin(rowBin,pos,19, 1.0f, posStride, featureStride);
    }
  }

  //Features 20, 21 - second encore starting stones
  if(hist.encorePhase >= 2) {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.secondEncoreStartColors[loc] == pla)
          setRowBin(rowBin,pos,20, 1.0f, posStride, featureStride);
        else if(hist.secondEncoreStartColors[loc] == opp)
          setRowBin(rowBin,pos,21, 1.0f, posStride, featureStride);
      }
    }
  }


  //Global features.
  //The first 5 of them were set already above to flag which of the past 5 moves were passes.

  //Komi and any score adjustments
  float selfKomi = hist.currentSelfKomi(nextPlayer,nnInputParams.drawEquivalentWinsForWhite);
  float bArea = (float)(xSize * ySize);
  //Bound komi just in case
  if(selfKomi > bArea+1.0f)
    selfKomi = bArea+1.0f;
  if(selfKomi < -bArea-1.0f)
    selfKomi = -bArea-1.0f;
  rowGlobal[5] = selfKomi/15.0f;

  //Ko rule
  if(hist.rules.koRule == Rules::KO_SIMPLE) {}
  else if(hist.rules.koRule == Rules::KO_POSITIONAL || hist.rules.koRule == Rules::KO_SPIGHT) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = 0.5f;
  }
  else if(hist.rules.koRule == Rules::KO_SITUATIONAL) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = -0.5f;
  }
  else
    ASSERT_UNREACHABLE;

  //Suicide
  if(hist.rules.multiStoneSuicideLegal)
    rowGlobal[8] = 1.0f;

  //Scoring
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {}
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY)
    rowGlobal[9] = 1.0f;
  else
    ASSERT_UNREACHABLE;

  //Encore phase
  if(hist.encorePhase > 0)
    rowGlobal[10] = 1.0f;
  if(hist.encorePhase > 1)
    rowGlobal[11] = 1.0f;

  //Does a pass end the current phase given the ruleset and history?
  bool passWouldEndPhase = hideHistory ? false : hist.passWouldEndPhase(board,nextPlayer);
  rowGlobal[12] = passWouldEndPhase ? 1.0f : 0.0f;

  //Provide parity information about the board size and komi
  //This comes from the following observation:
  //From white's perspective:
  //Komi = 0.0 - Draw possible
  //Komi = 0.5 - Win the games we would have drawn with komi 0.0
  //Komi = 1.0 - Usually no difference from komi 0.5
  //Komi = 1.5 - Usually no difference from komi 0.5
  //Komi = 2.0 - Draw possible
  //If we were to assign an "effective goodness" to these komis in order it would look like
  //0 1 1 1 2 3 3 3 4 5 5 5 6 ...
  //since when away from the right parity, increasing the komi doesn't help us except in cases of seki with odd numbers of dame.
  //If we were to add 0.5 times a vector like:
  //0 -1 0 1 0 -1 0 1 0 -1 0 ...
  //Then this would become a linear function and hopefully easier for a neural net to learn.
  //We expect that this is hard for a neural net to learn since it depends on the parity of the board size
  //and is very "xor"like.
  //So we provide it as an input.
  //Since we are using a model where games are jittered by 0.5 (see BoardHistory::whiteKomiAdjustmentForDraws)
  //in theory right thing to first order to provide should be a triangular wave with a period of 2 komi points:
  //  ../\........
  //  ./..\.......
  //  /....\..../.
  //  ......\../..
  //  .......\/...
  //The upsloping part of the wave is centered around the komi value where you could draw
  //since komi is extra valuable when it turns losses into draws into wins, peaking at the komi value where you could draw + 0.5.
  //It's downsloping around the komi value where you can't draw, since the marginal komi there is nearly useless, not causing you to win
  //more games except in case of odd-dame seki.

  if(hist.rules.scoringRule == Rules::SCORING_AREA || hist.encorePhase >= 2) {
    bool boardAreaIsEven = (xSize*ySize) % 2 == 0;

    //What is the parity of the komi values that can produce jigos?
    bool drawableKomisAreEven = boardAreaIsEven;

    //Find the difference between the komi viewed from our perspective and the nearest drawable komi below it.
    float komiFloor;
    if(drawableKomisAreEven)
      komiFloor = floor(selfKomi / 2.0f) * 2.0f;
    else
      komiFloor = floor((selfKomi-1.0f) / 2.0f) * 2.0f + 1.0f;

    //Cap just in case we have floating point weirdness
    float delta = selfKomi - komiFloor;
    assert(delta >= -0.0001f);
    assert(delta <= 2.0001f);
    if(delta < 0.0f)
      delta = 0.0f;
    if(delta > 2.0f)
      delta = 2.0f;

    //Create the triangle wave based on the difference
    float wave;
    if(delta < 0.5f)
      wave = delta;
    else if(delta < 1.5f)
      wave = 1.0f-delta;
    else
      wave = delta-2.0f;

    //NOTE: If ever changing which feature this is, must also update index in model.py where we multiply it into the scorebelief parity vector
    rowGlobal[13] = wave;
  }

}



//===========================================================================================
//INPUTSVERSION 5
//===========================================================================================

void NNInputs::fillRowV5(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  assert(nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= nnXLen);
  assert(board.y_size <= nnYLen);
  std::fill(rowBin,rowBin+NUM_FEATURES_SPATIAL_V5*nnXLen*nnYLen,false);
  std::fill(rowGlobal,rowGlobal+NUM_FEATURES_GLOBAL_V5,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_SPATIAL_V5;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowBin(rowBin,pos,0, 1.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      if(stone == pla)
        setRowBin(rowBin,pos,1, 1.0f, posStride, featureStride);
      else if(stone == opp)
        setRowBin(rowBin,pos,2, 1.0f, posStride, featureStride);
    }
  }

  //Feature 3 - ko-ban locations, including possibly superko.
  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(board.ko_loc,xSize,nnXLen,nnYLen);
      setRowBin(rowBin,pos,3, 1.0f, posStride, featureStride);
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc) {
          int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
          setRowBin(rowBin,pos,3, 1.0f, posStride, featureStride);
        }
      }
    }
  }
  else {
    //Feature 3,4,5 - in the encore, no-second-ko-capture locations, encore ko prohibitions where we have to pass for ko
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.superKoBanned[loc])
          setRowBin(rowBin,pos,3, 1.0f, posStride, featureStride);
        if(hist.koRecapBlocked[loc])
          setRowBin(rowBin,pos,4, 1.0f, posStride, featureStride);
      }
    }
  }

  //Hide history from the net if a pass would end things and we're behaving as if a pass won't.
  //Or if the game is in fact over right now!
  bool hideHistory =
    hist.isGameFinished ||
    hist.isPastNormalPhaseEnd ||
    (hist.passWouldEndGame(board,nextPlayer) && (
      nnInputParams.conservativePassAndIsRoot ||
      hist.shouldSuppressEndGameFromFriendlyPass(board,nextPlayer)
    ));

  //Features 6,7,8,9,10
  if(!hideHistory) {
    const vector<Move>& moveHistory = hist.moveHistory;
    size_t moveHistoryLen = moveHistory.size();
    if(moveHistoryLen >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
      Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
      if(prev1Loc == Board::PASS_LOC)
        rowGlobal[0] = 1.0;
      else if(prev1Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev1Loc,xSize,nnXLen,nnYLen);
        setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
      }
      if(moveHistoryLen >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
        Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
        if(prev2Loc == Board::PASS_LOC)
          rowGlobal[1] = 1.0;
        else if(prev2Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev2Loc,xSize,nnXLen,nnYLen);
          setRowBin(rowBin,pos,7, 1.0f, posStride, featureStride);
        }
        if(moveHistoryLen >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
          Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
          if(prev3Loc == Board::PASS_LOC)
            rowGlobal[2] = 1.0;
          else if(prev3Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev3Loc,xSize,nnXLen,nnYLen);
            setRowBin(rowBin,pos,8, 1.0f, posStride, featureStride);
          }
          if(moveHistoryLen >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
            Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
            if(prev4Loc == Board::PASS_LOC)
              rowGlobal[3] = 1.0;
            else if(prev4Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev4Loc,xSize,nnXLen,nnYLen);
              setRowBin(rowBin,pos,9, 1.0f, posStride, featureStride);
            }
            if(moveHistoryLen >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
              Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
              if(prev5Loc == Board::PASS_LOC)
                rowGlobal[4] = 1.0;
              else if(prev5Loc != Board::NULL_LOC) {
                int pos = NNPos::locToPos(prev5Loc,xSize,nnXLen,nnYLen);
                setRowBin(rowBin,pos,10, 1.0f, posStride, featureStride);
              }
            }
          }
        }
      }
    }
  }

  //Features 11, 12 - second encore starting stones
  if(hist.encorePhase >= 2) {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.secondEncoreStartColors[loc] == pla)
          setRowBin(rowBin,pos,11, 1.0f, posStride, featureStride);
        else if(hist.secondEncoreStartColors[loc] == opp)
          setRowBin(rowBin,pos,12, 1.0f, posStride, featureStride);
      }
    }
  }


  //Global features.
  //The first 5 of them were set already above to flag which of the past 5 moves were passes.

  //Komi and any score adjustments
  float selfKomi = hist.currentSelfKomi(nextPlayer,nnInputParams.drawEquivalentWinsForWhite);
  float bArea = (float)(xSize * ySize);
  //Bound komi just in case
  if(selfKomi > bArea+1.0f)
    selfKomi = bArea+1.0f;
  if(selfKomi < -bArea-1.0f)
    selfKomi = -bArea-1.0f;
  rowGlobal[5] = selfKomi/15.0f;

  //Ko rule
  if(hist.rules.koRule == Rules::KO_SIMPLE) {}
  else if(hist.rules.koRule == Rules::KO_POSITIONAL || hist.rules.koRule == Rules::KO_SPIGHT) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = 0.5f;
  }
  else if(hist.rules.koRule == Rules::KO_SITUATIONAL) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = -0.5f;
  }
  else
    ASSERT_UNREACHABLE;

  //Suicide
  if(hist.rules.multiStoneSuicideLegal)
    rowGlobal[8] = 1.0f;

  //Scoring
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {}
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY)
    rowGlobal[9] = 1.0f;
  else
    ASSERT_UNREACHABLE;

  //Encore phase
  if(hist.encorePhase > 0)
    rowGlobal[10] = 1.0f;
  if(hist.encorePhase > 1)
    rowGlobal[11] = 1.0f;

}

//===========================================================================================
//INPUTSVERSION 6
//===========================================================================================


void NNInputs::fillRowV6(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  assert(nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= nnXLen);
  assert(board.y_size <= nnYLen);
  std::fill(rowBin,rowBin+NUM_FEATURES_SPATIAL_V6*nnXLen*nnYLen,false);
  std::fill(rowGlobal,rowGlobal+NUM_FEATURES_GLOBAL_V6,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_SPATIAL_V6;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowBin(rowBin,pos,0, 1.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 - 1,2,3 libs
      if(stone == pla)
        setRowBin(rowBin,pos,1, 1.0f, posStride, featureStride);
      else if(stone == opp)
        setRowBin(rowBin,pos,2, 1.0f, posStride, featureStride);

      if(stone == pla || stone == opp) {
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowBin(rowBin,pos,3, 1.0f, posStride, featureStride);
        else if(libs == 2) setRowBin(rowBin,pos,4, 1.0f, posStride, featureStride);
        else if(libs == 3) setRowBin(rowBin,pos,5, 1.0f, posStride, featureStride);
      }
    }
  }

  //Feature 6 - ko-ban locations, including possibly superko.
  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(board.ko_loc,xSize,nnXLen,nnYLen);
      setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc) {
          int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
          setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
        }
      }
    }
  }
  else {
    //Feature 6,7,8 - in the encore, no-second-ko-capture locations, encore ko prohibitions where we have to pass for ko
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.superKoBanned[loc])
          setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
        if(hist.koRecapBlocked[loc])
          setRowBin(rowBin,pos,7, 1.0f, posStride, featureStride);
      }
    }
  }

  //Features 18,19 - current territory, not counting group tax
  Color area[Board::MAX_ARR_SIZE];
  bool hasAreaFeature = false;
  int groupTaxAdjustmentForPla = 0;
  if(hist.rules.scoringRule == Rules::SCORING_AREA && hist.rules.taxRule == Rules::TAX_NONE) {
    hasAreaFeature = true;
    bool nonPassAliveStones = true;
    bool safeBigTerritories = true;
    bool unsafeBigTerritories = true;
    board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,nnInputParams.getSuicideLegalForPassAlive(hist));
  }
  else {
    bool keepTerritories = false;
    bool keepStones = false;
    int whiteMinusBlackIndependentLifeRegionCount = 0;
    if(hist.rules.scoringRule == Rules::SCORING_AREA && (hist.rules.taxRule == Rules::TAX_SEKI || hist.rules.taxRule == Rules::TAX_ALL)) {
      hasAreaFeature = true;
      keepTerritories = false;
      keepStones = true;
    }
    else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && hist.rules.taxRule == Rules::TAX_NONE) {
      //Territory scoring omits feature until we reach the stage where scoring matters
      if(hist.encorePhase >= 2) {
        hasAreaFeature = true;
        keepTerritories = true;
        keepStones = false;
      }
    }
    else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && (hist.rules.taxRule == Rules::TAX_SEKI || hist.rules.taxRule == Rules::TAX_ALL)) {
      //Territory scoring omits feature until we reach the stage where scoring matters
      if(hist.encorePhase >= 2) {
        hasAreaFeature = true;
        keepTerritories = false;
        keepStones = false;
      }
    }
    else {
      ASSERT_UNREACHABLE;
    }

    if(hasAreaFeature) {
      board.calculateIndependentLifeArea(
        area,whiteMinusBlackIndependentLifeRegionCount,
        keepTerritories,
        keepStones,
        nnInputParams.getExcludeTerritoryAdjacentToAtari(hist),
        nnInputParams.getSuicideLegalForPassAlive(hist)
      );
      if(hist.rules.taxRule == Rules::TAX_ALL)
        groupTaxAdjustmentForPla = pla == P_WHITE ? -2 * whiteMinusBlackIndependentLifeRegionCount : 2 * whiteMinusBlackIndependentLifeRegionCount;
    }
  }

  bool finalPhaseAndGameEndWouldNotBeWin = false;
  if(hasAreaFeature) {
    int boardScoreForPla = groupTaxAdjustmentForPla;
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(area[loc] == pla) {
          setRowBin(rowBin,pos,18, 1.0f, posStride, featureStride);
          boardScoreForPla += 1;
        }
        else if(area[loc] == opp) {
          setRowBin(rowBin,pos,19, 1.0f, posStride, featureStride);
          boardScoreForPla -= 1;
        }
        else {
          if(hist.rules.scoringRule == Rules::SCORING_TERRITORY) {
            //Also we must be in the second encore phase, based on the logic above.
            if(board.colors[loc] == pla && hist.secondEncoreStartColors[loc] == pla) {
              setRowBin(rowBin,pos,18, 1.0f, posStride, featureStride);
              boardScoreForPla += 1;
            }
            else if(board.colors[loc] == opp && hist.secondEncoreStartColors[loc] == opp) {
              setRowBin(rowBin,pos,19, 1.0f, posStride, featureStride);
              boardScoreForPla -= 1;
            }
          }
        }
      }
    }
    float selfKomi = hist.currentSelfKomi(pla, nnInputParams.drawEquivalentWinsForWhite);
    float finalScorePla = (float)boardScoreForPla + selfKomi;
    // If the game ended here, and was scored instantly, it would be a loss or a draw?
    if(finalScorePla <= 0.0)
      finalPhaseAndGameEndWouldNotBeWin = true;
  }

  //Hide history from the net if a pass would end things and we're behaving as if a pass won't.
  //Or if the game is in fact over right now!
  int maxTurnsOfHistoryToInclude = 5;
  bool suppressPassWouldEndPhase = false;
  if(hist.passWouldEndGame(board,nextPlayer) && (
       //At the root, if assuming passing doesn't end the game, and it would, then need to mask that out.
       nnInputParams.conservativePassAndIsRoot ||
       //Deeper in the tree, we might not assume passes end the game in a friendly pass setting.
       hist.shouldSuppressEndGameFromFriendlyPass(board,nextPlayer) ||
       //Passing hacks suppress the net to end the game when losing if it thinks a premature pass will lose by less.
       (nnInputParams.enablePassingHacks && finalPhaseAndGameEndWouldNotBeWin)
     )
  ) {
    maxTurnsOfHistoryToInclude = 0;
    suppressPassWouldEndPhase = true;
  }
  else if(hist.isGameFinished || hist.isPastNormalPhaseEnd) {
    // Include one of the passes, at the end of that sequence
    maxTurnsOfHistoryToInclude = 1;
  }
  maxTurnsOfHistoryToInclude = std::min(maxTurnsOfHistoryToInclude, nnInputParams.maxHistory);

  int numTurnsOfHistoryIncluded = 0;

  //Features 9,10,11,12,13
  if(maxTurnsOfHistoryToInclude > 0) {
    const vector<Move>& moveHistory = hist.moveHistory;
    size_t moveHistoryLen = moveHistory.size();
    assert(moveHistoryLen >= hist.numApproxValidTurnsThisPhase);
    assert(moveHistoryLen >= hist.numConsecValidTurnsThisGame);

    //Effectively wipe history as we change phase by also capping it, and also on any historical rules violations
    int amountOfHistoryToTryToUse = std::min(maxTurnsOfHistoryToInclude, std::min(hist.numApproxValidTurnsThisPhase,hist.numConsecValidTurnsThisGame));

    if(amountOfHistoryToTryToUse >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
      Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
      numTurnsOfHistoryIncluded = 1;
      if(prev1Loc == Board::PASS_LOC)
        rowGlobal[0] = 1.0;
      else if(prev1Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev1Loc,xSize,nnXLen,nnYLen);
        setRowBin(rowBin,pos,9, 1.0f, posStride, featureStride);
      }
      if(amountOfHistoryToTryToUse >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
        Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
        numTurnsOfHistoryIncluded = 2;
        if(prev2Loc == Board::PASS_LOC)
          rowGlobal[1] = 1.0;
        else if(prev2Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev2Loc,xSize,nnXLen,nnYLen);
          setRowBin(rowBin,pos,10, 1.0f, posStride, featureStride);
        }
        if(amountOfHistoryToTryToUse >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
          Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
          numTurnsOfHistoryIncluded = 3;
          if(prev3Loc == Board::PASS_LOC)
            rowGlobal[2] = 1.0;
          else if(prev3Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev3Loc,xSize,nnXLen,nnYLen);
            setRowBin(rowBin,pos,11, 1.0f, posStride, featureStride);
          }
          if(amountOfHistoryToTryToUse >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
            Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
            numTurnsOfHistoryIncluded = 4;
            if(prev4Loc == Board::PASS_LOC)
              rowGlobal[3] = 1.0;
            else if(prev4Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev4Loc,xSize,nnXLen,nnYLen);
              setRowBin(rowBin,pos,12, 1.0f, posStride, featureStride);
            }
            if(amountOfHistoryToTryToUse >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
              Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
              numTurnsOfHistoryIncluded = 5;
              if(prev5Loc == Board::PASS_LOC)
                rowGlobal[4] = 1.0;
              else if(prev5Loc != Board::NULL_LOC) {
                int pos = NNPos::locToPos(prev5Loc,xSize,nnXLen,nnYLen);
                setRowBin(rowBin,pos,13, 1.0f, posStride, featureStride);
              }
            }
          }
        }
      }
    }
  }

  //Ladder features 14,15,16,17
  auto addLadderFeature = [&board,xSize,nnXLen,nnYLen,posStride,featureStride,rowBin,opp](Loc loc, int pos, const vector<Loc>& workingMoves) noexcept {
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBin(rowBin,pos,14, 1.0f, posStride, featureStride);
    if(board.colors[loc] == opp && board.getNumLiberties(loc) > 1) {
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = NNPos::locToPos(workingMoves[j],xSize,nnXLen,nnYLen);
        setRowBin(rowBin,workingPos,17, 1.0f, posStride, featureStride);
      }
    }
  };

  iterLadders(board, nnXLen, addLadderFeature);

  const Board& prevBoard = (numTurnsOfHistoryIncluded < 1) ? board : hist.getRecentBoard(1);
  auto addPrevLadderFeature = [&prevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves) noexcept {
    (void)workingMoves;
    (void)loc;
    assert(prevBoard.colors[loc] == P_BLACK || prevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBin(rowBin,pos,15, 1.0f, posStride, featureStride);
  };
  iterLadders(prevBoard, nnXLen, addPrevLadderFeature);

  const Board& prevPrevBoard = (numTurnsOfHistoryIncluded < 2) ? prevBoard : hist.getRecentBoard(2);
  auto addPrevPrevLadderFeature = [&prevPrevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves) noexcept {
    (void)workingMoves;
    (void)loc;
    assert(prevPrevBoard.colors[loc] == P_BLACK || prevPrevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBin(rowBin,pos,16, 1.0f, posStride, featureStride);
  };
  iterLadders(prevPrevBoard, nnXLen, addPrevPrevLadderFeature);

  //Features 20, 21 - second encore starting stones
  if(hist.encorePhase >= 2) {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.secondEncoreStartColors[loc] == pla)
          setRowBin(rowBin,pos,20, 1.0f, posStride, featureStride);
        else if(hist.secondEncoreStartColors[loc] == opp)
          setRowBin(rowBin,pos,21, 1.0f, posStride, featureStride);
      }
    }
  }


  //Global features.
  //The first 5 of them were set already above to flag which of the past 5 moves were passes.

  //Komi and any score adjustments
  float selfKomi = hist.currentSelfKomi(nextPlayer,nnInputParams.drawEquivalentWinsForWhite);
  float bArea = (float)(xSize * ySize);
  //Bound komi just in case
  if(selfKomi > bArea+1.0f)
    selfKomi = bArea+1.0f;
  if(selfKomi < -bArea-1.0f)
    selfKomi = -bArea-1.0f;
  rowGlobal[5] = selfKomi/20.0f;

  //Ko rule
  if(hist.rules.koRule == Rules::KO_SIMPLE) {}
  else if(hist.rules.koRule == Rules::KO_POSITIONAL || hist.rules.koRule == Rules::KO_SPIGHT) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = 0.5f;
  }
  else if(hist.rules.koRule == Rules::KO_SITUATIONAL) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = -0.5f;
  }
  else
    ASSERT_UNREACHABLE;

  //Suicide
  if(hist.rules.multiStoneSuicideLegal)
    rowGlobal[8] = 1.0f;

  //Scoring
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {}
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY)
    rowGlobal[9] = 1.0f;
  else
    ASSERT_UNREACHABLE;
  //Tax
  if(hist.rules.taxRule == Rules::TAX_NONE) {}
  else if(hist.rules.taxRule == Rules::TAX_SEKI)
    rowGlobal[10] = 1.0f;
  else if(hist.rules.taxRule == Rules::TAX_ALL) {
    rowGlobal[10] = 1.0f;
    rowGlobal[11] = 1.0f;
  }
  else
    ASSERT_UNREACHABLE;

  //Encore phase
  if(hist.encorePhase > 0)
    rowGlobal[12] = 1.0f;
  if(hist.encorePhase > 1)
    rowGlobal[13] = 1.0f;

  //Does a pass end the current phase given the ruleset and history?
  bool passWouldEndPhase = suppressPassWouldEndPhase ? false : hist.passWouldEndPhase(board,nextPlayer);
  rowGlobal[14] = passWouldEndPhase ? 1.0f : 0.0f;

  //Provide parity information about the board size and komi
  //This comes from the following observation:
  //From white's perspective:
  //Komi = 0.0 - Draw possible
  //Komi = 0.5 - Win the games we would have drawn with komi 0.0
  //Komi = 1.0 - Usually no difference from komi 0.5
  //Komi = 1.5 - Usually no difference from komi 0.5
  //Komi = 2.0 - Draw possible
  //If we were to assign an "effective goodness" to these komis in order it would look like
  //0 1 1 1 2 3 3 3 4 5 5 5 6 ...
  //since when away from the right parity, increasing the komi doesn't help us except in cases of seki with odd numbers of dame.
  //If we were to add 0.5 times a vector like:
  //0 -1 0 1 0 -1 0 1 0 -1 0 ...
  //Then this would become a linear function and hopefully easier for a neural net to learn.
  //We expect that this is hard for a neural net to learn since it depends on the parity of the board size
  //and is very "xor"like.
  //So we provide it as an input.
  //Since we are using a model where games are jittered by 0.5 (see BoardHistory::whiteKomiAdjustmentForDraws)
  //in theory right thing to first order to provide should be a triangular wave with a period of 2 komi points:
  //  ../\........
  //  ./..\.......
  //  /....\..../.
  //  ......\../..
  //  .......\/...
  //The upsloping part of the wave is centered around the komi value where you could draw
  //since komi is extra valuable when it turns losses into draws into wins, peaking at the komi value where you could draw + 0.5.
  //It's downsloping around the komi value where you can't draw, since the marginal komi there is nearly useless, not causing you to win
  //more games except in case of odd-dame seki.

  if(hist.rules.scoringRule == Rules::SCORING_AREA || hist.encorePhase >= 2) {
    bool boardAreaIsEven = (xSize*ySize) % 2 == 0;

    //What is the parity of the komi values that can produce jigos?
    bool drawableKomisAreEven = boardAreaIsEven;

    //Find the difference between the komi viewed from our perspective and the nearest drawable komi below it.
    float komiFloor;
    if(drawableKomisAreEven)
      komiFloor = floor(selfKomi / 2.0f) * 2.0f;
    else
      komiFloor = floor((selfKomi-1.0f) / 2.0f) * 2.0f + 1.0f;

    //Cap just in case we have floating point weirdness
    float delta = selfKomi - komiFloor;
    assert(delta >= -0.0001f);
    assert(delta <= 2.0001f);
    if(delta < 0.0f)
      delta = 0.0f;
    if(delta > 2.0f)
      delta = 2.0f;

    //Create the triangle wave based on the difference
    float wave;
    if(delta < 0.5f)
      wave = delta;
    else if(delta < 1.5f)
      wave = 1.0f-delta;
    else
      wave = delta-2.0f;

    //NOTE: If ever changing which feature this is, must also update index in model.py where we multiply it into the scorebelief parity vector
    rowGlobal[15] = wave;
  }

}

//===========================================================================================
//INPUTSVERSION 7
//===========================================================================================


void NNInputs::fillRowV7(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  assert(nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= nnXLen);
  assert(board.y_size <= nnYLen);
  std::fill(rowBin,rowBin+NUM_FEATURES_SPATIAL_V7*nnXLen*nnYLen,false);
  std::fill(rowGlobal,rowGlobal+NUM_FEATURES_GLOBAL_V7,0.0f);

  Player pla = nextPlayer;
  Player opp = getOpp(pla);
  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_SPATIAL_V7;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      Loc loc = Location::getLoc(x,y,xSize);

      //Feature 0 - on board
      setRowBin(rowBin,pos,0, 1.0f, posStride, featureStride);

      Color stone = board.colors[loc];

      //Features 1,2 - pla,opp stone
      //Features 3,4,5 - 1,2,3 libs
      if(stone == pla)
        setRowBin(rowBin,pos,1, 1.0f, posStride, featureStride);
      else if(stone == opp)
        setRowBin(rowBin,pos,2, 1.0f, posStride, featureStride);

      if(stone == pla || stone == opp) {
        int libs = board.getNumLiberties(loc);
        if(libs == 1) setRowBin(rowBin,pos,3, 1.0f, posStride, featureStride);
        else if(libs == 2) setRowBin(rowBin,pos,4, 1.0f, posStride, featureStride);
        else if(libs == 3) setRowBin(rowBin,pos,5, 1.0f, posStride, featureStride);
      }
    }
  }

  //Feature 6 - ko-ban locations, including possibly superko.
  if(hist.encorePhase == 0) {
    if(board.ko_loc != Board::NULL_LOC) {
      int pos = NNPos::locToPos(board.ko_loc,xSize,nnXLen,nnYLen);
      setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        if(hist.superKoBanned[loc] && loc != board.ko_loc) {
          int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
          setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
        }
      }
    }
  }
  else {
    //Feature 6,7,8 - in the encore, no-second-ko-capture locations, encore ko prohibitions where we have to pass for ko
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.superKoBanned[loc])
          setRowBin(rowBin,pos,6, 1.0f, posStride, featureStride);
        if(hist.koRecapBlocked[loc])
          setRowBin(rowBin,pos,7, 1.0f, posStride, featureStride);
      }
    }
  }


  //Features 18,19 - current territory, not counting group tax
  Color area[Board::MAX_ARR_SIZE];
  bool hasAreaFeature = false;
  int groupTaxAdjustmentForPla = 0;
  if(hist.rules.scoringRule == Rules::SCORING_AREA && hist.rules.taxRule == Rules::TAX_NONE) {
    hasAreaFeature = true;
    bool nonPassAliveStones = true;
    bool safeBigTerritories = true;
    bool unsafeBigTerritories = true;
    board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,nnInputParams.getSuicideLegalForPassAlive(hist));
  }
  else {
    bool keepTerritories = false;
    bool keepStones = false;
    int whiteMinusBlackIndependentLifeRegionCount = 0;
    if(hist.rules.scoringRule == Rules::SCORING_AREA && (hist.rules.taxRule == Rules::TAX_SEKI || hist.rules.taxRule == Rules::TAX_ALL)) {
      hasAreaFeature = true;
      keepTerritories = false;
      keepStones = true;
    }
    else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && hist.rules.taxRule == Rules::TAX_NONE) {
      //Territory scoring omits feature until we reach the stage where scoring matters
      if(hist.encorePhase >= 2) {
        hasAreaFeature = true;
        keepTerritories = true;
        keepStones = false;
      }
    }
    else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && (hist.rules.taxRule == Rules::TAX_SEKI || hist.rules.taxRule == Rules::TAX_ALL)) {
      //Territory scoring omits feature until we reach the stage where scoring matters
      if(hist.encorePhase >= 2) {
        hasAreaFeature = true;
        keepTerritories = false;
        keepStones = false;
      }
    }
    else {
      ASSERT_UNREACHABLE;
    }

    if(hasAreaFeature) {
      board.calculateIndependentLifeArea(
        area,
        whiteMinusBlackIndependentLifeRegionCount,
        keepTerritories,
        keepStones,
        nnInputParams.getExcludeTerritoryAdjacentToAtari(hist),
        nnInputParams.getSuicideLegalForPassAlive(hist)
      );
      if(hist.rules.taxRule == Rules::TAX_ALL)
        groupTaxAdjustmentForPla = pla == P_WHITE ? -2 * whiteMinusBlackIndependentLifeRegionCount : 2 * whiteMinusBlackIndependentLifeRegionCount;
    }
  }

  bool finalPhaseAndGameEndWouldNotBeWin = false;
  if(hasAreaFeature) {
    int boardScoreForPla = groupTaxAdjustmentForPla;
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(area[loc] == pla) {
          setRowBin(rowBin,pos,18, 1.0f, posStride, featureStride);
          boardScoreForPla += 1;
        }
        else if(area[loc] == opp) {
          setRowBin(rowBin,pos,19, 1.0f, posStride, featureStride);
          boardScoreForPla -= 1;
        }
        else {
          if(hist.rules.scoringRule == Rules::SCORING_TERRITORY) {
            //Also we must be in the second encore phase, based on the logic above.
            if(board.colors[loc] == pla && hist.secondEncoreStartColors[loc] == pla) {
              setRowBin(rowBin,pos,18, 1.0f, posStride, featureStride);
              boardScoreForPla += 1;
            }
            else if(board.colors[loc] == opp && hist.secondEncoreStartColors[loc] == opp) {
              setRowBin(rowBin,pos,19, 1.0f, posStride, featureStride);
              boardScoreForPla -= 1;
            }
          }
        }
      }
    }
    float selfKomi = hist.currentSelfKomi(pla, nnInputParams.drawEquivalentWinsForWhite);
    float finalScorePla = (float)boardScoreForPla + selfKomi;
    // If the game ended here, and was scored instantly, it would be a loss or a draw?
    if(finalScorePla <= 0.0)
      finalPhaseAndGameEndWouldNotBeWin = true;
  }

  //Hide history from the net if a pass would end things and we're behaving as if a pass won't.
  //Or if the game is in fact over right now!
  int maxTurnsOfHistoryToInclude = 5;
  bool suppressPassWouldEndPhase = false;
  if(hist.passWouldEndGame(board,nextPlayer) && (
       //At the root, if assuming passing doesn't end the game, and it would, then need to mask that out.
       nnInputParams.conservativePassAndIsRoot ||
       //Deeper in the tree, we might not assume passes end the game in a friendly pass setting.
       hist.shouldSuppressEndGameFromFriendlyPass(board,nextPlayer) ||
       //Passing hacks suppress the net to end the game when losing if it thinks a premature pass will lose by less.
       (nnInputParams.enablePassingHacks && finalPhaseAndGameEndWouldNotBeWin)
     )
  ) {
    maxTurnsOfHistoryToInclude = 0;
    suppressPassWouldEndPhase = true;
  }
  else if(hist.isGameFinished || hist.isPastNormalPhaseEnd) {
    // Include one of the passes, at the end of that sequence
    maxTurnsOfHistoryToInclude = 1;
  }
  maxTurnsOfHistoryToInclude = std::min(maxTurnsOfHistoryToInclude, nnInputParams.maxHistory);

  int numTurnsOfHistoryIncluded = 0;

  //Features 9,10,11,12,13
  if(maxTurnsOfHistoryToInclude > 0) {
    const vector<Move>& moveHistory = hist.moveHistory;
    size_t moveHistoryLen = moveHistory.size();
    assert(moveHistoryLen >= hist.numApproxValidTurnsThisPhase);

    //Effectively wipe history as we change phase by also capping it
    int amountOfHistoryToTryToUse = std::min(maxTurnsOfHistoryToInclude, hist.numApproxValidTurnsThisPhase);

    if(amountOfHistoryToTryToUse >= 1 && moveHistory[moveHistoryLen-1].pla == opp) {
      Loc prev1Loc = moveHistory[moveHistoryLen-1].loc;
      numTurnsOfHistoryIncluded = 1;
      if(prev1Loc == Board::PASS_LOC)
        rowGlobal[0] = 1.0;
      else if(prev1Loc != Board::NULL_LOC) {
        int pos = NNPos::locToPos(prev1Loc,xSize,nnXLen,nnYLen);
        setRowBin(rowBin,pos,9, 1.0f, posStride, featureStride);
      }
      if(amountOfHistoryToTryToUse >= 2 && moveHistory[moveHistoryLen-2].pla == pla) {
        Loc prev2Loc = moveHistory[moveHistoryLen-2].loc;
        numTurnsOfHistoryIncluded = 2;
        if(prev2Loc == Board::PASS_LOC)
          rowGlobal[1] = 1.0;
        else if(prev2Loc != Board::NULL_LOC) {
          int pos = NNPos::locToPos(prev2Loc,xSize,nnXLen,nnYLen);
          setRowBin(rowBin,pos,10, 1.0f, posStride, featureStride);
        }
        if(amountOfHistoryToTryToUse >= 3 && moveHistory[moveHistoryLen-3].pla == opp) {
          Loc prev3Loc = moveHistory[moveHistoryLen-3].loc;
          numTurnsOfHistoryIncluded = 3;
          if(prev3Loc == Board::PASS_LOC)
            rowGlobal[2] = 1.0;
          else if(prev3Loc != Board::NULL_LOC) {
            int pos = NNPos::locToPos(prev3Loc,xSize,nnXLen,nnYLen);
            setRowBin(rowBin,pos,11, 1.0f, posStride, featureStride);
          }
          if(amountOfHistoryToTryToUse >= 4 && moveHistory[moveHistoryLen-4].pla == pla) {
            Loc prev4Loc = moveHistory[moveHistoryLen-4].loc;
            numTurnsOfHistoryIncluded = 4;
            if(prev4Loc == Board::PASS_LOC)
              rowGlobal[3] = 1.0;
            else if(prev4Loc != Board::NULL_LOC) {
              int pos = NNPos::locToPos(prev4Loc,xSize,nnXLen,nnYLen);
              setRowBin(rowBin,pos,12, 1.0f, posStride, featureStride);
            }
            if(amountOfHistoryToTryToUse >= 5 && moveHistory[moveHistoryLen-5].pla == opp) {
              Loc prev5Loc = moveHistory[moveHistoryLen-5].loc;
              numTurnsOfHistoryIncluded = 5;
              if(prev5Loc == Board::PASS_LOC)
                rowGlobal[4] = 1.0;
              else if(prev5Loc != Board::NULL_LOC) {
                int pos = NNPos::locToPos(prev5Loc,xSize,nnXLen,nnYLen);
                setRowBin(rowBin,pos,13, 1.0f, posStride, featureStride);
              }
            }
          }
        }
      }
    }
  }

  //Ladder features 14,15,16,17
  auto addLadderFeature = [&board,xSize,nnXLen,nnYLen,posStride,featureStride,rowBin,opp](Loc loc, int pos, const vector<Loc>& workingMoves) noexcept {
    assert(board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBin(rowBin,pos,14, 1.0f, posStride, featureStride);
    if(board.colors[loc] == opp && board.getNumLiberties(loc) > 1) {
      for(size_t j = 0; j < workingMoves.size(); j++) {
        int workingPos = NNPos::locToPos(workingMoves[j],xSize,nnXLen,nnYLen);
        setRowBin(rowBin,workingPos,17, 1.0f, posStride, featureStride);
      }
    }
  };

  iterLadders(board, nnXLen, addLadderFeature);

  const Board& prevBoard = (numTurnsOfHistoryIncluded < 1) ? board : hist.getRecentBoard(1);
  auto addPrevLadderFeature = [&prevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves) noexcept {
    (void)workingMoves;
    (void)loc;
    assert(prevBoard.colors[loc] == P_BLACK || prevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBin(rowBin,pos,15, 1.0f, posStride, featureStride);
  };
  iterLadders(prevBoard, nnXLen, addPrevLadderFeature);

  const Board& prevPrevBoard = (numTurnsOfHistoryIncluded < 2) ? prevBoard : hist.getRecentBoard(2);
  auto addPrevPrevLadderFeature = [&prevPrevBoard,posStride,featureStride,rowBin](Loc loc, int pos, const vector<Loc>& workingMoves) noexcept {
    (void)workingMoves;
    (void)loc;
    assert(prevPrevBoard.colors[loc] == P_BLACK || prevPrevBoard.colors[loc] == P_WHITE);
    assert(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
    setRowBin(rowBin,pos,16, 1.0f, posStride, featureStride);
  };
  iterLadders(prevPrevBoard, nnXLen, addPrevPrevLadderFeature);

  //Features 20, 21 - second encore starting stones
  if(hist.encorePhase >= 2) {
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        Loc loc = Location::getLoc(x,y,xSize);
        int pos = NNPos::locToPos(loc,xSize,nnXLen,nnYLen);
        if(hist.secondEncoreStartColors[loc] == pla)
          setRowBin(rowBin,pos,20, 1.0f, posStride, featureStride);
        else if(hist.secondEncoreStartColors[loc] == opp)
          setRowBin(rowBin,pos,21, 1.0f, posStride, featureStride);
      }
    }
  }


  //Global features.
  //The first 5 of them were set already above to flag which of the past 5 moves were passes.

  //Komi and any score adjustments
  float selfKomi = hist.currentSelfKomi(nextPlayer,nnInputParams.drawEquivalentWinsForWhite);
  float bArea = (float)(xSize * ySize);
  //Bound komi just in case
  if(selfKomi > bArea+NNPos::KOMI_CLIP_RADIUS)
    selfKomi = bArea+NNPos::KOMI_CLIP_RADIUS;
  if(selfKomi < -bArea-NNPos::KOMI_CLIP_RADIUS)
    selfKomi = -bArea-NNPos::KOMI_CLIP_RADIUS;
  rowGlobal[5] = selfKomi/20.0f;

  //Ko rule
  if(hist.rules.koRule == Rules::KO_SIMPLE) {}
  else if(hist.rules.koRule == Rules::KO_POSITIONAL || hist.rules.koRule == Rules::KO_SPIGHT) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = 0.5f;
  }
  else if(hist.rules.koRule == Rules::KO_SITUATIONAL) {
    rowGlobal[6] = 1.0f;
    rowGlobal[7] = -0.5f;
  }
  else
    ASSERT_UNREACHABLE;

  //Suicide
  if(hist.rules.multiStoneSuicideLegal)
    rowGlobal[8] = 1.0f;

  //Scoring
  if(hist.rules.scoringRule == Rules::SCORING_AREA) {}
  else if(hist.rules.scoringRule == Rules::SCORING_TERRITORY)
    rowGlobal[9] = 1.0f;
  else
    ASSERT_UNREACHABLE;
  //Tax
  if(hist.rules.taxRule == Rules::TAX_NONE) {}
  else if(hist.rules.taxRule == Rules::TAX_SEKI)
    rowGlobal[10] = 1.0f;
  else if(hist.rules.taxRule == Rules::TAX_ALL) {
    rowGlobal[10] = 1.0f;
    rowGlobal[11] = 1.0f;
  }
  else
    ASSERT_UNREACHABLE;

  //Encore phase
  if(hist.encorePhase > 0)
    rowGlobal[12] = 1.0f;
  if(hist.encorePhase > 1)
    rowGlobal[13] = 1.0f;

  //Does a pass end the current phase given the ruleset and history?
  bool passWouldEndPhase = suppressPassWouldEndPhase ? false : hist.passWouldEndPhase(board,nextPlayer);
  rowGlobal[14] = passWouldEndPhase ? 1.0f : 0.0f;

  //Used for handicap play
  //Parameter 15 is used because there's actually a discontinuity in how training behavior works when this is
  //nonzero, no matter how slightly.
  if(nnInputParams.playoutDoublingAdvantage != 0) {
    rowGlobal[15] = 1.0;
    rowGlobal[16] = (float)(0.5 * nnInputParams.playoutDoublingAdvantage);
  }

  //Button
  if(hist.hasButton)
    rowGlobal[17] = 1.0;

  //Provide parity information about the board size and komi
  //This comes from the following observation:
  //From white's perspective:
  //Komi = 0.0 - Draw possible
  //Komi = 0.5 - Win the games we would have drawn with komi 0.0
  //Komi = 1.0 - Usually no difference from komi 0.5
  //Komi = 1.5 - Usually no difference from komi 0.5
  //Komi = 2.0 - Draw possible
  //If we were to assign an "effective goodness" to these komis in order it would look like
  //0 1 1 1 2 3 3 3 4 5 5 5 6 ...
  //since when away from the right parity, increasing the komi doesn't help us except in cases of seki with odd numbers of dame.
  //If we were to add 0.5 times a vector like:
  //0 -1 0 1 0 -1 0 1 0 -1 0 ...
  //Then this would become a linear function and hopefully easier for a neural net to learn.
  //We expect that this is hard for a neural net to learn since it depends on the parity of the board size
  //and is very "xor"like.
  //So we provide it as an input.
  //Since we are using a model where games are jittered by 0.5 (see BoardHistory::whiteKomiAdjustmentForDraws)
  //in theory right thing to first order to provide should be a triangular wave with a period of 2 komi points:
  //  ../\........
  //  ./..\.......
  //  /....\..../.
  //  ......\../..
  //  .......\/...
  //The upsloping part of the wave is centered around the komi value where you could draw
  //since komi is extra valuable when it turns losses into draws into wins, peaking at the komi value where you could draw + 0.5.
  //It's downsloping around the komi value where you can't draw, since the marginal komi there is nearly useless, not causing you to win
  //more games except in case of odd-dame seki.

  if(hist.rules.scoringRule == Rules::SCORING_AREA || hist.encorePhase >= 2) {
    bool boardAreaIsEven = (xSize*ySize) % 2 == 0;

    //What is the parity of the komi values that can produce jigos?
    bool drawableKomisAreEven = boardAreaIsEven;

    //Find the difference between the komi viewed from our perspective and the nearest drawable komi below it.
    float komiFloor;
    if(drawableKomisAreEven)
      komiFloor = floor(selfKomi / 2.0f) * 2.0f;
    else
      komiFloor = floor((selfKomi-1.0f) / 2.0f) * 2.0f + 1.0f;

    //Cap just in case we have floating point weirdness
    float delta = selfKomi - komiFloor;
    assert(delta >= -0.0001f);
    assert(delta <= 2.0001f);
    if(delta < 0.0f)
      delta = 0.0f;
    if(delta > 2.0f)
      delta = 2.0f;

    //Create the triangle wave based on the difference
    float wave;
    if(delta < 0.5f)
      wave = delta;
    else if(delta < 1.5f)
      wave = 1.0f-delta;
    else
      wave = delta-2.0f;

    //NOTE: If ever changing which feature this is, must also update index in model.py where we multiply it into the scorebelief parity vector
    rowGlobal[18] = wave;
  }

}

//===========================================================================================
//INPUTSVERSION 8 (Enclosure fork)
//===========================================================================================

//The spatial planes of the inputs versions 8 to 11: V9 has V8's, V10 the game's extended planes too, V11 V10's.
static int gameSpatialFeatures(int inputsVersion) {
  return inputsVersion >= 10 ? NNInputs::NUM_FEATURES_SPATIAL_V10 : NNInputs::NUM_FEATURES_SPATIAL_V8;
}

//KataGame fork: the game's inputs (GameEncoding::spatial_inputs and global_inputs) after the front's on-board plane.
//See NUM_FEATURES_SPATIAL_V8.
//Fills rowGlobal and calls setEntry(i) for each spatial entry i that is 1 (the spatial row must start all 0); shared
//by fillRowV8 (floats) and fillRowV8Packed (bits), so the two rows cannot differ. symmetry (owner, 2026-10-06): the
//row of board's image under that allowed symmetry, which the game writes from board (InputOptions::symmetry); the
//on-board plane of a square board does not change. inputsVersion 9 (2026-10-13): rowGlobal also holds the two
//playout doubling advantage inputs (NUM_FEATURES_GLOBAL_V9). 10 (2026-10-26): the game's extended inputs too, its
//planes after the game's planes and its global inputs after V9's (NUM_FEATURES_SPATIAL_V10). 11 (2026-10-27): the
//history of the game's turns, and its turn inputs after V10's global inputs (NUM_FEATURES_GLOBAL_V11).
template<typename SetEntry>
static void fillRowV8Entries(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowGlobal, int symmetry, int inputsVersion, SetEntry setEntry
) {
  //The network's board is the game's largest board (NNPos).
  assert(nnXLen == NNPos::MAX_BOARD_LEN && nnYLen == NNPos::MAX_BOARD_LEN);
  assert(board.x_size <= nnXLen && board.y_size <= nnYLen);
  assert(SymmetryHelpers::isAllowed(symmetry));
  assert(!SymmetryHelpers::isTranspose(symmetry) || board.x_size == board.y_size);
  assert(inputsVersion >= 8 && inputsVersion <= 11);
  //The inputs are the position's, which gives the player to move.
  assert(nextPlayer == board.nextPla());
  (void)nextPlayer;
  const int numPos = nnXLen * nnYLen;
  using NNInputs::NUM_FEATURES_GLOBAL_V8;
  const bool playoutInputs = inputsVersion >= 9;
  const bool extended = inputsVersion >= 10;
  const bool turnHistory = inputsVersion >= 11;
  const int numGlobal =
    turnHistory ? NNInputs::NUM_FEATURES_GLOBAL_V11 :
    extended ? NNInputs::NUM_FEATURES_GLOBAL_V10 : playoutInputs ? NNInputs::NUM_FEATURES_GLOBAL_V9 : NUM_FEATURES_GLOBAL_V8;
  std::fill(rowGlobal, rowGlobal + numGlobal, 0.0f);

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = gameSpatialFeatures(inputsVersion);
  }
  else {
    featureStride = numPos;
    posStride = 1;
  }
  //Every spatial entry set is 1.
  auto setOne = [&](int pos, int feature) {
    setEntry(pos * posStride + feature * featureStride);
  };
  //Feature 0 - on board (the front's). A flip keeps the board, and a game allows a transpose only if its boards are
  //square, so the plane is the same at every allowed symmetry.
  for(int y = 0; y < board.y_size; y++)
    for(int x = 0; x < board.x_size; x++)
      setOne(NNPos::xyToPos(x, y, nnXLen), 0);

  //The game's planes, after the on-board plane, at the image's points.
  GameInterface::InputOptions options;
  options.symmetry = symmetry;
  //maxHistory <= 0 asks for no history (KataGo's ZOBRIST_ZERO_HISTORY case): the game leaves out its history planes.
  options.history = nnInputParams.maxHistory > 0;
  options.komi = hist.rules.komi;
  options.extended = extended;
  options.turnHistory = turnHistory;
  GameEncoding::spatial_inputs(board.pos, options, [&](int x, int y, int plane) {
    setOne(NNPos::xyToPos(x, y, nnXLen), NNInputs::V8_GAME_PLANES_START + plane);
  });
  if(!extended)
    GameEncoding::global_inputs(board.pos, options, rowGlobal);
  else {
    //The game writes its global inputs, then its extended ones, which V10 puts after V9's, then in V11 its turn inputs.
    constexpr int numExtended = GameInterface::extendedGlobalInputs<GameEncoding>();
    constexpr int numTurn = GameInterface::turnGlobalInputs<GameEncoding>();
    float game[NUM_FEATURES_GLOBAL_V8 + numExtended + numTurn];
    GameEncoding::global_inputs(board.pos, options, game);
    const int numAfter = numExtended + (turnHistory ? numTurn : 0);
    std::copy(game, game + NUM_FEATURES_GLOBAL_V8, rowGlobal);
    std::copy(game + NUM_FEATURES_GLOBAL_V8, game + NUM_FEATURES_GLOBAL_V8 + numAfter,
              rowGlobal + NNInputs::NUM_FEATURES_GLOBAL_V9);
  }
  //V9 (2026-10-13): KataGo's playout doubling advantage, as V7's inputs 15 and 16. The first is 1 whenever the
  //advantage is nonzero, because "there's actually a discontinuity in how training behavior works when this is
  //nonzero, no matter how slightly" (KataGo's comment). The search and the training rows give the advantage from the
  //player to move's side.
  if(playoutInputs && nnInputParams.playoutDoublingAdvantage != 0) {
    rowGlobal[NUM_FEATURES_GLOBAL_V8] = 1.0f;
    rowGlobal[NUM_FEATURES_GLOBAL_V8 + 1] = (float)(0.5 * nnInputParams.playoutDoublingAdvantage);
  }
}

static void fillRowV8Floats(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal, std::vector<int>* setEntries,
  int inputsVersion
) {
  //The spatial planes are sparse 0/1: with setEntries, clear only the entries the last fill set (nninputs.h).
  if(setEntries != NULL) {
    for(int i : *setEntries)
      rowBin[i] = 0.0f;
    setEntries->clear();
  }
  else
    std::fill(rowBin, rowBin + gameSpatialFeatures(inputsVersion) * nnXLen * nnYLen, 0.0f);
  //Each entry set is recorded for the next fill's clearing.
  fillRowV8Entries(board, hist, nextPlayer, nnInputParams, nnXLen, nnYLen, useNHWC, rowGlobal, 0, inputsVersion, [&](int i) {
    rowBin[i] = 1.0f;
    if(setEntries != NULL)
      setEntries->push_back(i);
  });
}

static void fillRowV8Bits(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, uint32_t* rowBits, float* rowGlobal, int symmetry, int inputsVersion
) {
  std::fill(rowBits, rowBits + (inputsVersion >= 10 ? NNInputs::NUM_SPATIAL_WORDS_V10 : NNInputs::NUM_SPATIAL_WORDS_V8), 0u);
  fillRowV8Entries(board, hist, nextPlayer, nnInputParams, nnXLen, nnYLen, useNHWC, rowGlobal, symmetry, inputsVersion, [&](int i) {
    rowBits[i >> 5] |= 1u << (i & 31);
  });
}

void NNInputs::fillRowV8(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal, std::vector<int>* setEntries
) {
  fillRowV8Floats(board, hist, nextPlayer, nnInputParams, nnXLen, nnYLen, useNHWC, rowBin, rowGlobal, setEntries, 8);
}

void NNInputs::fillRowV8Packed(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, uint32_t* rowBits, float* rowGlobal, int symmetry
) {
  fillRowV8Bits(board, hist, nextPlayer, nnInputParams, nnXLen, nnYLen, useNHWC, rowBits, rowGlobal, symmetry, 8);
}

void NNInputs::fillRowV9(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal, std::vector<int>* setEntries
) {
  fillRowV8Floats(board, hist, nextPlayer, nnInputParams, nnXLen, nnYLen, useNHWC, rowBin, rowGlobal, setEntries, 9);
}

void NNInputs::fillRowV9Packed(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, uint32_t* rowBits, float* rowGlobal, int symmetry
) {
  fillRowV8Bits(board, hist, nextPlayer, nnInputParams, nnXLen, nnYLen, useNHWC, rowBits, rowGlobal, symmetry, 9);
}

void NNInputs::fillRowV10(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal, std::vector<int>* setEntries
) {
  fillRowV8Floats(board, hist, nextPlayer, nnInputParams, nnXLen, nnYLen, useNHWC, rowBin, rowGlobal, setEntries, 10);
}

void NNInputs::fillRowV10Packed(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, uint32_t* rowBits, float* rowGlobal, int symmetry
) {
  fillRowV8Bits(board, hist, nextPlayer, nnInputParams, nnXLen, nnYLen, useNHWC, rowBits, rowGlobal, symmetry, 10);
}

void NNInputs::fillRowV11(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal, std::vector<int>* setEntries
) {
  fillRowV8Floats(board, hist, nextPlayer, nnInputParams, nnXLen, nnYLen, useNHWC, rowBin, rowGlobal, setEntries, 11);
}

void NNInputs::fillRowV11Packed(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, uint32_t* rowBits, float* rowGlobal, int symmetry
) {
  fillRowV8Bits(board, hist, nextPlayer, nnInputParams, nnXLen, nnYLen, useNHWC, rowBits, rowGlobal, symmetry, 11);
}

void NNInputs::unpackSpatialBits(const uint32_t* rowBits, float* out, int n) {
  for(int i = 0; i < n; i++)
    out[i] = (float)((rowBits[i >> 5] >> (i & 31)) & 1u);
}
