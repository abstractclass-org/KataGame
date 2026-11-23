// KataGame fork (abscls-shfz.6): the compact policy's rank mask (neuralnet/nninputs.h, PolicyRankMask) at more than 64
// channels a point, and the build's own compact policy (NNOutput). It names no game, so it builds against any game's
// build (tests/enclosure/build.sh BUILD_DIR, with TESTS=testpolicymask):
//
//   testpolicymask
//
// 1. The mask alone, for 1, 6, 48, 64, 65, 128 and 255 channels a point (Tak's 6 x 6 has 255), on 36 points: random
//    sets of legal positions, the empty set, the full set, and sets of the channels next to the words' edges. A
//    position's slot (PolicyRankMask::slot) is its rank in the ascending list of legal positions, and -1 when it is not
//    legal. The point's start is the number of legal positions before the point, as NNOutput keeps it.
// 2. The build's own policy (the game's POLICY_CHANNELS): random dense policies through setPolicyFromDense and
//    setPolicyFromLegal, read back through policy() at every position and through toDense.
//
// Prints each failure and the count. Exits 1 on a failure.

#include <cstdio>
#include <vector>

#include "../../core/rand.h"
#include "../../neuralnet/nninputs.h"

using namespace std;

static int failures = 0;

static void fail(const char* what, int channels, int p, int got, int expected) {
  if(failures < 20)
    printf("FAIL %s: %d channels, position %d: %d, expected %d\n", what, channels, p, got, expected);
  failures++;
}

//The legal positions (each point's channels, no pass) through PolicyRankMask<CHANNELS> on `points` points, each
//position's slot against its rank.
template<int CHANNELS>
static void checkSlots(const vector<char>& legal, int points) {
  using Mask = PolicyRankMask<CHANNELS>;
  vector<uint64_t> mask(points * Mask::WORDS, 0);
  vector<uint16_t> start(points);
  int count = 0;
  for(int point = 0; point < points; point++) {
    start[point] = (uint16_t)count;
    for(int channel = 0; channel < CHANNELS; channel++)
      if(legal[point * CHANNELS + channel]) {
        Mask::mark(mask.data(), point, channel);
        count++;
      }
  }
  int rank = 0;
  for(int p = 0; p < points * CHANNELS; p++) {
    const int point = p / CHANNELS, channel = p - point * CHANNELS;
    const int got = Mask::slot(mask.data(), start.data(), point, channel);
    const int expected = legal[p] ? rank : -1;
    if(got != expected)
      fail("slot", CHANNELS, p, got, expected);
    rank += legal[p] ? 1 : 0;
  }
}

template<int CHANNELS>
static void checkChannels(Rand& rand) {
  const int points = 36;
  const int n = points * CHANNELS;
  vector<char> legal(n);
  for(double density: {0.0, 0.02, 0.3, 0.5, 0.97, 1.0}) {
    for(int trial = 0; trial < 20; trial++) {
      for(int p = 0; p < n; p++)
        legal[p] = rand.nextDouble() < density ? 1 : 0;
      checkSlots<CHANNELS>(legal, points);
    }
  }
  //The channels next to each word's edge, alone and together, at each point.
  for(int edge = 0; edge < CHANNELS; edge += 64) {
    for(int c: {edge - 1, edge, edge + 63}) {
      if(c < 0 || c >= CHANNELS)
        continue;
      for(int p = 0; p < n; p++)
        legal[p] = (p % CHANNELS == c) ? 1 : 0;
      checkSlots<CHANNELS>(legal, points);
      for(int p = 0; p < n; p++)
        legal[p] = (p % CHANNELS == c || rand.nextBool(0.5)) ? 1 : 0;
      checkSlots<CHANNELS>(legal, points);
    }
  }
}

//The build's compact policy: a random dense policy (each position legal with the probability density, pass too) kept
//by setPolicyFromDense and by setPolicyFromLegal, read back at every position and as a dense array.
static void checkOutput(Rand& rand, double density) {
  const int size = NNPos::MAX_NN_POLICY_SIZE;
  vector<float> dense(size);
  vector<uint16_t> pos;
  vector<float> probs;
  vector<char> legal;
  for(int p = 0; p < size; p++) {
    dense[p] = rand.nextDouble() < density ? (float)rand.nextDouble() : -1.0f;
    //setPolicyFromLegal's list holds every legal position and some illegal ones, which it skips.
    if(dense[p] >= 0 || rand.nextBool(0.1)) {
      pos.push_back((uint16_t)p);
      probs.push_back(dense[p] >= 0 ? dense[p] : 0.5f);
      legal.push_back(dense[p] >= 0 ? 1 : 0);
    }
  }
  NNOutput fromDense;
  fromDense.setPolicyFromDense(dense.data());
  NNOutput fromLegal;
  fromLegal.setPolicyFromLegal(pos.data(), probs.data(), legal.data(), (int)pos.size());
  vector<float> back(size);
  for(const NNOutput* out: {&fromDense, &fromLegal}) {
    const PolicyView view = out->policy();
    for(int p = 0; p < size; p++)
      if(view[p] != dense[p])
        fail(out == &fromDense ? "setPolicyFromDense" : "setPolicyFromLegal", NNPos::POLICY_CHANNELS, p,
             (int)(view[p] * 1e6f), (int)(dense[p] * 1e6f));
    view.toDense(back.data());
    for(int p = 0; p < size; p++)
      if(back[p] != dense[p])
        fail("toDense", NNPos::POLICY_CHANNELS, p, (int)(back[p] * 1e6f), (int)(dense[p] * 1e6f));
  }
}

int main() {
  static_assert(PolicyRankMask<1>::WORDS == 1 && PolicyRankMask<64>::WORDS == 1, "up to 64 channels: one word");
  static_assert(PolicyRankMask<65>::WORDS == 2 && PolicyRankMask<255>::WORDS == 4, "64 channels a word");
  Rand rand(20261123ULL);
  checkChannels<1>(rand);
  checkChannels<6>(rand);
  checkChannels<48>(rand);
  checkChannels<64>(rand);
  checkChannels<65>(rand);
  checkChannels<128>(rand);
  checkChannels<255>(rand);
  for(double density: {0.0, 0.01, 0.3, 1.0})
    for(int trial = 0; trial < 5; trial++)
      checkOutput(rand, density);
  printf("testpolicymask: %d channels a point in this build (%d words), %d failures\n", NNPos::POLICY_CHANNELS,
         PolicyRankMask<NNPos::POLICY_CHANNELS>::WORDS, failures);
  return failures == 0 ? 0 : 1;
}
