// Enclosure fork (owner, 2026-10-04): RGSC's pieces (program/rgsc.h) on hand-worked cases.
//
//   testrgsc SCRATCH_DIR
//
// Eq. 2 on a short game; the position key's order independence; the buffer's fill, replace-the-lowest, skip-known
// and EMA rules; the floor for tree candidates; sampling frequencies against the paper's R^(1/tau) / sum; the
// start probability; and the save/load round trip (SCRATCH_DIR/rgsc_buffer.txt).

#include <cmath>
#include <cstdio>
#include <map>

#include "../../program/rgsc.h"

using namespace std;

static int failures = 0;
static void check(bool ok, const string& what) {
  if(!ok) {
    fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  }
}
static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

int main(int argc, char** argv) {
  if(argc != 2) {
    fprintf(stderr, "usage: testrgsc SCRATCH_DIR\n");
    return 2;
  }
  Board::initHash();

  // Eq. 2: values (white's view) 0.5, -0.2, 0.9; white won (z = 1). Squared errors 0.25, 1.44, 0.01.
  {
    vector<float> r = Rgsc::regretsOfGame({0.5f, -0.2f, 0.9f}, 1.0);
    check(r.size() == 3, "regrets: one per turn");
    check(near(r[2], 0.01, 1e-6), "regret at the last turn: its own error");
    check(near(r[1], (1.44 + 0.01) / 2, 1e-6), "regret at turn 1: mean of turns 1-2");
    check(near(r[0], (0.25 + 1.44 + 0.01) / 3, 1e-6), "regret at turn 0: mean of all");
    vector<float> d = Rgsc::regretsOfGame({0.0f, 0.0f}, 0.0);
    check(d[0] == 0.0f && d[1] == 0.0f, "a no-result game with zero values has zero regret");
  }

  // Keys: the same moves in another order give the same key; another player or move does not.
  {
    vector<Move> a = {Move(100, P_BLACK), Move(200, P_BLACK), Move(300, P_WHITE)};
    vector<Move> b = {Move(200, P_BLACK), Move(100, P_BLACK), Move(300, P_WHITE)};
    vector<Move> c = {Move(100, P_WHITE), Move(200, P_BLACK), Move(300, P_WHITE)};
    check(Rgsc::keyOf(a, 3) == Rgsc::keyOf(b, 3), "key ignores the order");
    check(Rgsc::keyOf(a, 3) != Rgsc::keyOf(c, 3), "key sees the player");
    check(Rgsc::keyOf(a, 2) != Rgsc::keyOf(a, 3), "key sees the prefix length");
  }

  Rgsc::Settings settings;
  settings.enabled = true;
  settings.bufferFile = string(argv[1]) + "/rgsc_buffer.txt";
  settings.bufferSize = 3;
  settings.startProb = 1.0;
  settings.temperature = 0.1;
  settings.emaAlpha = 0.5;
  auto movesOf = [](int i) { return vector<Move>{Move((Loc)(100 + i), P_BLACK), Move((Loc)(200 + i), P_BLACK)}; };
  auto keyOfI = [&](int i) { vector<Move> m = movesOf(i); return Rgsc::keyOf(m, m.size()); };

  // Fill, replace the lowest only when beaten, skip a known state, EMA.
  Rgsc::Buffer buffer(settings);
  buffer.offer(keyOfI(1), 0.5, movesOf(1));
  buffer.offer(keyOfI(2), 0.3, movesOf(2));
  check(buffer.floorRegret() == 0.0, "floor is 0 while the buffer has room");
  buffer.offer(keyOfI(3), 0.8, movesOf(3));
  check(buffer.size() == 3, "fills to its size");
  check(near(buffer.floorRegret(), 0.3, 1e-12), "floor is the lowest once full");
  buffer.offer(keyOfI(4), 0.2, movesOf(4));
  check(!buffer.contains(keyOfI(4)) && buffer.contains(keyOfI(2)), "a lower candidate does not enter a full buffer");
  buffer.offer(keyOfI(5), 0.6, movesOf(5));
  check(buffer.contains(keyOfI(5)) && !buffer.contains(keyOfI(2)), "a higher candidate replaces the lowest");
  buffer.offer(keyOfI(1), 9.0, movesOf(1));
  check(buffer.size() == 3 && near(buffer.floorRegret(), 0.5, 1e-12), "a known state is skipped, not updated");
  check(buffer.updateAfterRestart(keyOfI(3), 0.2), "EMA update finds the state");
  check(!buffer.updateAfterRestart(keyOfI(2), 0.2), "EMA update reports a state that left");
  // Now: 1 -> 0.5, 3 -> 0.5 * 0.8 + 0.5 * 0.2 = 0.5, 5 -> 0.6.

  // Sampling: P ~ R^(1/tau) = R^10: weights 0.5^10, 0.5^10, 0.6^10.
  {
    Rand rand("testrgsc");
    map<uint64_t, int> counts;
    const int n = 200000;
    for(int i = 0; i < n; i++) {
      Rgsc::Entry e;
      check(buffer.maybeSample(rand, e), "start probability 1 always samples");
      counts[e.key]++;
    }
    double w5 = pow(0.6, 10), w05 = pow(0.5, 10), total = w5 + 2 * w05;
    double p5 = (double)counts[keyOfI(5)] / n, p1 = (double)counts[keyOfI(1)] / n;
    check(near(p5, w5 / total, 0.01), "sampling frequency of the 0.6 state: " + to_string(p5) + " vs " + to_string(w5 / total));
    check(near(p1, w05 / total, 0.01), "sampling frequency of a 0.5 state: " + to_string(p1) + " vs " + to_string(w05 / total));
    Rgsc::Settings never = settings;
    never.startProb = 0.0;
    Rgsc::Buffer other(never);
    other.offer(keyOfI(1), 1.0, movesOf(1));
    Rgsc::Entry e;
    check(!other.maybeSample(rand, e), "start probability 0 never samples");
  }

  // Warm-up count and save/load.
  {
    check(buffer.countFinishedGame() == 0 && buffer.countFinishedGame() == 1, "finished games count up");
    buffer.save();
    Rgsc::Buffer loaded(settings);
    loaded.load();
    check(loaded.size() == 3, "load gets every state");
    check(loaded.contains(keyOfI(1)) && loaded.contains(keyOfI(3)) && loaded.contains(keyOfI(5)), "load gets the same states");
    check(near(loaded.floorRegret(), 0.5, 1e-6), "load gets the regrets");
    check(loaded.countFinishedGame() == 2, "load gets the finished-game count");
  }

  printf("%s (%d failures)\n", failures == 0 ? "all RGSC checks passed" : "RGSC checks FAILED", failures);
  return failures == 0 ? 0 : 1;
}
