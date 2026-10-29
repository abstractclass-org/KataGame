// Enclosure fork (owner, 2026-10-04): RGSC's pieces (program/rgsc.h) on hand-worked cases.
//
//   testrgsc SCRATCH_DIR
//
// Eq. 2 on a short game; the position key's order independence; the buffer's fill, replace-the-lowest, skip-known
// and EMA rules; the floor for tree candidates; sampling frequencies against the paper's R^(1/tau) / sum; the
// start probability; and the save/load round trip (SCRATCH_DIR/rgsc_buffer.txt), with each state's origin and AZdb
// member and side, and older files without them.

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

  // The swing regret (2026-10-11), the same values and outcome, gamma 0.5, scale 2: the steps to the next value (the
  // outcome after the last move) are -0.7, 1.1, 0.1, squared 0.49, 1.21, 0.01.
  {
    vector<float> s = Rgsc::swingsOfGame({0.5f, -0.2f, 0.9f}, 1.0, 0.5, 2.0, 4.0);
    check(s.size() == 3, "swings: one per turn");
    check(near(s[2], 2.0 * 0.01, 1e-6), "swing at the last turn: the step to the outcome");
    check(near(s[1], 2.0 * (1.21 + 0.5 * 0.01) / 1.5, 1e-6), "swing at turn 1: the discounted mean of its steps");
    check(near(s[0], 2.0 * (0.49 + 0.5 * 1.21 + 0.25 * 0.01) / 1.75, 1e-6), "swing at turn 0: every step, discounted");
    vector<float> capped = Rgsc::swingsOfGame({0.5f, -0.2f, 0.9f}, 1.0, 0.5, 10.0, 4.0);
    check(capped[1] == 4.0f && near(capped[2], 0.1, 1e-6), "swing is capped (8.1 -> 4), the rest is not");
    vector<float> flat = Rgsc::swingsOfGame({1.0f, 1.0f, 1.0f}, 1.0, 0.9, 50.0, 4.0);
    check(flat[0] == 0.0f && flat[2] == 0.0f, "a value that never moves has zero swing");
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

  // The swing buffer (2026-10-11): its settings, draws only when full, and each state's origin through a save and
  // a load; a file saved before the origins loads with the origin unknown.
  {
    Rgsc::Settings both = settings;
    both.startProb = 0.2;
    both.swingStartProb = 0.4;
    both.swingBufferFile = string(argv[1]) + "/rgsc_swing_buffer.txt";
    Rgsc::Settings sw = both.forSwingBuffer();
    check(sw.bufferFile == both.swingBufferFile && sw.startProb == 0.4 && sw.warmupGames == 0 && sw.sampleOnlyWhenFull,
          "the swing buffer's settings: its file and probability, no warm-up, draws only when full");
    Rgsc::Buffer swing(sw, "RGSC swing");
    Rand rand("testrgsc swing");
    Rgsc::Entry e;
    swing.offer(keyOfI(1), 0.5, movesOf(1), Rgsc::ORIGIN_TREE);
    swing.offer(keyOfI(2), 0.7, movesOf(2), Rgsc::ORIGIN_PLAYED);
    check(!swing.sample(rand, e), "the swing buffer does not draw while it has room");
    swing.offer(keyOfI(3), 0.9, movesOf(3), Rgsc::ORIGIN_PLAYED);
    check(swing.sample(rand, e), "the swing buffer draws once full");
    swing.save();
    Rgsc::Buffer loaded(sw, "RGSC swing");
    loaded.load();
    map<uint64_t, int> origins;
    for(int i = 0; i < 2000; i++) {
      check(loaded.sample(rand, e), "a loaded full buffer draws");
      origins[e.key] = e.origin;
    }
    check(origins[keyOfI(1)] == Rgsc::ORIGIN_TREE && origins[keyOfI(2)] == Rgsc::ORIGIN_PLAYED &&
          origins[keyOfI(3)] == Rgsc::ORIGIN_PLAYED, "the origins survive a save and a load");
    check(loaded.summary().find("from search trees 1, played 2") != string::npos, "the summary counts the origins");

    Rgsc::Settings old = settings;
    old.bufferFile = string(argv[1]) + "/rgsc_buffer_old.txt";
    FILE* f = fopen(old.bufferFile.c_str(), "w");
    fprintf(f, "gamesFinished 7\nentry 0.5 3 2 101 1 201 1\n");
    fclose(f);
    Rgsc::Buffer oldBuffer(old);
    oldBuffer.load();
    check(oldBuffer.size() == 1 && oldBuffer.contains(keyOfI(1)), "an old file loads");
    check(oldBuffer.maybeSample(rand, e) && e.origin == Rgsc::ORIGIN_UNKNOWN && e.moves.size() == 2,
          "an old entry's origin is unknown and its moves intact");
    check(e.member == -1 && e.side == -1, "an old entry's member and side are unknown");
  }

  // KataGame fork (2026-10-29): each state's AZdb member and side (Entry::member, Entry::side: the search that produced
  // it) through a save and a load; a file saved before them, with origins, loads with both unknown.
  {
    Rgsc::Settings tagged = settings;
    tagged.bufferFile = string(argv[1]) + "/rgsc_buffer_members.txt";
    Rgsc::Buffer members(tagged);
    const int memberOf[3] = {2, 4, -1}, sideOf[3] = {0, 1, -1};
    for(int i = 1; i <= 3; i++) {
      Rgsc::Entry entry;
      entry.key = keyOfI(i);
      entry.regret = 0.5;
      entry.moves = movesOf(i);
      entry.origin = i == 1 ? Rgsc::ORIGIN_TREE : Rgsc::ORIGIN_PLAYED;
      entry.member = memberOf[i - 1];
      entry.side = sideOf[i - 1];
      members.offer(entry);
    }
    members.save();
    Rgsc::Buffer loaded(tagged);
    loaded.load();
    Rand rand("testrgsc members");
    map<uint64_t, pair<int, int>> tags;
    Rgsc::Entry e;
    for(int i = 0; i < 2000; i++) {
      check(loaded.sample(rand, e), "a loaded buffer of tagged states draws");
      tags[e.key] = {e.member, e.side};
    }
    bool same = tags.size() == 3;
    for(int i = 1; i <= 3; i++)
      same = same && tags[keyOfI(i)] == make_pair(memberOf[i - 1], sideOf[i - 1]);
    check(same, "the members and sides survive a save and a load");
    check(loaded.summary().find("by member unknown:1 2:1 4:1") != string::npos, "the summary counts the members");

    Rgsc::Settings old = settings;
    old.bufferFile = string(argv[1]) + "/rgsc_buffer_old_origins.txt";
    FILE* f = fopen(old.bufferFile.c_str(), "w");
    fprintf(f, "gamesFinished 7\nentry 0.5 3 2 101 1 201 1 2\n");
    fclose(f);
    Rgsc::Buffer oldBuffer(old);
    oldBuffer.load();
    check(oldBuffer.size() == 1 && oldBuffer.sample(rand, e) && e.origin == Rgsc::ORIGIN_PLAYED && e.member == -1 &&
          e.side == -1, "an entry with its origin and no member loads, with the member and side unknown");
  }

  printf("%s (%d failures)\n", failures == 0 ? "all RGSC checks passed" : "RGSC checks FAILED", failures);
  return failures == 0 ? 0 : 1;
}
