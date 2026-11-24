// Tak (abscls-shfz.22): AZdb's statistics of a position (game/tak/diversity.h), without a build of katago:
//   - the game interface's member Diversity: an encoding part that takes it gives the front 24 statistics, 28 moves and
//     the normalizer (GameInterface::diversityStatistics, diversityMoves, diversityStatisticsOf, DiversityNormalizer);
//   - the statistics of positions worked by hand (the engine repository's games/tak/tests/test_diversity_cpp.py holds
//     the rest, against a plain Python computation);
//   - each statistic on the 8 images of a position;
//   - the normalizer's z, its floor, its clip, the weights of the groups and one finished side.
//
// Built and run in WSL, from katagame/cpp:
//   g++ -std=c++17 -O2 -iquote . tests/tak/testdiversity.cpp -o testdiversity
//   ./testdiversity
// It prints the failures, and its exit code is 0 when every check holds.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../../game/tak/diversity.h"

using namespace std;

namespace {

// The members of an encoding part that the interface asks for Diversity: the rules' Position and the type itself.
struct StubRules {
  using Position = tak::Position;
};
struct StubEncoding {
  using Rules = StubRules;
  using Diversity = tak::Diversity;
};
static_assert(GameInterface::encoding::HasDiversity<StubEncoding>::value, "the interface finds Tak's Diversity");
static_assert(GameInterface::diversityStatistics<StubEncoding>() == 24, "24 statistics");
static_assert(GameInterface::diversityMoves<StubEncoding>() == 28, "28 moves");

int failures = 0;

void check(bool ok, const string& what) {
  if(!ok) {
    failures++;
    printf("failed: %s\n", what.c_str());
  }
}
void near(double a, double b, const string& what) {
  check(fabs(a - b) < 1e-12, what + ": " + to_string(a) + " against " + to_string(b));
}

tak::Position position(const string& text) {
  tak::Position p;
  string why;
  if(!tak::GameRules::parse_position(text, p, why)) {
    printf("no position %s: %s\n", text.c_str(), why.c_str());
    exit(1);
  }
  return p;
}

const string ROW = "x5/x5/x5/x5/1,1,1,1,x 1 4 0";            // a white row of four: e1 completes a road
const string FULL_ROW = "x5/x5/x5/x5/1,1,1,1,1 1 4 0";       // a white road, the game's end
const string FULL_COLUMN = "2,x4/2,x4/2,x4/2,x4/2,x4 1 4 0";  // a black road
const string STACKS = "x5/x5/x2,21,x2/x5/1221,x4 1 3 0";     // white tops on c3 (over B) and a1 (over W, B, B)
const string WALLS = "1C,x3,1S/x5/x,2,x3/x,1S,x,2S,x/x5 1 4 0";
const string MIXED = "1,1,x,1,1/x5/2,2,2,x2/x2,1S,x2/x,12,x3 1 6 0";

void byHand() {
  double x[24];
  GameInterface::diversityStatisticsOf<StubEncoding>(position(ROW), 0, x);
  near(x[tak::Diversity::ROADS], log1p(1.0), "one road threat");
  near(x[tak::Diversity::ROADS + 1], 0.8, "the longest reach is 4 of 5");
  near(x[tak::Diversity::ROADS + 2], 1.0, "one group holds every road piece");
  near(x[tak::Diversity::ROADS + 3], 0.0, "the opponent threatens no road");
  near(x[tak::Diversity::EDGES], 1.0 / (16.0 / 25.0), "every top piece is on the edge, over the edge's share");
  near(x[tak::Diversity::TEMPO], 0.0, "no spread");
  GameInterface::diversityStatisticsOf<StubEncoding>(position(FULL_ROW), 0, x);
  near(x[tak::Diversity::ROADS], 0.0, "a complete road leaves no threat");
  near(x[tak::Diversity::ROADS + 1], 0.8, "a complete road of White reaches N - 1");
  GameInterface::diversityStatisticsOf<StubEncoding>(position(FULL_COLUMN), 1, x);
  near(x[tak::Diversity::ROADS + 1], 0.8, "a complete road of Black reaches N - 1");
  GameInterface::diversityStatisticsOf<StubEncoding>(position(FULL_COLUMN), 0, x);
  near(x[tak::Diversity::ROADS + 4], 0.8, "and so from the other player's side");
  GameInterface::diversityStatisticsOf<StubEncoding>(position(STACKS), 0, x);
  near(x[tak::Diversity::STACKS], 1.0, "White's captives are all of Black's three pieces");
  near(x[tak::Diversity::STACKS + 1], 1.0 / 3.0, "White has one piece of its own under a top piece");
  near(x[tak::Diversity::STACKS + 2], 0.8, "the tallest stack is four of five");
  GameInterface::diversityStatisticsOf<StubEncoding>(position(WALLS), 0, x);
  near(x[tak::Diversity::KINDS], 2.0 / 3.0, "two walls of three top pieces");
  near(x[tak::Diversity::KINDS + 1], 1.0, "the capstone is on the board");
  near(x[tak::Diversity::CONTACT], 1.0 / 3.0, "one top piece of three touches Black's");
  near(x[tak::Diversity::CONTACT + 1], 0.5, "one wall of two is beside a road piece of Black's");
}

// p under one of the 8 symmetries of the square: a rotation by symmetry % 4 quarters, then a mirror.
tak::Position image(const tak::Position& p, int symmetry) {
  tak::Position q = p;
  for(int i = 0; i < tak::NUM_POINTS; i++) {
    q.top[i] = q.rest[i] = 0;
    q.height[i] = 0;
  }
  q.owner[0] = q.owner[1] = q.walls = q.caps = 0;
  const int n = p.size;
  for(int y = 0; y < n; y++)
    for(int x = 0; x < n; x++) {
      int a = x, b = y;
      for(int r = 0; r < symmetry % 4; r++) {
        const int t = a;
        a = n - 1 - b;
        b = t;
      }
      if(symmetry >= 4)
        a = n - 1 - a;
      const int from = tak::point(x, y), to = tak::point(a, b);
      q.top[to] = p.top[from];
      q.rest[to] = p.rest[from];
      q.height[to] = p.height[from];
      tak::set_top(q, to, tak::kind_at(p, from));
    }
  return q;
}

void symmetries() {
  for(const string& text : {WALLS, STACKS, MIXED, ROW, FULL_ROW, FULL_COLUMN}) {
    const tak::Position p = position(text);
    for(int player = 0; player < 2; player++) {
      double expected[24];
      tak::Diversity::statistics(p, player, expected);
      for(int s = 1; s < 8; s++) {
        double x[24];
        tak::Diversity::statistics(image(p, s), player, x);
        for(int i = 0; i < 24; i++)
          near(x[i], expected[i], text + ", player " + to_string(player) + ", symmetry " + to_string(s) +
                                      ", statistic " + to_string(i));
      }
    }
  }
}

void normalizer() {
  double mean[24], sd[24], x[24], z[24];
  for(int i = 0; i < 24; i++) {
    mean[i] = 0.0;
    sd[i] = 2.0;
    x[i] = 2.0;
  }
  sd[5] = 0.0;  // the floor of 0.1
  GameInterface::DiversityNormalizer<StubEncoding> n(mean, sd);
  n.z(x, z);
  near(z[0], 1.0 / 2.0, "one standard deviation, over the group of four");
  near(z[5], 3.0 / sqrt(6.0), "a statistic that never varied: the floor and the clip");
  near(z[23], 1.0 / sqrt(2.0), "the tempo's group of two");
  // One finished side, on an alpha of 0.5: m1 and m2 move half of the way to the side's mean of x and of x^2.
  for(int i = 0; i < 24; i++)
    sd[i] = 1.0;
  tak::DiversityNormalizer half(mean, sd, 0.5);
  double side[2 * 24];
  for(int i = 0; i < 24; i++) {
    side[i] = 2.0;
    side[24 + i] = 4.0;
  }
  half.update(side, 2);
  near(half.mean(0), 1.5, "m1: half of the side's mean of 3");
  near(half.sd(0), sqrt(0.5 * 1.0 + 0.5 * 10.0 - 1.5 * 1.5), "m2: half of the side's mean of x^2, 10");
  const double before = half.mean(0);
  half.update(nullptr, 0);
  near(half.mean(0), before, "a side without positions changes nothing");
}

}  // namespace

int main() {
  byHand();
  symmetries();
  normalizer();
  if(failures > 0)
    printf("%d checks failed\n", failures);
  else
    printf("Tak's statistics of a position: every check holds\n");
  return failures > 0 ? 1 : 0;
}
