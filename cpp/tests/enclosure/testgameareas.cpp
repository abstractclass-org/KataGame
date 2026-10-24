// KataGame fork: the areas that KataGo's look-ahead horizons and its komi sampler read where KataGo reads the board's
// area (GameInterface::lookaheadArea and komiArea, gameinterface.h), through the front: the defaults with stub games,
// and the build's game on its standard board, against the values pinned below.
//
//   testgameareas
//
// It prints the build's game, its standard board and both areas, then for each short-term value target its lambda
// (KataGo's: 1 minus the weight of the row's own search value, 1 / (1 + A * c) on the look-ahead area A, as
// trainingwrite.cpp writes it) and its mean look-ahead in moves, and the komi draw's standard deviation for one
// configured as 100 points, each beside its value with the board's area.

#include <cmath>
#include <cstdio>
#include <cstring>

#include "../../core/rand.h"
#include "../../game/board.h"
#include "../../program/playutils.h"

using namespace std;

static int failures = 0;

static void check(bool ok, const char* what) {
  if(!ok) {
    fprintf(stderr, "FAIL %s\n", what);
    failures++;
  }
}

//A game that gives neither area, and one that gives both.
struct NoAreas {};
struct BothAreas {
  static constexpr double LOOKAHEAD_AREA = 100.0;
  static constexpr double KOMI_AREA = 49.0;
};
static_assert(GameInterface::lookaheadArea<NoAreas>(19, 19) == 361.0, "the default is the board's area");
static_assert(GameInterface::lookaheadArea<NoAreas>(7, 6) == 42.0, "the default is the board's area");
static_assert(GameInterface::komiArea<NoAreas>(67, 67) == 4489.0, "the default is the board's area");
static_assert(GameInterface::lookaheadArea<BothAreas>(67, 67) == 100.0, "a game's own area");
static_assert(GameInterface::komiArea<BothAreas>(67, 67) == 49.0, "a game's own area");

//The areas of each game of KataGame on its standard board: the look-ahead area, then the komi area. Blueprints gives
//Go's 361 for both (its 67 x 67 view holds the window, and its moves do not reach that far). The other games give
//none, so they keep their board's area, as before the areas were the game's.
static bool pinnedAreas(double& lookahead, double& komi) {
  if(strcmp(GameRules::NAME, "blueprints") == 0) {
    lookahead = komi = 361.0;
    return true;
  }
  if(strcmp(GameRules::NAME, "enclosure") == 0) {
    lookahead = komi = 19.0 * 19.0;
    return true;
  }
  if(strcmp(GameRules::NAME, "connect4") == 0) {
    lookahead = komi = 7.0 * 6.0;
    return true;
  }
  return false;
}

int main() {
  Board::initHash();
  const Board board;
  const double boardArea = (double)board.x_size * board.y_size;
  printf("game %s, board %d x %d: look-ahead area %g, komi area %g (the board's area %g)\n", GameRules::NAME,
         board.x_size, board.y_size, board.lookaheadArea(), board.komiArea(), boardArea);

  double lookahead = 0.0, komi = 0.0;
  check(pinnedAreas(lookahead, komi), "the build's game has pinned areas");
  check(board.lookaheadArea() == lookahead, "the game's look-ahead area");
  check(board.komiArea() == komi, "the game's komi area");

  //The short-term value targets: KataGo's formula (trainingwrite.cpp) on the look-ahead area.
  static const double FACTORS[3] = {0.176, 0.056, 0.016};
  static const char* SLOTS[3] = {"C4-7", "C8-11", "C12-15"};
  for(int i = 0; i < 3; i++) {
    const double now = 1.0 / (1.0 + board.lookaheadArea() * FACTORS[i]);
    const double boardNow = 1.0 / (1.0 + boardArea * FACTORS[i]);
    printf("short-term value target %d (%s): lambda %.6f, mean look-ahead %.1f moves (the board's area: lambda %.6f, "
           "%.1f moves)\n", i, SLOTS[i], 1.0 - now, (1.0 - now) / now, 1.0 - boardNow, (1.0 - boardNow) / boardNow);
  }

  //The komi draw: a standard deviation configured as 100 points, times sqrt(komi area) / 19, and no handicap at
  //probability 0.
  Rand rand("testgameareas");
  const ExtraBlackAndKomi drawn =
    PlayUtils::chooseExtraBlackAndKomi(0.0f, 100.0f, 1.0, 0.0, 0, 0.0, 0.0f, 0.0, 0.0f, board, rand);
  check(drawn.komiStdev == 100.0f * (float)(sqrt(komi) / 19.0), "the komi draw's standard deviation");
  check(drawn.extraBlack == 0 && drawn.komiMean == 0.0f, "the komi draw's mean and handicap");
  printf("komi standard deviation configured as 100 points: drawn with %.2f points (the board's area: %.2f)\n",
         drawn.komiStdev, 100.0 * sqrt(boardArea) / 19.0);

  if(failures > 0) {
    fprintf(stderr, "%d failures\n", failures);
    return 1;
  }
  printf("testgameareas: all checks pass\n");
  return 0;
}
