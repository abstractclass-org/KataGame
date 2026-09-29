// Enclosure fork: which board symmetries the rules respect (plan step 4, D5).
//
//   testsymmetry [GAMES_PER_SYMMETRY]
//
// For each of KataGo's 8 symmetries (bit 0 flipY, bit 1 flipX, bit 2 transpose),
// random games are played twice: on the board, and on its image under the
// symmetry, each move mapped by SymmetryHelpers::getSymAction. After every move
// the image game must be the image of the game: the same legal moves (mapped
// one to one), the same edges, protected and placed edges and history, and the
// same scores. The feature planes (net1's inputs) of the image must be the
// feature planes moved point by point. The start position's own symmetries
// are reported too.
//
// Prints one line per symmetry. Exit code 0 when every symmetry either agrees
// throughout or is reported as broken by the rules (the table is the result).

#include <cstdio>
#include <cstdlib>
#include <set>

#include "../../core/rand.h"
#include "../../neuralnet/nninputs.h"

using namespace std;

typedef tuple<int,int,int,int> EdgeKey;

static set<EdgeKey> edgeSet(const vector<enclosure::Edge>& v) {
  set<EdgeKey> s;
  for(const enclosure::Edge& e : v)
    s.insert(EdgeKey(e.ax, e.ay, e.bx, e.by));
  return s;
}

static set<EdgeKey> edgeSet(const enclosure::Edge* v, int n) {
  return edgeSet(vector<enclosure::Edge>(v, v + n));
}

//Differences between two positions (edge sets, turn state, history, scores); maxScoreDiff grows.
static int stateDifferences(const enclosure::Position& a, const enclosure::Position& b, double& maxScoreDiff) {
  int d = 0;
  for(int k = 0; k < 2; k++) {
    d += edgeSet(a.edges[k]) != edgeSet(b.edges[k]);
    d += a.scores[k] != b.scores[k];
    maxScoreDiff = max(maxScoreDiff, fabs(a.scores[k] - b.scores[k]));
  }
  d += edgeSet(a.protected_) != edgeSet(b.protected_);
  d += edgeSet(a.placed) != edgeSet(b.placed);
  d += a.player != b.player || a.moves_left != b.moves_left || a.move_count != b.move_count;
  for(int h = 0; h < enclosure::HISTORY_TURNS; h++)
    d += edgeSet(a.history[h], a.history_n[h]) != edgeSet(b.history[h], b.history_n[h]);
  return d;
}

static void symXY(int& x, int& y, int symmetry) {
  if(SymmetryHelpers::isFlipX(symmetry)) x = enclosure::SIZE - x - 1;
  if(SymmetryHelpers::isFlipY(symmetry)) y = enclosure::SIZE - y - 1;
  if(SymmetryHelpers::isTranspose(symmetry)) swap(x, y);
}

//Feature plane entries of the image that are not the original's moved point by point.
static int featureDifferences(const enclosure::Position& p, const enclosure::Position& image, int symmetry) {
  const int S = enclosure::SIZE, F = enclosure::FEATURE_PLANES;
  vector<uint8_t> a(S * S * F), b(S * S * F);
  enclosure::feature_planes(p, a.data());
  enclosure::feature_planes(image, b.data());
  int d = 0;
  for(int y = 0; y < S; y++)
    for(int x = 0; x < S; x++) {
      int sx = x, sy = y;
      symXY(sx, sy, symmetry);
      for(int f = 0; f < F; f++)
        d += a[(y * S + x) * F + f] != b[(sy * S + sx) * F + f];
    }
  return d;
}

int main(int argc, char** argv) {
  const int gamesPerSymmetry = argc > 1 ? atoi(argv[1]) : 100;
  Board::initHash();
  const Board start;
  printf("sym  flipY flipX transpose | start fixed | moves  legal-diff state-diff feature-diff max-score-diff\n");
  int unexpected = 0;
  for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
    Rand rand("testsymmetry " + Global::intToString(symmetry));
    double maxScoreDiff = 0;
    bool startFixed = stateDifferences(start.pos, SymmetryHelpers::getSymPosition(start.pos, symmetry), maxScoreDiff) == 0;
    long moves = 0, legalDiff = 0, stateDiff = 0, featureDiff = 0;
    for(int game = 0; game < gamesPerSymmetry; game++) {
      Board board;
      Board image = SymmetryHelpers::getSymBoard(board, symmetry);
      //Feature planes are expensive: check them at a few move counts per game.
      const int featureEvery = 7 + game % 5;
      while(!board.isGameOver()) {
        vector<Loc> legal, imageLegal;
        board.getLegalMoveLocs(legal);
        image.getLegalMoveLocs(imageLegal);
        set<Loc> mapped;
        for(Loc loc : legal)
          mapped.insert(SymmetryHelpers::getSymMoveLoc(loc, image, symmetry));
        if(mapped != set<Loc>(imageLegal.begin(), imageLegal.end()) || mapped.size() != legal.size())
          legalDiff++;
        Loc loc = legal[rand.nextUInt((uint32_t)legal.size())];
        Loc imageLoc = SymmetryHelpers::getSymMoveLoc(loc, image, symmetry);
        Player pla = board.nextPla();
        if(image.nextPla() != pla || !image.isLegal(imageLoc, pla, false)) {
          legalDiff++;
          break;
        }
        board.playMoveAssumeLegal(loc, pla);
        image.playMoveAssumeLegal(imageLoc, pla);
        moves++;
        stateDiff += stateDifferences(SymmetryHelpers::getSymPosition(board.pos, symmetry), image.pos, maxScoreDiff) != 0;
        if(moves % featureEvery == 0)
          featureDiff += featureDifferences(board.pos, image.pos, symmetry) != 0;
      }
    }
    printf("%d    %d     %d     %d         | %-11s | %6ld %10ld %10ld %12ld %.3g\n",
           symmetry, SymmetryHelpers::isFlipY(symmetry), SymmetryHelpers::isFlipX(symmetry),
           SymmetryHelpers::isTranspose(symmetry), startFixed ? "yes" : "no",
           moves, legalDiff, stateDiff, featureDiff, maxScoreDiff);
    //The identity must agree with itself and fix the start.
    if(symmetry == 0 && (!startFixed || legalDiff || stateDiff || featureDiff))
      unexpected++;
  }

  //Round trips and composition, on positions from random games.
  {
    Rand rand("testsymmetry round trips");
    long checks = 0, bad = 0;
    for(int game = 0; game < 20; game++) {
      Board board;
      while(!board.isGameOver()) {
        vector<Loc> legal;
        board.getLegalMoveLocs(legal);
        for(int s1 = 0; s1 < SymmetryHelpers::NUM_SYMMETRIES; s1++) {
          double ignored = 0;
          Board image = SymmetryHelpers::getSymBoard(board, s1);
          //The inverse maps the image back, position and moves.
          bad += stateDifferences(SymmetryHelpers::getSymPosition(image.pos, SymmetryHelpers::invert(s1)), board.pos, ignored) != 0;
          for(Loc loc : legal) {
            Loc back = SymmetryHelpers::getSymMoveLoc(SymmetryHelpers::getSymMoveLoc(loc, image, s1), board, SymmetryHelpers::invert(s1));
            bad += back != loc;
          }
          //Composition: s1 then s2 is compose(s1, s2).
          for(int s2 = 0; s2 < SymmetryHelpers::NUM_SYMMETRIES; s2++) {
            enclosure::Position twice = SymmetryHelpers::getSymPosition(image.pos, s2);
            enclosure::Position once = SymmetryHelpers::getSymPosition(board.pos, SymmetryHelpers::compose(s1, s2));
            bad += stateDifferences(twice, once, ignored) != 0;
            checks++;
          }
        }
        board.playMoveAssumeLegal(legal[rand.nextUInt((uint32_t)legal.size())], board.nextPla());
      }
    }
    printf("round trips and composition: %ld checks, %ld failures\n", checks, bad);
    unexpected += bad != 0;
  }

  //Root symmetry pruning: the start is its own image under the symmetries that fix it (flipY alone),
  //and of each pair of mirror-image moves one is kept.
  {
    const BoardHistory hist(start, P_BLACK, Rules(), 0);
    vector<int> validSymmetries;
    bool* isSymDupLoc = new bool[Board::NUM_MOVE_LOCS];
    SymmetryHelpers::markDuplicateMoveLocs(start, hist, NULL, vector<int>(), isSymDupLoc, validSymmetries);
    vector<int> expected;
    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      double ignored = 0;
      if(stateDifferences(start.pos, SymmetryHelpers::getSymPosition(start.pos, symmetry), ignored) == 0)
        expected.push_back(symmetry);
    }
    vector<Loc> legal;
    start.getLegalMoveLocs(legal);
    int kept = 0, dups = 0, bad = 0;
    for(Loc loc : legal) {
      if(isSymDupLoc[loc]) {
        dups++;
        //Some image of a duplicate is kept.
        bool hasKept = false;
        for(int symmetry : validSymmetries)
          hasKept = hasKept || !isSymDupLoc[SymmetryHelpers::getSymMoveLoc(loc, start, symmetry)];
        bad += !hasKept;
      }
      else {
        kept++;
        //No two kept moves are images of each other.
        for(int symmetry : validSymmetries) {
          Loc symLoc = SymmetryHelpers::getSymMoveLoc(loc, start, symmetry);
          bad += symLoc != loc && !isSymDupLoc[symLoc];
        }
      }
    }
    for(int loc = 0; loc < Board::NUM_MOVE_LOCS; loc++)
      if(isSymDupLoc[loc] && !start.isLegal((Loc)loc, P_BLACK, false))
        bad++;
    printf("root pruning at the start: symmetries");
    for(int symmetry : validSymmetries)
      printf(" %d", symmetry);
    printf("; %d legal moves, %d kept, %d duplicates, %d failures\n", (int)legal.size(), kept, dups, bad);
    unexpected += bad != 0 || validSymmetries != expected;
    delete[] isSymDupLoc;
  }
  return unexpected == 0 ? 0 : 1;
}
