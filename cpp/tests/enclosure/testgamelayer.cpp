// Enclosure fork: the game layer (Board, BoardHistory) against the Enclosure
// repository's reference engine, move by move.
//
//   testgamelayer tests/data/enclosure_games.txt
//
// The file comes from the Enclosure repository's tools/export_katago_fixture.py
// (archived and random games replayed in its Python engine, which is verified
// against the sites). For every move: the player to move, the legal set (count
// and hash), both scores after the move; then the result through BoardHistory.
// Also checks that the move notation round-trips, that the position hash is
// consistent with the position, and that a turn's two edges in either order
// reach the same hash.

#include <cstdio>
#include <fstream>
#include <sstream>

#include "../../core/rand.h"
#include "../../game/boardhistory.h"

using namespace std;

static uint64_t legalHash(vector<int> actions) {
  sort(actions.begin(), actions.end());
  uint64_t h = 1469598103934665603ULL;
  for(int a : actions)
    h = (h * 1000003ULL) ^ (uint64_t)a;
  return h;
}

int main(int argc, char** argv) {
  if(argc != 2) {
    fprintf(stderr, "usage: testgamelayer GAMES_FILE\n");
    return 2;
  }
  Board::initHash();
  ifstream in(argv[1]);
  if(!in) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  string line;
  int games = 0, moves = 0, failures = 0, transpositions = 0;
  auto fail = [&](const string& what) {
    if(failures < 20)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  };
  while(getline(in, line)) {
    istringstream head(line);
    string word, label;
    int n;
    if(!(head >> word >> label >> n) || word != "game")
      continue;
    games++;
    Board board;
    BoardHistory hist(board, P_BLACK, Rules(), 0);
    for(int i = 0; i < n; i++) {
      getline(in, line);
      istringstream row(line);
      int action, count;
      uint64_t hash;
      double blue, red;
      row >> action >> count >> hash >> blue >> red;
      string where = label + " move " + Global::intToString(i + 1);
      vector<Loc> legalLocs;
      board.getLegalMoveLocs(legalLocs);
      vector<int> legal;
      for(Loc loc : legalLocs)
        legal.push_back(Board::actionOfMoveLoc(loc));
      if((int)legal.size() != count || legalHash(legal) != hash)
        fail(where + ": legal moves differ (" + Global::intToString((int)legal.size()) + " against " +
             Global::intToString(count) + ")");
      Loc loc = Board::moveLocOfAction(action);
      Player pla = hist.presumedNextMovePla;
      if(pla != board.nextPla())
        fail(where + ": BoardHistory and Board disagree on the player to move");
      //isLegal (the cached set) agrees with the legal list on every action, pass included.
      int numLegal = 0;
      for(int a = 0; a <= enclosure::PASS; a++)
        if(board.isLegal(Board::moveLocOfAction(a), pla, false))
          numLegal++;
      for(int a : legal)
        if(!board.isLegal(Board::moveLocOfAction(a), pla, false))
          numLegal = -1;
      if(numLegal != (int)legal.size())
        fail(where + ": isLegal differs from the legal list");
      if(!hist.isLegal(board, loc, pla))
        fail(where + ": the move is not legal for BoardHistory");
      if(loc != Board::PASS_LOC) {
        Loc parsed;
        if(!Location::tryOfString(Location::toString(loc, board), board, parsed) || parsed != loc)
          fail(where + ": notation does not round-trip (" + Location::toString(loc, board) + ")");
      }
      hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
      moves++;
      if(fabs(board.getScore(P_BLACK) - blue) > 1e-6 || fabs(board.getScore(P_WHITE) - red) > 1e-6)
        fail(where + ": scores differ");
      board.checkConsistency();
    }
    getline(in, line);
    istringstream tail(line);
    int result;
    tail >> word >> result;
    Player expected = result > 0 ? P_BLACK : result < 0 ? P_WHITE : C_EMPTY;
    if(!hist.isGameFinished || hist.winner != expected)
      fail(label + ": result differs (finished " + Global::boolToString(hist.isGameFinished) + ")");
    if((result > 0) != (hist.finalWhiteMinusBlackScore < 0) || (result < 0) != (hist.finalWhiteMinusBlackScore > 0))
      fail(label + ": finalWhiteMinusBlackScore has the wrong sign");
  }

  //Transpositions: in turns with two edges whose order can be swapped, both orders reach one hash.
  {
    Rand rand("testgamelayer transpositions");
    for(int g = 0; g < 200; g++) {
      Board board;
      BoardHistory hist(board, P_BLACK, Rules(), 0);
      while(!board.isGameOver()) {
        vector<Loc> legal;
        board.getLegalMoveLocs(legal);
        Loc first = legal[rand.nextUInt((uint32_t)legal.size())];
        if(board.pos.moves_left == 2 && first != Board::PASS_LOC) {
          Board a(board);
          a.playMoveAssumeLegal(first, a.nextPla());
          vector<Loc> legal2;
          a.getLegalMoveLocs(legal2);
          Loc second = legal2[rand.nextUInt((uint32_t)legal2.size())];
          Board b(board);
          if(second != Board::PASS_LOC && b.isLegal(second, b.nextPla(), false)) {
            b.playMoveAssumeLegal(second, b.nextPla());
            if(b.isLegal(first, b.nextPla(), false)) {
              a.playMoveAssumeLegal(second, a.nextPla());
              b.playMoveAssumeLegal(first, b.nextPla());
              bool same = a.pos.edges[0].size() == b.pos.edges[0].size() && a.pos.edges[1].size() == b.pos.edges[1].size() &&
                a.pos.scores[0] == b.pos.scores[0] && a.pos.scores[1] == b.pos.scores[1];
              if(same) {
                transpositions++;
                if(a.pos_hash != b.pos_hash)
                  fail("transposition: the same edges in the other order give another hash");
              }
            }
          }
        }
        hist.makeBoardMoveAssumeLegal(board, first, board.nextPla(), NULL);
      }
    }
  }

  printf("%d games, %d moves, %d transpositions checked, %d failures\n", games, moves, transpositions, failures);
  return failures == 0 ? 0 : 1;
}
