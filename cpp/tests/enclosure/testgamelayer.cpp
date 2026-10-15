// Enclosure fork: the game layer (Board, BoardHistory) against the Enclosure
// repository's reference engine, move by move.
//
//   testgamelayer tests/data/enclosure_games.txt
//
// The file comes from the Enclosure repository's games/enclosure/tools/export_katago_fixture.py
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
#include "../../neuralnet/nninputs.h"

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
  int games = 0, moves = 0, failures = 0, transpositions = 0, singleChecks = 0;
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
      //Position::is_legal (one move, what Board::isLegal asks before the list is built) agrees with the list:
      //every listed move at every position, every action at every 50th.
      for(int a : legal)
        if(!board.pos.is_legal(a))
          fail(where + ": is_legal rejects listed action " + Global::intToString(a));
      if(moves % 50 == 0) {
        vector<char> listed(enclosure::PASS + 1, 0);
        for(int a : legal)
          listed[a] = 1;
        for(int a = 0; a <= enclosure::PASS; a++)
          if(board.pos.is_legal(a) != (bool)listed[a])
            fail(where + ": is_legal differs from the list at action " + Global::intToString(a));
        singleChecks++;
        //The image's legal moves under each symmetry, mapped (getSymBoard) and listed afresh, are the same.
        for(int symmetry = 1; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
          Board mapped = SymmetryHelpers::getSymBoard(board, symmetry);
          Board fresh(board.x_size, board.y_size);
          fresh.setPosition(SymmetryHelpers::getSymPosition(board.pos, symmetry));
          if(mapped.legalActions() != fresh.legalActions())
            fail(where + ": mapped legal moves differ under symmetry " + Global::intToString(symmetry));
        }
      }
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

  //Timeout passes (owner, 2026-10-04): a replayed game accepts the site's pass anywhere (makeBoardMoveTolerant), as the
  //engine's forfeit_turn; the search's legality (isLegal) still allows it only without a legal edge.
  int timeoutPasses = 0;
  {
    Rand rand("testgamelayer timeout passes");
    for(int g = 0; g < 200; g++) {
      Board board;
      BoardHistory hist(board, P_BLACK, Rules(), 0);
      while(!board.isGameOver()) {
        Player pla = board.nextPla();
        vector<Loc> legal;
        board.getLegalMoveLocs(legal);
        bool stuck = legal.size() == 1 && legal[0] == Board::PASS_LOC;
        if(!stuck && rand.nextBool(0.1)) {
          if(board.isLegal(Board::PASS_LOC, pla, false) || hist.isLegal(board, Board::PASS_LOC, pla))
            fail("timeout pass: isLegal allows a pass while edges are legal");
          enclosure::Position expected = board.pos;
          expected.forfeit_turn();
          if(!hist.makeBoardMoveTolerant(board, Board::PASS_LOC, pla))
            fail("timeout pass: makeBoardMoveTolerant refuses it");
          const enclosure::Position& got = board.pos;
          if(got.move_count != expected.move_count || got.player != expected.player || got.moves_left != expected.moves_left ||
             got.scores[0] != expected.scores[0] || got.scores[1] != expected.scores[1] ||
             got.edges[0].size() != expected.edges[0].size() || got.edges[1].size() != expected.edges[1].size())
            fail("timeout pass: the replayed pass differs from forfeit_turn");
          if(hist.presumedNextMovePla != board.nextPla())
            fail("timeout pass: BoardHistory and Board disagree on the player to move");
          board.checkConsistency();
          timeoutPasses++;
          continue;
        }
        hist.makeBoardMoveAssumeLegal(board, legal[rand.nextUInt((uint32_t)legal.size())], pla, NULL);
      }
      if(!hist.isGameFinished)
        fail("timeout pass: a game with passes did not finish");
    }
  }

  //Komi (owner, 2026-10-04): points added to Red's final score. The result counts it; the network cache hash folds it
  //in only when nonzero; and the inputs equal those of a position where the favoured side's score is higher by komi.
  int komiChecks = 0;
  {
    Rand rand("testgamelayer komi");
    const float komis[] = {0.0f, 0.5f, -0.5f, 12.0f, -37.5f, 144.0f, -900.0f, 2500.0f};
    for(int g = 0; g < 60; g++) {
      Board board;
      BoardHistory hist(board, P_BLACK, Rules(), 0);
      const float komi = komis[g % 8];
      hist.setKomi(komi);
      BoardHistory hist0(hist);
      hist0.setKomi(0.0f);
      MiscNNInputParams params;
      float bin[NNInputs::NUM_FEATURES_SPATIAL_V8 * 19 * 19], glob[NNInputs::NUM_FEATURES_GLOBAL_V8];
      float bin0[NNInputs::NUM_FEATURES_SPATIAL_V8 * 19 * 19], glob0[NNInputs::NUM_FEATURES_GLOBAL_V8];
      int ply = 0;
      while(!board.isGameOver()) {
        if(ply % 7 == 0) {
          Player pla = board.nextPla();
          Hash128 h = BoardHistory::getSituationRulesAndKoHash(board, hist, pla, 0.0, hist.modes);
          Hash128 h0 = BoardHistory::getSituationRulesAndKoHash(board, hist0, pla, 0.0, hist0.modes);
          if((komi == 0.0f) != (h == h0))
            fail("komi: the hash must change with a nonzero komi only");
          //The same position with the favoured side's score raised by komi, at komi 0, gives the same inputs.
          Board shifted(board);
          if(komi > 0) shifted.pos.scores[1] += komi;
          if(komi < 0) shifted.pos.scores[0] -= komi;
          NNInputs::fillRowV8(board, hist, pla, params, 19, 19, true, bin, glob);
          NNInputs::fillRowV8(shifted, hist0, pla, params, 19, 19, true, bin0, glob0);
          for(int i = 0; i < NNInputs::NUM_FEATURES_SPATIAL_V8 * 19 * 19; i++)
            if(bin[i] != bin0[i]) { fail("komi: the spatial inputs change with komi"); break; }
          for(int i = 0; i < NNInputs::NUM_FEATURES_GLOBAL_V8; i++)
            if(glob[i] != glob0[i]) { fail("komi: the inputs differ from the shifted score's"); break; }
          if(hist.currentSelfKomi(P_WHITE, 0.0) != komi || hist.currentSelfKomi(P_BLACK, 0.0) != -komi)
            fail("komi: currentSelfKomi is not Red's komi, from each side");
          komiChecks++;
        }
        vector<Loc> legal;
        board.getLegalMoveLocs(legal);
        Loc loc = legal[rand.nextUInt((uint32_t)legal.size())];
        Player pla = board.nextPla();
        hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
        ply++;
      }
      double margin = board.getScore(P_WHITE) - board.getScore(P_BLACK) + komi;
      Player expected = margin > 0 ? P_WHITE : margin < 0 ? P_BLACK : C_EMPTY;
      if(!hist.isGameFinished || hist.winner != expected)
        fail("komi: the winner does not count komi");
      if(fabs(hist.finalWhiteMinusBlackScore - margin) > 1e-3 * (1.0 + fabs(margin)))
        fail("komi: finalWhiteMinusBlackScore does not count komi");
    }
  }

  printf("%d games, %d moves, %d transpositions checked, %d positions with every action checked one by one, "
         "%d timeout passes replayed, %d komi checks, %d failures\n",
         games, moves, transpositions, singleChecks, timeoutPasses, komiChecks, failures);
  return failures == 0 ? 0 : 1;
}
