// KataGame fork (2026-10-27, abscls-luzl.7): RGSC's exact states (program/rgsc.h), on a build of a game that gives
// its whole state as a text (GameInterface::stateText: Blueprints).
//
//   testrgscstates SCRATCH_DIR [GAMES] [--buffer FILE] [--write FILE] [SETTING=VALUE ...]
//
// SETTING=VALUE: the game's settings (GameInterface::setSetting), such as viewOrientation=randomGame. The test plays
// GAMES random games (default 30) of the legal actions, each with the game's draws at its start (drawStart), a komi
// and, in a third of them, asymmetric playouts. At every position:
// - the state's text gives the state back (parseState): the same position's hash with the player to move, the same
//   input_hash (the history planes), the same player, moves played, pending move (continuesMove: a pick waiting for
//   its drop), legal actions, the same text again, and the same network inputs (V10 and V11, with history, at each
//   allowed symmetry), so the same state key;
// - every position of a game has its own state key.
// The played positions of 3 more games (self-play's candidates after a game): their keys (playedKeys) and entries
// (playedEntry) are their states', and an entry's start (startOf) is its state with its turn number and its context.
// The keys: an action played twice in a row by the same player cancels in the moves' unordered key (keyOf) and not
// in the state key; two actions in either order give the same moves' key and two state keys; two edits in either
// order (when the search finds a pair that commutes) give one state and one key; the komi and the asymmetric playouts
// change the key, and a context of -0 or of no advantage is normalized.
// The records: restoreState rejects a garbled text, a text of another state (a score changed) and a wrong key. The
// buffer: a state offered again is a duplicate, the same key with another state a collision; a save and a load give
// the states back with their contexts and turn numbers; a load rejects a line of another version, of another game, a
// state whose key does not match, a garbled state and a line of moves, and keeps the good lines.
// With --buffer FILE (a self-play's buffer, rgscBufferFile): every entry of the file restores (the load checks its
// key), and the entries that the draws reach restore with their texts as they were, or with texts that extend them
// (a buffer of an older text: Blueprints' from before the inputs version 11 lacks the turns). With --write FILE: a
// buffer of a state of each random game (in every second game one with a pick waiting for its drop), with its komi
// and asymmetric playouts and the regret 1, for a self-play that restarts from them (rgscStartProb 1).
// On a build of a game without a state's text (Enclosure), the test checks the defaults: no text, a load that rejects
// a state's line, and in 3 random games the played positions' keys and entries (the moves' key, keyOf, and the moves)
// and their starts (the empty board and the moves). With --buffer FILE (a copy of a run's buffer), every line of the
// file loads.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "../../core/rand.h"
#include "../../neuralnet/nninputs.h"
#include "../../program/rgsc.h"

using namespace std;

static int failures = 0;
static void check(bool ok, const string& what) {
  if(!ok) {
    if(failures < 20)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  }
}

//The network's inputs of a position (V10, or V11 with its turns, with history), packed, at a symmetry.
static void inputsOf(const Board& board, const Rgsc::Context& context, int symmetry, vector<uint32_t>& bits,
                     vector<float>& global, int inputsVersion = 10) {
  Rules rules;
  rules.komi = context.komi;
  BoardHistory hist(board, board.nextPla(), rules, 0);
  MiscNNInputParams params;
  if(context.advantage != 0)
    params.playoutDoublingAdvantage =
      board.nextPla() == context.advantagePla ? context.advantage : -context.advantage;
  bits.assign(NNInputs::NUM_SPATIAL_WORDS_V10, 0u);
  global.assign(NNInputs::NUM_FEATURES_GLOBAL_V11, 0.0f);  //V10's global inputs fill its first ones
  if(inputsVersion == 11)
    NNInputs::fillRowV11Packed(board, hist, board.nextPla(), params, Board::MAX_LEN, Board::MAX_LEN, false, bits.data(),
                               global.data(), symmetry);
  else
    NNInputs::fillRowV10Packed(board, hist, board.nextPla(), params, Board::MAX_LEN, Board::MAX_LEN, false, bits.data(),
                               global.data(), symmetry);
}

//A random game of the legal actions from the game's start with its draws: the boards after each action, the first
//included, and the moves.
static void randomGame(Rand& rand, vector<Board>& boards, vector<Move>& moves) {
  Board board;
  GameRules::Position start = board.pos;
  GameInterface::drawStart<GameRules>(start, [&rand]() { return rand.nextUInt64(); });
  board.setPosition(start);
  boards.assign(1, board);
  moves.clear();
  vector<Loc> legal;
  while(!board.isGameOver()) {
    board.getLegalMoveLocs(legal);
    if(legal.empty())
      break;
    const Loc loc = legal[rand.nextUInt((uint32_t)legal.size())];
    moves.push_back(Move(loc, board.nextPla()));
    board.playMoveAssumeLegal(loc, board.nextPla());
    boards.push_back(board);
  }
}

static string withoutLast(const string& s) { return s.substr(0, s.size() - 1); }

static bool sameMoves(const vector<Move>& a, const vector<Move>& b) {
  if(a.size() != b.size())
    return false;
  for(size_t i = 0; i < a.size(); i++)
    if(a[i].loc != b[i].loc || a[i].pla != b[i].pla)
      return false;
  return true;
}

//The played positions of a random game (Rgsc::playedKeys, playedEntry) against its boards, and the starts of their
//entries (Rgsc::startOf): in a game that keeps states each state's key, entry and start with the context; in a game
//that keeps moves the moves' key (keyOf), the moves, and their start from the empty board.
static void checkPlayed(const vector<Board>& boards, const vector<Move>& moves, const Rgsc::Context& context) {
  Board board = boards[0];
  BoardHistory hist(board, board.nextPla(), Rules(), 0);
  for(const Move& m : moves)
    hist.makeBoardMoveAssumeLegal(board, m.loc, m.pla, NULL);
  const size_t start = moves.size() / 4, count = moves.size() - start;
  const vector<uint64_t> keys = Rgsc::playedKeys(hist, start, count, context);
  for(size_t k = 0; k < count; k++)
    check(keys[k] == (Rgsc::keepsStates() ? Rgsc::stateKey(boards[start + k], context) : Rgsc::keyOf(moves, start + k)),
          "the key of the played position after " + to_string(start + k) + " moves");
  for(size_t length : {start, moves.size() / 2, moves.size() - 1}) {
    const string where = "the played position after " + to_string(length) + " moves";
    const Rgsc::Entry e = Rgsc::playedEntry(hist, length, context);
    const vector<Move> prefix(moves.begin(), moves.begin() + length);
    if(Rgsc::keepsStates()) {
      const Rgsc::Entry want = Rgsc::stateEntry(boards[length], context, (int64_t)length);
      check(e.key == want.key && e.state == want.state && e.context == want.context &&
              e.turnNumber == want.turnNumber && e.moves.empty(),
            where + ": its state's entry");
    }
    else
      check(e.key == Rgsc::keyOf(moves, length) && sameMoves(e.moves, prefix) && e.state.empty(),
            where + ": its moves' entry");
    Sgf::PositionSample s;
    string why;
    check(Rgsc::startOf(e, boards[0].x_size, boards[0].y_size, s, why), where + ": a start: " + why);
    check(s.exactStart && s.hintLoc == Board::NULL_LOC && s.weight == 1.0 && s.trainingWeight == 1.0,
          where + ": an exact start");
    if(Rgsc::keepsStates()) {
      check(s.board.pos_hash == boards[length].pos_hash && s.nextPla == boards[length].nextPla() && s.moves.empty() &&
              s.initialTurnNumber == (int64_t)length,
            where + ": the start is the state, with its turn number");
      check(s.frozenContext && s.komi == context.komi && s.playoutDoublingAdvantagePla == context.advantagePla &&
              s.playoutDoublingAdvantage == context.advantage,
            where + ": the start keeps the context");
      Rgsc::Entry bad = e;
      bad.key ^= 1;
      check(!Rgsc::startOf(bad, boards[0].x_size, boards[0].y_size, s, why), where + ": a wrong key gives no start");
    }
    else
      check(s.board.pos_hash == Board(boards[0].x_size, boards[0].y_size).pos_hash && s.nextPla == P_BLACK &&
              sameMoves(s.moves, prefix) && s.initialTurnNumber == 0 && !s.frozenContext,
            where + ": the start is the empty board and the moves");
  }
}

//The defaults, on a game without a state's text. With a buffer's file (a copy of a run's), its lines of moves load.
static int checkDefaults(const string& dir, const string& bufferFile) {
  Board board;
  string why;
  GameRules::Position p;
  check(GameInterface::stateText<GameRules>(board.pos).empty(), "a game without a state's text gives none");
  check(!GameInterface::parseState<GameRules>("x", p, why) && !why.empty(), "and reads none, with a reason");
  Rgsc::Settings settings;
  settings.bufferFile = dir + "/rgsc_states_default.txt";
  FILE* f = fopen(settings.bufferFile.c_str(), "w");
  fprintf(f, "gamesFinished 3\nstate 1 %s 0000000000000001 0.5 0 1 4 0 0 0 x\nentry 0.5 3 2 101 1 201 1\n",
          GameRules::NAME);
  fclose(f);
  Rgsc::Buffer buffer(settings);
  buffer.load();
  check(buffer.size() == 1, "a game that keeps moves loads its lines of moves");
  check(buffer.summary().find("rejected at the load 1,") != string::npos, "and rejects a line of a state");
  Rand rand("testrgscstates defaults");
  vector<Board> boards;
  vector<Move> moves;
  for(int g = 0; g < 3; g++) {
    randomGame(rand, boards, moves);
    checkPlayed(boards, moves, Rgsc::Context());
  }
  if(!bufferFile.empty()) {
    Rgsc::Settings run;
    run.bufferFile = bufferFile;
    run.bufferSize = 1000000;
    Rgsc::Buffer loaded(run);
    loaded.load();
    check(loaded.size() > 0 && loaded.summary().find("rejected at the load 0,") != string::npos,
          "every line of " + bufferFile + " loads");
    printf("%s: %zu states; %s\n", bufferFile.c_str(), loaded.size(), loaded.summary().c_str());
  }
  printf("%s: the game keeps moves (%d failures)\n", failures == 0 ? "defaults checked" : "defaults FAILED", failures);
  return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  if(argc < 2) {
    fprintf(stderr, "usage: testrgscstates SCRATCH_DIR [GAMES] [--buffer FILE] [SETTING=VALUE ...]\n");
    return 2;
  }
  Board::initHash();
  const string dir = argv[1];
  int numGames = 30;
  string bufferFile, writeFile;
  for(int i = 2; i < argc; i++) {
    const string arg = argv[i];
    const size_t eq = arg.find('=');
    string why;
    if(arg == "--buffer" && i + 1 < argc)
      bufferFile = argv[++i];
    else if(arg == "--write" && i + 1 < argc)
      writeFile = argv[++i];
    else if(eq != string::npos) {
      if(!GameInterface::setSetting<GameRules>(arg.substr(0, eq), arg.substr(eq + 1), why)) {
        fprintf(stderr, "%s\n", why.c_str());
        return 2;
      }
    }
    else
      numGames = atoi(arg.c_str());
  }
  if(!Rgsc::keepsStates())
    return checkDefaults(dir, bufferFile);

  Rand rand("testrgscstates");

  // Every position of random games: the state's text gives the state back.
  int64_t positions = 0, pending = 0, withAdvantage = 0;
  vector<Board> boards;
  vector<Move> moves;
  vector<uint32_t> bits, bits2;
  vector<float> global, global2;
  GameRules::Position p;
  vector<Rgsc::Entry> written;  //--write: a state of each game
  for(int g = 0; g < numGames; g++) {
    randomGame(rand, boards, moves);
    const float komi = g % 3 == 0 ? 0.0f : (float)(rand.nextInt(-400, 400) * 0.5);
    const Rgsc::Context context = Rgsc::Context::of(
      komi, g % 3 == 1 ? P_WHITE : P_BLACK, g % 3 == 1 ? rand.nextDouble(3.0) : 0.0);
    withAdvantage += context.advantage != 0;
    //The state that --write keeps: in odd games one with a move pending, if any, else one a third into the game.
    size_t chosen = boards.size() / 3;
    for(size_t j = boards.size() / 4; g % 2 == 1 && j < boards.size(); j++)
      if(GameInterface::continuesMove<GameRules>(boards[j].pos)) {
        chosen = j;
        break;
      }
    Rgsc::Entry keep = Rgsc::stateEntry(boards[chosen], context, (int64_t)chosen);
    keep.regret = 1.0;
    keep.origin = Rgsc::ORIGIN_PLAYED;
    written.push_back(keep);
    set<uint64_t> keys;
    for(size_t j = 0; j < boards.size(); j++) {
      const Board& b = boards[j];
      const string where = "game " + to_string(g) + ", position " + to_string(j);
      const string text = GameInterface::stateText<GameRules>(b.pos);
      string why;
      if(!GameInterface::parseState<GameRules>(text, p, why)) {
        check(false, where + ": the state's text does not parse: " + why + ": " + text);
        continue;
      }
      Board r;
      r.setPosition(p);
      positions++;
      const bool continues = GameInterface::continuesMove<GameRules>(b.pos);
      pending += continues;
      check(r.pos_hash == b.pos_hash, where + ": the same position's hash");
      check(GameEncoding::input_hash(r.pos) == GameEncoding::input_hash(b.pos), where + ": the same input_hash");
      check(r.nextPla() == b.nextPla() && GameRules::moves_played(r.pos) == GameRules::moves_played(b.pos) &&
              GameInterface::continuesMove<GameRules>(r.pos) == continues && r.isGameOver() == b.isGameOver(),
            where + ": the same player, moves played, pending move and end");
      check(r.legalActions() == b.legalActions(), where + ": the same legal actions");
      check(GameInterface::stateText<GameRules>(r.pos) == text, where + ": the same text again");
      for(int s : SymmetryHelpers::ALLOWED)
        for(int version : {10, 11}) {
          inputsOf(b, context, s, bits, global, version);
          inputsOf(r, context, s, bits2, global2, version);
          check(bits == bits2 && global == global2,
                where + ": the same V" + to_string(version) + " inputs at symmetry " + to_string(s));
        }
      const uint64_t key = Rgsc::stateKey(b, context);
      check(Rgsc::stateKey(r, context) == key, where + ": the same state key");
      check(keys.insert(key).second, where + ": a key of its own in its game");
      Rgsc::Entry e = Rgsc::stateEntry(b, context, (int64_t)j);
      Board restored;
      check(Rgsc::restoreState(e, restored, why) && restored.pos_hash == b.pos_hash, where + ": the entry restores");
    }
  }
  check(pending > 0, "some positions have a pick waiting for its drop");
  printf("round trips: %lld positions of %d games (%lld with a move pending, %lld games with an advantage)\n",
         (long long)positions, numGames, (long long)pending, (long long)withAdvantage);
  {
    Rand played("testrgscstates played");
    for(int g = 0; g < 3; g++) {
      randomGame(played, boards, moves);
      checkPlayed(boards, moves, Rgsc::Context::of(7.0f * g - 3.5f, P_WHITE, g == 1 ? 1.25 : 0.0));
    }
  }
  if(!writeFile.empty()) {
    Rgsc::Settings settings;
    settings.bufferFile = writeFile;
    settings.bufferSize = (int)written.size();
    Rgsc::Buffer buffer(settings);
    int withPending = 0;
    for(const Rgsc::Entry& e : written) {
      Board r;
      string why;
      check(Rgsc::restoreState(e, r, why), "a state to write restores: " + why);
      withPending += GameInterface::continuesMove<GameRules>(r.pos);
      buffer.offer(e);
    }
    buffer.save();
    printf("%s: %zu states written, %d with a move pending\n", writeFile.c_str(), buffer.size(), withPending);
  }

  // The keys. An action that the same player plays twice in a row: the moves' key cancels it, the state's does not.
  // Two actions in either order: one moves' key, two states.
  {
    const Rgsc::Context context;
    bool repeated = false, swapped = false;
    for(int g = 0; g < 40 && !(repeated && swapped); g++) {
      randomGame(rand, boards, moves);
      for(size_t j = 0; j + 1 < boards.size() && !(repeated && swapped); j++) {
        const Board& b = boards[j];
        if(GameRules::turn_moves_left(b.pos) < 2 || GameInterface::continuesMove<GameRules>(b.pos))
          continue;
        vector<Loc> legal;
        b.getLegalMoveLocs(legal);
        const Player pla = b.nextPla();
        for(size_t a = 0; a < legal.size() && a < 60 && !(repeated && swapped); a++) {
          Board once = b;
          once.playMoveAssumeLegal(legal[a], pla);
          if(once.nextPla() != pla || GameInterface::continuesMove<GameRules>(once.pos))
            continue;
          if(!repeated && once.isLegal(legal[a], pla, false)) {
            Board twice = once;
            twice.playMoveAssumeLegal(legal[a], pla);
            vector<Move> m(moves.begin(), moves.begin() + j);
            const uint64_t before = Rgsc::keyOf(m, m.size());
            m.push_back(Move(legal[a], pla));
            m.push_back(Move(legal[a], pla));
            check(Rgsc::keyOf(m, m.size()) == before, "the moves' key cancels an action played twice");
            check(Rgsc::stateKey(twice, context) != Rgsc::stateKey(b, context) &&
                    Rgsc::stateKey(twice, context) != Rgsc::stateKey(once, context),
                  "the state key does not cancel it");
            repeated = true;
          }
          for(size_t c = a + 1; c < legal.size() && c < 60 && !swapped; c++) {
            if(!once.isLegal(legal[c], pla, false))
              continue;
            Board first = b, second = b;
            first.playMoveAssumeLegal(legal[a], pla);
            first.playMoveAssumeLegal(legal[c], pla);
            second.playMoveAssumeLegal(legal[c], pla);
            if(second.nextPla() != pla || GameInterface::continuesMove<GameRules>(second.pos) ||
               !second.isLegal(legal[a], pla, false))
              continue;
            second.playMoveAssumeLegal(legal[a], pla);
            const vector<Move> ac = {Move(legal[a], pla), Move(legal[c], pla)};
            const vector<Move> ca = {Move(legal[c], pla), Move(legal[a], pla)};
            check(Rgsc::keyOf(ac, 2) == Rgsc::keyOf(ca, 2), "the moves' key ignores the order");
            check(Rgsc::stateKey(first, context) != Rgsc::stateKey(second, context),
                  "two actions in either order: two states, two state keys");
            swapped = true;
          }
        }
      }
    }
    check(repeated && swapped, "the search found an action played twice and two actions in either order");

    // Two edits (a pick, then its drop) in either order: when both orders give one state, one key.
    bool commuted = false;
    for(int g = 0; g < 40 && !commuted; g++) {
      randomGame(rand, boards, moves);
      for(size_t j = 0; j + 1 < boards.size() && !commuted; j++) {
        const Board& b = boards[j];
        if(GameRules::turn_moves_left(b.pos) < 2 || GameInterface::continuesMove<GameRules>(b.pos))
          continue;
        const Player pla = b.nextPla();
        //Edits of b: a pick and a drop, the first, the middle and the last drop of each pick.
        vector<pair<Loc, Loc>> edits;
        vector<Loc> legal, drops;
        b.getLegalMoveLocs(legal);
        for(Loc pick : legal) {
          Board picked = b;
          picked.playMoveAssumeLegal(pick, pla);
          if(!GameInterface::continuesMove<GameRules>(picked.pos))
            continue;
          picked.getLegalMoveLocs(drops);
          for(size_t d : {(size_t)0, drops.size() / 2, drops.size() - 1})
            if(d < drops.size() && (edits.empty() || edits.back() != make_pair(pick, drops[d])))
              edits.push_back({pick, drops[d]});
        }
        for(size_t x = 0; x < edits.size() && !commuted; x++)
          for(size_t y = x + 1; y < edits.size() && !commuted; y++) {
            auto play = [&](const vector<pair<Loc, Loc>>& order, Board& out) {
              out = b;
              for(const auto& edit : order) {
                if(out.nextPla() != pla || !out.isLegal(edit.first, pla, false))
                  return false;
                out.playMoveAssumeLegal(edit.first, pla);
                if(!out.isLegal(edit.second, pla, false))
                  return false;
                out.playMoveAssumeLegal(edit.second, pla);
              }
              return true;
            };
            Board xy, yx;
            if(!play({edits[x], edits[y]}, xy) || !play({edits[y], edits[x]}, yx) || xy.pos_hash != yx.pos_hash)
              continue;
            check(GameInterface::stateText<GameRules>(xy.pos) == GameInterface::stateText<GameRules>(yx.pos),
                  "two edits in either order: one state's text");
            check(Rgsc::stateKey(xy, context) == Rgsc::stateKey(yx, context), "and one key");
            commuted = true;
          }
      }
    }
    check(commuted, "the search found two edits that commute");

    // The context.
    const Board& b = boards[boards.size() / 2];
    const uint64_t key = Rgsc::stateKey(b, Rgsc::Context::of(0.0f, C_EMPTY, 0.0));
    check(Rgsc::stateKey(b, Rgsc::Context::of(-0.0f, P_WHITE, -0.0)) == key,
          "-0 and a player of no advantage: the same key");
    check(Rgsc::stateKey(b, Rgsc::Context::of(0.5f, C_EMPTY, 0.0)) != key, "the komi changes the key");
    const uint64_t white = Rgsc::stateKey(b, Rgsc::Context::of(0.0f, P_WHITE, 1.0));
    check(white != key && Rgsc::stateKey(b, Rgsc::Context::of(0.0f, P_BLACK, 1.0)) != white,
          "the advantage and its player change the key");
  }

  // The records: a garbled text, another state's text and a wrong key are rejected.
  {
    randomGame(rand, boards, moves);
    const Board& b = boards[boards.size() / 2];
    const Rgsc::Context context = Rgsc::Context::of(3.5f, C_EMPTY, 0.0);
    const Rgsc::Entry good = Rgsc::stateEntry(b, context, 7);
    Board r;
    string why;
    check(Rgsc::restoreState(good, r, why), "a good record restores: " + why);
    Rgsc::Entry bad = good;
    bad.state = withoutLast(withoutLast(good.state));
    check(!Rgsc::restoreState(bad, r, why), "a truncated text is rejected");
    bad.state = good.state.substr(0, good.state.rfind(' '));
    check(!Rgsc::restoreState(bad, r, why), "a text without its footprints is rejected");
    bad.state = good.state;
    bad.state.replace(bad.state.find(' '), 1, " 9");  //the first score, 9 before its digits
    check(!Rgsc::restoreState(bad, r, why) && why.find("key") != string::npos,
          "another state's text is rejected: " + why);
    bad = good;
    bad.key ^= 1;
    check(!Rgsc::restoreState(bad, r, why), "a wrong key is rejected");
    bad = good;
    bad.context = Rgsc::Context::of(4.0f, C_EMPTY, 0.0);
    check(!Rgsc::restoreState(bad, r, why), "another context is rejected");
  }

  // The buffer: duplicates and collisions, a save and a load, and the lines that a load rejects.
  {
    randomGame(rand, boards, moves);
    Rgsc::Settings settings;
    settings.enabled = true;
    settings.bufferFile = dir + "/rgsc_states.txt";
    settings.bufferSize = 4;
    Rgsc::Buffer buffer(settings);
    vector<Rgsc::Entry> entries;
    for(int i = 0; i < 3; i++) {
      //The third with an advantage of every digit, which the file must keep for the key.
      const Rgsc::Context context =
        Rgsc::Context::of((float)i - 0.5f, i == 2 ? P_BLACK : C_EMPTY, i == 2 ? 0.123456789012345678 : 0.0);
      Rgsc::Entry e = Rgsc::stateEntry(boards[10 + 20 * i], context, 10 + 20 * i);
      e.regret = 1.0 + 0.01 * i;  //close, so that the draws (R^10) reach each one
      e.origin = i == 0 ? Rgsc::ORIGIN_TREE : Rgsc::ORIGIN_PLAYED;
      entries.push_back(e);
      buffer.offer(e);
    }
    check(buffer.size() == 3, "the buffer takes three states");
    buffer.offer(entries[1]);
    check(buffer.size() == 3 && buffer.summary().find("key collisions 0") != string::npos,
          "a state again: a duplicate");
    Rgsc::Entry forged = Rgsc::stateEntry(boards[15], entries[0].context, 15);
    forged.key = entries[0].key;
    buffer.offer(forged);
    check(buffer.size() == 3 && buffer.summary().find("key collisions 1") != string::npos,
          "the same key with another state: a collision, refused");
    check(buffer.countFinishedGame() == 0, "finished games count up");
    buffer.save();
    // The good lines, then lines that a load rejects.
    FILE* f = fopen(settings.bufferFile.c_str(), "a");
    const Rgsc::Entry& e = entries[0];
    auto line = [&](int version, const string& game, uint64_t key, const string& state) {
      fprintf(f, "state %d %s %s 0.5 0 1 10 -0.5 0 0 %s\n", version, game.c_str(),
              Global::uint64ToHexString(key).c_str(), state.c_str());
    };
    line(2, GameRules::NAME, e.key, e.state);
    line(1, "another", e.key, e.state);
    line(1, GameRules::NAME, e.key ^ 2, e.state);
    line(1, GameRules::NAME, e.key, withoutLast(e.state));
    fprintf(f, "entry 0.5 3 2 101 1 201 1\n");
    fclose(f);
    Rgsc::Buffer loaded(settings);
    loaded.load();
    check(loaded.size() == 3, "a load gets the three states");
    check(loaded.summary().find("rejected at the load 5,") != string::npos,
          "and rejects another version, another game, a wrong key, a garbled state and moves: " + loaded.summary());
    check(loaded.countFinishedGame() == 1, "a load gets the finished games");
    Rand draw("testrgscstates draw");
    Rgsc::Settings always = settings;
    always.startProb = 1.0;
    Rgsc::Buffer drawn(always);
    drawn.load();
    set<uint64_t> seen;
    for(int i = 0; i < 2000 && seen.size() < 3; i++) {
      Rgsc::Entry out;
      check(drawn.maybeSample(draw, out), "a full draw");
      for(const Rgsc::Entry& want : entries)
        if(want.key == out.key) {
          check(out.state == want.state && out.context == want.context && out.turnNumber == want.turnNumber &&
                  out.origin == want.origin,
                "a loaded state keeps its text, context, turn number and origin");
          seen.insert(out.key);
        }
    }
    check(seen.size() == 3, "every state loaded draws");
    drawn.reject(entries[1].key);
    check(drawn.size() == 2 && !drawn.contains(entries[1].key) &&
            drawn.summary().find("at a restart 1;") != string::npos,
          "a state rejected at its restart leaves the buffer");
  }

  // The regret targets by kind of action: a move of one action, a pick (its next action continues it), a drop.
  {
    Rgsc::Settings settings;
    Rgsc::Buffer buffer(settings);
    buffer.addRegrets({1.0f, 2.0f, 3.0f, 4.0f}, {false, false, true, false});
    buffer.addRegrets({5.0f}, {});
    const string want = "moves of one action 3 mean 3.33333, first actions of a move 1 mean 2, later actions 1 mean 3";
    check(buffer.summary().find(want) != string::npos, "the regret by kind of action: " + buffer.summary());
  }

  // A self-play's buffer: every entry restores, with its text.
  if(!bufferFile.empty()) {
    Rgsc::Settings settings;
    settings.bufferFile = bufferFile;
    settings.startProb = 1.0;
    settings.bufferSize = 1000000;
    Rgsc::Buffer buffer(settings);
    buffer.load();
    const string summary = buffer.summary();
    check(buffer.size() > 0 && summary.find("rejected at the load 0,") != string::npos,
          "every entry of " + bufferFile + " loads: " + summary);
    //The draws (by regret) restore each state with its text.
    Rand draw("testrgscstates buffer");
    set<uint64_t> restored;
    int pendingStates = 0;
    for(int i = 0; i < 20000 && restored.size() < buffer.size(); i++) {
      Rgsc::Entry e;
      if(!buffer.maybeSample(draw, e) || restored.count(e.key) > 0)
        continue;
      Board r;
      string why;
      //The text of an older version restores as the state's text now, which extends it.
      const bool ok = Rgsc::restoreState(e, r, why);
      check(ok && GameInterface::stateText<GameRules>(r.pos).compare(0, e.state.size(), e.state) == 0,
            "an entry of the file restores with its text: " + why);
      pendingStates += GameInterface::continuesMove<GameRules>(r.pos);
      restored.insert(e.key);
    }
    printf("%s: %zu states loaded, %zu drawn and restored (%d with a move pending); %s\n", bufferFile.c_str(),
           buffer.size(), restored.size(), pendingStates, summary.c_str());
  }

  printf("%s (%d failures)\n", failures == 0 ? "all RGSC state checks passed" : "RGSC state checks FAILED", failures);
  return failures == 0 ? 0 : 1;
}
