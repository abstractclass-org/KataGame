// KataGame fork (2026-11-11, abscls-7ald): self-play's starts from a file of people's positions
// (program/humanstarts.h), on a build of a game that gives its whole state as a text (GameInterface::stateText:
// Blueprints).
//
//   testhumanstarts [GAMES] [--file FILE] [SETTING=VALUE ...]
//
// SETTING=VALUE: the game's settings (GameInterface::setSetting), such as viewOrientation=randomGame. The test plays
// GAMES random games (default 20) of the legal actions from the game's start, with the game's draws at its start, and
// writes a file of states from them: the header, then a state after each 3rd action that does not wait for the next
// action of its move (as a site's move ends), with its turn number. It adds lines that the load must reject or leave
// out: a state's text with a character changed, a line that is not JSON, a start without its turn number, a negative
// turn number, and a state already in the file. The checks:
// - The settings: without humanStartProb, none; a probability with its file; the file's key read either way; a
//   probability above 1 refused.
// - The load: it keeps every good state once, in the file's order, with its text and its turn number, and the key of
//   each state after its load is the key of the position that the random game reached, with the komi 0 and no
//   asymmetric playouts (Rgsc::stateKey, as RGSC's load checks its entries). It counts the rejected lines and the
//   duplicate. A file without the header, with a header of another format or version, or with no good state, is
//   refused.
// - The draw: 200 draws a state, uniformly (each state's count within 6 standard deviations of its mean, and every
//   state drawn), each start exact (RGSC's exact start: the state with no moves, its turn number, the komi 0 frozen and
//   no asymmetric playouts). The start's state is the drawn state's, but for the game's draws at a start: with no such
//   draw (viewOrientation 0, the default), its key is the entry's; under viewOrientation=randomGame the view's
//   orientation is drawn, and the starts of each state take more than one orientation, a third each over all draws
//   (within 6 standard deviations), and each start's state text gives that start's state back.
// With --file FILE (a file of the engine repository's games/blueprints/tools/human_starts.py), the test loads it and
// prints the summary: every line must load, under the settings given (a run's viewOrientation).
// On a build of a game without a state's text (Enclosure), the test checks that a probability above 0 is refused.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "../../core/config_parser.h"
#include "../../core/rand.h"
#include "../../external/nlohmann_json/json.hpp"
#include "../../program/humanstarts.h"

using namespace std;

static int failures = 0;
static void check(bool ok, const string& what) {
  if(!ok) {
    if(failures < 20)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  }
}

static HumanStarts::Settings settingsOf(const map<string, string>& kvs, bool& refused) {
  ConfigParser cfg(kvs);
  refused = false;
  HumanStarts::Settings s;
  try {
    s = HumanStarts::Settings::load(cfg);
  }
  catch(const StringError&) {
    refused = true;
  }
  return s;
}

static void readText(HumanStarts::Starts& starts, const string& text) {
  istringstream in(text);
  starts.read(in);
}

static bool refusedText(const string& text, const string& what) {
  HumanStarts::Starts starts;
  try {
    readText(starts, text);
  }
  catch(const StringError&) {
    return true;
  }
  fprintf(stderr, "not refused: %s\n", what.c_str());
  return false;
}

static string startLine(const string& state, int64_t turn) {
  nlohmann::json j;
  j["state"] = state;
  j["turn"] = turn;
  j["game"] = "test";
  return j.dump();
}

int main(int argc, char** argv) {
  Board::initHash();
  int numGames = 20;
  string file;
  bool randomGame = false;
  for(int i = 1; i < argc; i++) {
    const string arg = argv[i];
    const size_t eq = arg.find('=');
    string why;
    if(arg == "--file" && i + 1 < argc)
      file = argv[++i];
    else if(eq != string::npos) {
      if(!GameInterface::setSetting<GameRules>(arg.substr(0, eq), arg.substr(eq + 1), why)) {
        fprintf(stderr, "%s\n", why.c_str());
        return 2;
      }
      randomGame = randomGame || arg == "viewOrientation=randomGame";
    }
    else
      numGames = atoi(arg.c_str());
  }

  bool refused;
  if(!Rgsc::keepsStates()) {
    settingsOf({{"humanStartProb", "0.1"}, {"humanStartsFile", "x"}}, refused);
    check(refused, "a game without a state's text refuses a probability above 0");
    HumanStarts::Settings off = settingsOf({}, refused);
    check(!refused && !off.enabled(), "and has none by default");
    printf("%s: the game gives no state's text (%d failures)\n", failures == 0 ? "defaults checked" : "defaults FAILED",
           failures);
    return failures == 0 ? 0 : 1;
  }

  // The settings.
  {
    HumanStarts::Settings s = settingsOf({}, refused);
    check(!refused && !s.enabled() && s.prob == 0.0 && s.file.empty(), "no starts without humanStartProb");
    s = settingsOf({{"humanStartProb", "0.1"}, {"humanStartsFile", "/x/starts.jsonl"}}, refused);
    check(!refused && s.enabled() && s.prob == 0.1 && s.file == "/x/starts.jsonl", "a probability with its file");
    s = settingsOf({{"humanStartProb", "0"}, {"humanStartsFile", "/x/starts.jsonl"}}, refused);
    check(!refused && !s.enabled() && s.file == "/x/starts.jsonl", "the file's key read with the starts off");
    settingsOf({{"humanStartProb", "0.1"}}, refused);
    check(refused, "a probability without its file is refused");
    settingsOf({{"humanStartProb", "1.5"}, {"humanStartsFile", "x"}}, refused);
    check(refused, "a probability above 1 is refused");
  }

  // The file of the random games' states.
  Rand rand("testhumanstarts");
  vector<string> texts;
  vector<int64_t> turns;
  vector<uint64_t> keys;
  set<string> seen;
  ostringstream out;
  out << "{\"format\":\"human-starts\",\"version\":1,\"source\":\"testhumanstarts\"}\n";
  for(int g = 0; g < numGames; g++) {
    Board board;
    GameRules::Position start = board.pos;
    GameInterface::drawStart<GameRules>(start, [&rand]() { return rand.nextUInt64(); });
    board.setPosition(start);
    vector<Loc> legal;
    for(int64_t turn = 0; !board.isGameOver(); turn++) {
      if(turn > 0 && turn % 3 == 0 && !GameInterface::continuesMove<GameRules>(board.pos)) {
        const string text = GameInterface::stateText<GameRules>(board.pos);
        if(seen.insert(text).second) {
          texts.push_back(text);
          turns.push_back(turn);
          keys.push_back(Rgsc::stateKey(board, HumanStarts::context()));
          out << startLine(text, turn) << "\n";
        }
      }
      board.getLegalMoveLocs(legal);
      if(legal.empty())
        break;
      const Loc loc = legal[rand.nextUInt((uint32_t)legal.size())];
      board.playMoveAssumeLegal(loc, board.nextPla());
    }
  }
  check(texts.size() >= 20, "the random games give at least 20 states");
  // The lines that the load rejects or leaves out. The garbled text is cut in its middle, with a character that no
  // state's text holds: a changed digit could give another state, which the load cannot tell from a good one.
  const string garbled = texts[0].substr(0, texts[0].size() / 2) + "#";
  out << startLine(garbled, 3) << "\n";
  out << "not JSON\n";
  out << "{\"state\":" << nlohmann::json(texts[1]).dump() << "}\n";
  out << startLine(texts[2], -1) << "\n";
  out << startLine(texts[3], turns[3]) << "\n";  // a duplicate
  out << "\n";
  const string text = out.str();

  HumanStarts::Starts starts;
  readText(starts, text);
  const vector<Rgsc::Entry>& states = starts.states();
  check(states.size() == texts.size(), "every good state is kept once: " + Global::uint64ToString(states.size()) +
        " of " + Global::uint64ToString(texts.size()));
  check(starts.rejected() == 4, "the 4 bad lines are rejected: " + Global::int64ToString(starts.rejected()));
  check(starts.duplicates() == 1, "the duplicate is left out");
  for(size_t i = 0; i < states.size() && i < texts.size(); i++) {
    check(states[i].state == texts[i], "state " + Global::uint64ToString(i) + " keeps its text");
    check(states[i].turnNumber == turns[i], "state " + Global::uint64ToString(i) + " keeps its turn number");
    check(states[i].key == keys[i], "state " + Global::uint64ToString(i) + "'s key after its load is its position's");
    check(states[i].context == HumanStarts::context(), "state " + Global::uint64ToString(i) + " has the komi 0 and "
          "no asymmetric playouts");
  }
  printf("loaded: %s\n", starts.summary().c_str());

  // The refused files.
  const string body = text.substr(text.find('\n') + 1);
  check(refusedText(body, "no header"), "a file without the header is refused");
  check(refusedText("{\"format\":\"other\",\"version\":1}\n" + body, "another format"), "another format is refused");
  check(refusedText("{\"format\":\"human-starts\",\"version\":2}\n" + body, "another version"),
        "another version is refused");
  check(refusedText("{\"format\":\"human-starts\",\"version\":1}\nnot JSON\n", "no good state"),
        "a file with no good state is refused");

  // The draws.
  const int n = (int)states.size();
  const int draws = 200 * n;
  vector<int> counts(n, 0);
  vector<set<string>> startTexts(n);
  map<string, int> index;
  for(int i = 0; i < n; i++)
    index[states[i].state] = i;
  int64_t asWritten = 0;
  Rand drawRand("testhumanstarts draws");
  const Board standard;
  GameRules::Position p;
  for(int d = 0; d < draws; d++) {
    Rgsc::Entry entry;
    Sgf::PositionSample start;
    string why;
    const bool ok = starts.draw(drawRand, standard.x_size, standard.y_size, entry, start, why);
    check(ok, "a drawn state comes back: " + why);
    if(!ok || !index.count(entry.state))
      continue;
    const int i = index[entry.state];
    counts[i]++;
    check(start.exactStart && start.frozenContext && start.moves.empty() && start.komi == 0.0f &&
          start.playoutDoublingAdvantage == 0.0 && start.playoutDoublingAdvantagePla == C_EMPTY &&
          start.initialTurnNumber == entry.turnNumber && start.nextPla == start.board.nextPla(),
          "a start is exact, with its turn number, the komi 0 frozen and no asymmetric playouts");
    const string startText = GameInterface::stateText<GameRules>(start.board.pos);
    startTexts[i].insert(startText);
    asWritten += startText == entry.state;
    if(!randomGame)
      check(Rgsc::stateKey(start.board, HumanStarts::context()) == entry.key && startText == entry.state,
            "with no draw at a start, the start is the state itself");
    string parseWhy;
    check(GameInterface::parseState<GameRules>(startText, p, parseWhy) &&
            GameInterface::stateText<GameRules>(p) == startText,
          "a start's text gives its state back: " + parseWhy);
  }
  int unevenStates = 0, turned = 0;
  for(int i = 0; i < n; i++) {
    unevenStates += std::abs(counts[i] - 200) > 6 * std::sqrt(200.0) || counts[i] == 0;
    turned += startTexts[i].size() > 1;
  }
  check(unevenStates == 0, "the draws are uniform over the states: " + Global::intToString(unevenStates) +
        " states off");
  if(randomGame) {
    // A third of the draws keep the state's orientation (the file's, 0), within 6 standard deviations.
    const double expect = draws / 3.0, sd = std::sqrt(draws * (1.0 / 3) * (2.0 / 3));
    check(std::abs(asWritten - expect) <= 6 * sd, "a third of the starts keep the file's orientation: " +
          Global::int64ToString(asWritten) + " of " + Global::intToString(draws));
    check(turned == n, "every state starts in more than one orientation: " + Global::intToString(turned) + " of " +
          Global::intToString(n));
  }
  else
    check(asWritten == draws, "with no draw at a start, every start keeps the file's state");
  printf("draws: %d over %d states; %lld kept the file's text, %d states started in more than one orientation\n",
         draws, n, (long long)asWritten, turned);

  if(!file.empty()) {
    HumanStarts::Starts run;
    run.load(file);
    check(run.rejected() == 0 && run.duplicates() == 0, "every line of " + file + " loads");
    printf("%s: %s\n", file.c_str(), run.summary().c_str());
  }
  printf("%s (%d failures)\n", failures == 0 ? "human starts checked" : "human starts FAILED", failures);
  return failures == 0 ? 0 : 1;
}
