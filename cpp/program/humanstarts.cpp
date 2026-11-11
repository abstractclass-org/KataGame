#include "../program/humanstarts.h"

#include <fstream>
#include <sstream>
#include <unordered_set>

#include "../core/global.h"
#include "../core/test.h"
#include "../dataio/trainingwrite.h"
#include "../external/nlohmann_json/json.hpp"

using namespace std;

const char* HumanStarts::FORMAT = "human-starts";

//The rows' C74 and the SGF record's gtype read the same code (FinishedGameData::rgscSource).
static_assert((int)Rgsc::SOURCE_HUMAN == FinishedGameData::SOURCE_HUMAN_START, "one code for a human start");

HumanStarts::Settings HumanStarts::Settings::load(ConfigParser& cfg) {
  Settings s;
  if(cfg.contains("humanStartProb"))
    s.prob = cfg.getDouble("humanStartProb", 0.0, 1.0);
  //Read the file's key either way, so that a config with the starts off raises no unused-key warning.
  if(s.prob > 0.0 || cfg.contains("humanStartsFile"))
    s.file = cfg.getString("humanStartsFile");
  if(s.prob > 0.0 && !Rgsc::keepsStates())
    throw StringError("Human starts: this game gives no state's text (GameInterface::stateText), so it has no file of "
                      "states (humanStartProb must be 0)");
  return s;
}

Rgsc::Context HumanStarts::context() {
  return Rgsc::Context::of(0.0f, C_EMPTY, 0.0);
}

void HumanStarts::Starts::load(const string& file) {
  ifstream in(file);
  if(!in)
    throw StringError("Human starts: cannot read " + file);
  ostringstream text;
  text << in.rdbuf();
  try {
    loadText(text.str());
  }
  catch(const StringError& e) {
    throw StringError(string(e.what()) + " (" + file + ")");
  }
}

void HumanStarts::Starts::loadText(const string& text) {
  entries.clear();
  numRejected = 0;
  numDuplicates = 0;
  firstRejections.clear();
  istringstream in(text);
  string line;
  bool header = false;
  int64_t lineNumber = 0;
  unordered_set<uint64_t> keys;
  Board board;
  auto reject = [&](const string& why) {
    numRejected++;
    if(firstRejections.size() < 5)
      firstRejections.push_back("line " + Global::int64ToString(lineNumber) + ": " + why);
  };
  while(getline(in, line)) {
    lineNumber++;
    if(Global::trim(line).empty())
      continue;
    nlohmann::json j;
    try {
      j = nlohmann::json::parse(line);
    }
    catch(const nlohmann::json::exception& e) {
      if(!header)
        throw StringError("Human starts: the first line is not the header: " + string(e.what()));
      reject(string("not JSON: ") + e.what());
      continue;
    }
    if(!header) {
      if(!j.is_object() || !j.contains("format") || !j["format"].is_string() || j["format"].get<string>() != FORMAT ||
         !j.contains("version") || !j["version"].is_number_integer() || j["version"].get<int>() != VERSION)
        throw StringError(string("Human starts: the first line is not the header of the format ") + FORMAT +
                          ", version " + Global::intToString(VERSION));
      header = true;
      continue;
    }
    if(!j.is_object() || !j.contains("state") || !j["state"].is_string() || !j.contains("turn") ||
       !j["turn"].is_number_integer() || j["turn"].get<int64_t>() < 0) {
      reject("not a start (a state's text and its turn number)");
      continue;
    }
    const string state = j["state"].get<string>();
    static thread_local GameRules::Position p;
    string why;
    if(!GameInterface::parseState<GameRules>(state, p, why)) {
      reject("the state does not parse: " + why);
      continue;
    }
    board.setPosition(p);
    Rgsc::Entry e = Rgsc::stateEntry(board, context(), j["turn"].get<int64_t>());
    //The state that the text gives must write the same text: the whole state came back.
    if(e.state != state) {
      reject("the state's text comes back as another: " + e.state.substr(0, 120));
      continue;
    }
    if(!keys.insert(e.key).second) {
      numDuplicates++;
      continue;
    }
    entries.push_back(e);
  }
  if(!header)
    throw StringError("Human starts: no header line");
  if(entries.empty())
    throw StringError("Human starts: no state kept (" + Global::int64ToString(numRejected) + " rejected)");
}

string HumanStarts::Starts::summary() const {
  string s = Global::uint64ToString(entries.size()) + " states kept, " + Global::int64ToString(numRejected) +
             " rejected, " + Global::int64ToString(numDuplicates) + " duplicates left out";
  for(const string& why : firstRejections)
    s += "; " + why;
  return s;
}

void HumanStarts::drawAtStart(Rand& rand, Sgf::PositionSample& start) {
  if constexpr(GameInterface::rules::HasDrawStart<GameRules>::value) {
    GameRules::Position p = start.board.pos;
    GameInterface::drawStart<GameRules>(p, [&rand]() { return rand.nextUInt64(); });
    start.board.setPosition(p);
    start.nextPla = start.board.nextPla();
  }
  else
    (void)rand;
}

bool HumanStarts::Starts::draw(
  Rand& rand, int xSize, int ySize, Rgsc::Entry& entry, Sgf::PositionSample& start, string& why
) const {
  testAssert(!entries.empty());
  entry = entries[rand.nextUInt((uint32_t)entries.size())];
  if(!Rgsc::startOf(entry, xSize, ySize, start, why))
    return false;
  drawAtStart(rand, start);
  return true;
}
