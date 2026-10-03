// Enclosure fork: the territory income (enclosure::enclosed_area's `income`, V2's territory-income target)
// against the Python reference.
//
//   testincome tests/data/enclosure_inputs.txt tests/data/enclosure_income.txt
//
// For each position of the input fixture, replays its moves and compares each player's income per lattice point
// (the enclosed area inside the unit square centred on the point) with the reference (planes.income_points,
// tools/export_income_fixture.py), and checks that it sums to the enclosed area.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

#include "../../game/boardhistory.h"

using namespace std;

int main(int argc, char** argv) {
  if(argc != 3) {
    fprintf(stderr, "usage: testincome INPUTS_FILE INCOME_FILE\n");
    return 2;
  }
  Board::initHash();
  const int S = enclosure::SIZE;
  std::map<string, vector<float>> reference[2];
  {
    ifstream in(argv[2]);
    string line;
    while(getline(in, line)) {
      istringstream row(line);
      string word, label;
      int player, count;
      if(!(row >> word >> label >> player >> count) || word != "income")
        continue;
      vector<float> values(S * S, 0.0f);
      for(int i = 0; i < count; i++) {
        int q;
        double area;
        row >> q >> area;
        values[q] = (float)area;
      }
      reference[player][label] = values;
    }
  }
  ifstream in(argv[1]);
  string line;
  int positions = 0, failures = 0;
  double worst = 0.0;
  auto fail = [&](const string& what) {
    if(failures < 20)
      fprintf(stderr, "FAIL %s\n", what.c_str());
    failures++;
  };
  while(getline(in, line)) {
    istringstream head(line);
    string word, label;
    int n;
    if(!(head >> word >> label >> n) || word != "position")
      continue;
    positions++;
    Board board;
    BoardHistory hist(board, P_BLACK, Rules(), 0);
    for(int i = 0; i < n; i++) {
      int action;
      head >> action;
      Loc loc = Board::moveLocOfAction(action);
      hist.makeBoardMoveAssumeLegal(board, loc, hist.presumedNextMovePla, NULL);
    }
    for(int player = 0; player < 2; player++) {
      auto it = reference[player].find(label);
      if(it == reference[player].end()) {
        fail(label + ": no reference");
        continue;
      }
      float income[enclosure::SIZE * enclosure::SIZE];
      const double area = enclosure::enclosed_area(board.pos.edges[player], nullptr, income);
      double sum = 0.0;
      for(int x = 0; x < S; x++)
        for(int y = 0; y < S; y++) {
          const float got = income[x * S + y], want = it->second[y * S + x];
          sum += got;
          worst = std::max(worst, (double)std::fabs(got - want));
          if(std::fabs(got - want) > 1e-4)
            fail(label + " player " + Global::intToString(player) + " x " + Global::intToString(x) + " y " +
                 Global::intToString(y) + ": " + Global::floatToString(got) + ", want " + Global::floatToString(want));
        }
      if(std::fabs(sum - area) > 1e-3)
        fail(label + " player " + Global::intToString(player) + ": income sums to " + Global::doubleToString(sum) +
             ", area " + Global::doubleToString(area));
    }
  }
  printf("%d positions, largest difference %.3g, %d failures\n", positions, worst, failures);
  return failures == 0 ? 0 : 1;
}
