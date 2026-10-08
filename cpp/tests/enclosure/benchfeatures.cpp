// Enclosure fork (owner, 2026-10-06; CPU trick 4): where feature_planes, legal_actions and enclosed_area spend their
// time, for attribution under perf (not for speed decisions: those are the self-play benchmark's).
//
//   benchfeatures INPUTS_FILE [ROUNDS]
//
// Replays each position of INPUTS_FILE (tests/data/enclosure_inputs.txt) and calls feature_planes and legal_actions
// on every one, ROUNDS times (default 200), printing microseconds per call. Header-only (game/enclosure), so it builds
// alone; for attribution build it without inlining, e.g.
//   g++ -O2 -g -fno-inline -fno-omit-frame-pointer -std=c++17 -I. tests/enclosure/benchfeatures.cpp -o benchfeatures
// and run it under `perf record -g`.

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../../game/enclosure/features.h"

int main(int argc, char** argv) {
  if(argc < 2) {
    fprintf(stderr, "usage: benchfeatures INPUTS_FILE [ROUNDS]\n");
    return 2;
  }
  const int rounds = argc > 2 ? std::atoi(argv[2]) : 200;
  //DUMP_FILE (optional third argument): every position's feature planes, legal actions and enclosed areas written
  //there, for comparing two builds byte for byte.
  const char* dumpPath = argc > 3 ? argv[3] : nullptr;
  std::ifstream in(argv[1]);
  std::vector<enclosure::Position> positions;
  std::string line;
  while(std::getline(in, line)) {
    std::istringstream head(line);
    std::string word, label;
    int n;
    if(!(head >> word >> label >> n) || word != "position")
      continue;
    enclosure::Position p;
    for(int i = 0; i < n; i++) {
      int a;
      head >> a;
      p.play(a);
    }
    if(!p.terminal())
      positions.push_back(p);
  }
  std::vector<uint8_t> planes(enclosure::N_POINTS * enclosure::FEATURE_PLANES);
  std::vector<int> legal;
  if(dumpPath != nullptr) {
    FILE* f = fopen(dumpPath, "wb");
    uint8_t inside[enclosure::N_POINTS];
    float income[enclosure::N_POINTS];
    for(const enclosure::Position& p : positions) {
      enclosure::feature_planes(p, planes.data());
      fwrite(planes.data(), 1, planes.size(), f);
      p.legal_actions(legal);
      fwrite(legal.data(), sizeof(int), legal.size(), f);
      for(int k = 0; k < 2; k++) {
        const double area = enclosure::enclosed_area(p.edges[k], inside, income);
        fwrite(&area, sizeof(area), 1, f);
        fwrite(inside, 1, sizeof(inside), f);
        fwrite(income, sizeof(float), enclosure::N_POINTS, f);
      }
    }
    fclose(f);
  }
  long checksum = 0;
  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  for(int r = 0; r < rounds; r++)
    for(const enclosure::Position& p : positions) {
      enclosure::feature_planes(p, planes.data());
      checksum += planes[r % planes.size()];
    }
  auto t1 = clock::now();
  for(int r = 0; r < rounds; r++)
    for(const enclosure::Position& p : positions) {
      p.legal_actions(legal);
      checksum += (long)legal.size();
    }
  auto t2 = clock::now();
  const double calls = double(rounds) * positions.size();
  printf("%zu positions x %d rounds: feature_planes %.1f us, legal_actions %.1f us (checksum %ld)\n", positions.size(),
         rounds, std::chrono::duration<double, std::micro>(t1 - t0).count() / calls,
         std::chrono::duration<double, std::micro>(t2 - t1).count() / calls, checksum);
  return 0;
}
