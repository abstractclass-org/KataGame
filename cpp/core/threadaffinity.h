#ifndef CORE_THREADAFFINITY_H_
#define CORE_THREADAFFINITY_H_

#include <string>

//Enclosure fork (owner, 2026-10-05): CPU affinity for self-play's two kinds of threads.
//With 384 game threads and 4 NN server threads on 16 logical CPUs, the server threads (every evaluation passes
//through them) sat runnable without a CPU 36% of the time while about 22% of the CPU stayed idle: a server thread
//wakes about 60 game threads after each batch and they queue on its CPU. Pinning the server threads to their own
//CPUs took that to 3% (v2/docs/investigations/selfplay-throughput-2026-10-05.md, sections 8-13). Off unless set (Linux only).
namespace ThreadAffinity {
  //"0-11", "12-15", "0,2,4-7"; empty clears. Called by selfplay before any NNEvaluator spawns its threads.
  void setServerCpus(const std::string& cpuList);
  void setGameCpus(const std::string& cpuList);
  //Pin the calling thread to the server or game list (if set), and name it for profilers ("nnserver", "gameloop").
  void applyServer();
  void applyGame();
  //Whether a list parses; the reason in error if not.
  bool parse(const std::string& cpuList, std::string& error);
}

#endif  // CORE_THREADAFFINITY_H_
