#include "../core/threadaffinity.h"

#include <mutex>
#include <vector>

#include "../core/global.h"

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

using namespace std;

static std::mutex affinityMutex;
static vector<int> serverCpus;
static vector<int> gameCpus;

static vector<int> parseList(const string& cpuList) {
  vector<int> cpus;
  for(const string& rawPart: Global::split(cpuList, ',')) {
    string part = Global::trim(rawPart);
    if(part.empty())
      continue;
    size_t dash = part.find('-');
    int lo, hi;
    if(dash == string::npos) {
      lo = hi = Global::stringToInt(part);
    } else {
      lo = Global::stringToInt(Global::trim(part.substr(0, dash)));
      hi = Global::stringToInt(Global::trim(part.substr(dash + 1)));
    }
    if(lo < 0 || hi < lo || hi > 1023)
      throw StringError("ThreadAffinity: bad CPU range '" + part + "' in '" + cpuList + "'");
    for(int c = lo; c <= hi; c++)
      cpus.push_back(c);
  }
  return cpus;
}

bool ThreadAffinity::parse(const string& cpuList, string& error) {
  try {
    parseList(cpuList);
    return true;
  } catch(const StringError& e) {
    error = e.what();
    return false;
  }
}

void ThreadAffinity::setServerCpus(const string& cpuList) {
  vector<int> cpus = parseList(cpuList);
  lock_guard<std::mutex> lock(affinityMutex);
  serverCpus = cpus;
}

void ThreadAffinity::setGameCpus(const string& cpuList) {
  vector<int> cpus = parseList(cpuList);
  lock_guard<std::mutex> lock(affinityMutex);
  gameCpus = cpus;
}

static void applyAffinity(const vector<int>& cpusRef, const char* name) {
#ifdef __linux__
  vector<int> cpus;
  {
    lock_guard<std::mutex> lock(affinityMutex);
    cpus = cpusRef;
  }
  pthread_setname_np(pthread_self(), name);
  if(cpus.empty())
    return;
  cpu_set_t set;
  CPU_ZERO(&set);
  for(int c: cpus)
    CPU_SET(c, &set);
  int ret = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if(ret != 0)
    throw StringError(string("ThreadAffinity: pthread_setaffinity_np failed for ") + name + ": error " + Global::intToString(ret));
#else
  (void)cpusRef;
  (void)name;
#endif
}

void ThreadAffinity::applyServer() { applyAffinity(serverCpus, "nnserver"); }
void ThreadAffinity::applyGame() { applyAffinity(gameCpus, "gameloop"); }
