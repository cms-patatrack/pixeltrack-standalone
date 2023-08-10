#ifndef bin_Timestamp_h
#define bin_Timestamp_h

#include <chrono>

#include "PosixClockGettime.h"

struct Timestamp {
  PosixClockGettime<CLOCK_PROCESS_CPUTIME_ID>::time_point cpu;
  std::chrono::steady_clock::time_point time;

  void mark() {
    cpu = PosixClockGettime<CLOCK_PROCESS_CPUTIME_ID>::now();
    time = std::chrono::steady_clock::now();
  }

  static Timestamp now() {
    Timestamp t;
    t.mark();
    return t;
  }
};

#endif  // bin_Timestamp_h
