#ifndef PosixClockGettime_h
#define PosixClockGettime_h

// C++ standard headers
#include <chrono>
#include <type_traits>

// POSIX standard headers
#include <time.h>

namespace detail {
  template <clockid_t CLOCK>
  struct IsSteady : public std::false_type {};

  // CLOCK_REALTIME is not "steady" because it "s affected by discontinuous jumps in the system time [...] and by the incremental adjustments performed by adjtime(3) and NTP"

  // CLOCK_REALTIME_ALARM is not "steady" because it is "like CLOCK_REALTIME, but not settable"

  // CLOCK_TAI probably is "steady" beacuse it "does not experience discontinuities and backwards jumps caused by NTP inserting leap seconds as CLOCK_REALTIME does"
#ifdef CLOCK_TAI
  template <>
  struct IsSteady<CLOCK_TAI> : public std::true_type {};
#endif

  // CLOCK_MONOTONIC is not "steady" because it "is affected by the incremental adjustments performed by adjtime(3) and NTP"

  // CLOCK_MONOTONIC_COARSE is not "steady" because it is "a faster but less precise version of CLOCK_MONOTONIC"

  // CLOCK_MONOTONIC_RAW is "steady" beacuse it "is not subject to NTP adjustments or the incremental adjustments performed by adjtime(3)"
#ifdef CLOCK_MONOTONIC_RAW
  template <>
  struct IsSteady<CLOCK_MONOTONIC_RAW> : public std::true_type {};
#endif

  // CLOCK_BOOTTIME is not "steady" because it "is identical to CLOCK_MONOTONIC, except that it also includes any time that the system is suspended"

  // CLOCK_BOOTTIME_ALARM is not "steady" because it is "like CLOCK_BOOTTIME"

  // CLOCK_PROCESS_CPUTIME_ID is not "steady" because it "measures CPU time consumed by this process"

  // CLOCK_THREAD_CPUTIME_ID is not "steady" because it "measures CPU time consumed by this thread"
}  // namespace detail

// A template class that wraps the POSIX clock_gettime function in the std::chrono interface
template <clockid_t CLOCK>
struct PosixClockGettime {
  using duration = std::chrono::nanoseconds;
  using rep = duration::rep;
  using period = duration::period;
  using time_point = std::chrono::time_point<PosixClockGettime, duration>;

  static constexpr bool is_steady = detail::IsSteady<CLOCK>::value;

  static time_point now() noexcept {
    timespec t;
    clock_gettime(CLOCK, &t);
    return time_point(std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec));
  }
};

#endif  // PosixClockGettime_h
