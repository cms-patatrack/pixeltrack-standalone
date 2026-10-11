from std.ffi import external_call, c_long, c_int

comptime TimeType = c_int
comptime Long = c_long
comptime ClockIdType = c_int

comptime CLOCK_REALTIME: ClockIdType = 0
comptime CLOCK_MONOTONIC: ClockIdType = 1
comptime CLOCK_PROCESS_CPUTIME_ID: ClockIdType = 2
comptime CLOCK_THREAD_CPUTIME_ID: ClockIdType = 3


@always_inline
def is_steady[CLOCK: ClockIdType]() -> Bool:
    comptime if CLOCK == CLOCK_REALTIME:
        return False
    elif CLOCK == CLOCK_MONOTONIC:
        return False
    elif CLOCK == CLOCK_PROCESS_CPUTIME_ID:
        return False
    elif CLOCK == CLOCK_THREAD_CPUTIME_ID:
        return False
    return False


@fieldwise_init
struct TimeSpec(Copyable, Defaultable, Movable, TrivialRegisterPassable):
    var tv_sec: TimeType
    var tv_nsec: c_long

    @always_inline
    def __init__(out self):
        self.tv_sec = 0
        self.tv_nsec = 0
struct PosixClockGettime[CLOCK: ClockIdType]:
    comptime rep = Int
    comptime period = 10**9

    comptime is_steady = is_steady[Self.CLOCK]()

    @staticmethod
    def now() -> Self.rep:
        """Returns clock_gettime in nsec."""
        var t = TimeSpec()
        debug_assert(
            external_call[
                "clock_gettime",
                c_int,
                ClockIdType,
                UnsafePointer[TimeSpec, origin_of(t)],
            ](Self.CLOCK, UnsafePointer(to=t))
            == 0
        )
        return Int(c_long(t.tv_sec) * Self.period + t.tv_nsec)
