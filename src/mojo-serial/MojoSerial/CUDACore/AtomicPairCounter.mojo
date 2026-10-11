from std.memory import bitcast

from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct Counters(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    var c: SIMD[DType.uint32, 2]
    # alias n: UInt32 = c[0] # in a "One to Many" association is the number of "One"
    # alias m: UInt32 = c[1] # in a "One to Many" association is the total number of associations
    # alias ac: UInt64 = bitcast[DType.uint64, 1](c)

    @always_inline
    def __init__(out self):
        self.c = SIMD[DType.uint32, 2]()

    @always_inline
    def get_ac(self) -> UInt64:
        return bitcast[DType.uint64, 1](self.c)

    @always_inline
    def set_ac(mut self, var ac: UInt64):
        self.c = bitcast[DType.uint32, 2](ac)

    @always_inline
    def __getitem__(self) -> UInt64:
        return self.get_ac()

    @always_inline
    def __getitem__(self, i: Int) -> UInt32:
        return self.c[i]

    @always_inline
    def __setitem__(mut self, i: Int, val: UInt32):
        self.c[i] = val

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "Counters"


@fieldwise_init
struct AtomicPairCounter(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    var counter: Counters
    comptime CounterType = UInt64
    comptime incr: Self.CounterType = 1 << 32

    @always_inline
    def __init__(out self):
        self.counter = Counters()

    @always_inline
    def __init__(out self, i: Self.CounterType):
        self = Self()
        self.counter.set_ac(i)

    @always_inline
    def get(self) -> Counters:
        return self.counter

    @always_inline
    def add(mut self, i: UInt32) -> Counters:
        var ret = self.counter
        var c = Self.incr + i.cast[DType.uint64]()
        self.counter.set_ac(self.counter.get_ac() + c)
        return ret

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "AtomicPairCounter"
