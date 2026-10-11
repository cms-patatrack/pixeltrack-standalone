from std.sys import size_of
from std.memory import memset

from MojoSerial.CUDACore.AtomicPairCounter import AtomicPairCounter
from MojoSerial.CUDACore.CUDAStdAlgorithm import CUDAStdAlgorithm
from MojoSerial.CUDACore.PrefixScan import blockPrefixScan
from MojoSerial.MojoBridge.DTypes import Typeable, signed_to_unsigned


def countFromVector[
    T: DType, //
](
    mut h: HistoContainer[T, ...],
    nh: UInt32,
    v: Span[Scalar[T], _],
    offsets: Span[UInt32, _],
):
    for i in range(Int(offsets[Int(nh)])):
        # C++: off = upper_bound(offsets, offsets + nh + 1, i); ih = off - offsets - 1
        var off = CUDAStdAlgorithm.upper_bound(
            offsets[0 : Int(nh) + 1], UInt32(i)
        )

        debug_assert(offsets[off] > 0)
        var ih = Int32(off) - 1

        debug_assert(ih >= 0)
        debug_assert(ih < Int32(nh))
        h.count(v[i], ih.cast[DType.uint32]())


def fillFromVector[
    T: DType, //
](
    mut h: HistoContainer[T, ...],
    nh: UInt32,
    v: Span[Scalar[T], _],
    offsets: Span[UInt32, _],
):
    for i in range(Int(offsets[Int(nh)])):
        # C++: off = upper_bound(offsets, offsets + nh + 1, i); ih = off - offsets - 1
        var off = CUDAStdAlgorithm.upper_bound(
            offsets[0 : Int(nh) + 1], UInt32(i)
        )

        debug_assert(offsets[off] > 0)
        var ih = Int32(off) - 1

        debug_assert(ih >= 0)
        debug_assert(ih < Int32(nh))
        h.fill(v[i], Scalar[h.IndexType](i), ih.cast[DType.uint32]())


@always_inline
def launchZero(mut h: HistoContainer):
    var poff = h.off.unsafe_ptr()
    var size = Int(h.totbins())
    debug_assert(size >= Int(h.totbins()))
    memset(poff, 0, size)  # memset sets by bytes in C++, but by elements here
    h.psws = 0  # included in C++ memset


@always_inline
def launchFinalize(mut h: HistoContainer[...]):
    h.finalize()


@always_inline
def fillManyFromVector[
    T: DType
](
    mut h: HistoContainer[T, ...],
    nh: UInt32,
    v: Span[Scalar[T], _],
    offsets: Span[UInt32, _],
    totSize: UInt32,
):
    launchZero(h)
    countFromVector(h, nh, v, offsets)
    h.finalize()
    fillFromVector(h, nh, v, offsets)


def finalizeBulk(
    apc: AtomicPairCounter, mut assoc: HistoContainer[...]
):
    assoc.bulkFinalizeFill(apc)


def forEachInBins[
    V: DType,
    NBINS: UInt32,
    SIZE: UInt32,
    S: UInt32,
    I: DType,
    NHISTS: UInt32,
    func: def (Scalar[I]) capturing -> None,
](ref hist: HistoContainer[V, NBINS, SIZE, S, I, NHISTS], value: Scalar[V], n: Int32):
    """Iterate over N bins left and right of the one containing "v"."""
    var bs = hist.bin(value).cast[DType.int32]()
    var be = min(hist.nbins().cast[DType.int32]() - 1, bs + n)
    bs = max(Int32(0), bs - n)
    debug_assert(be >= bs)

    var pj = hist.begin(UInt32(bs))
    while pj < hist.end(UInt32(be)):
        func(pj[])
        pj += 1


def forEachInWindow[
    V: DType,
    NBINS: UInt32,
    SIZE: UInt32,
    S: UInt32,
    I: DType,
    NHISTS: UInt32,
    func: def (Scalar[I]) capturing -> None,
](
    ref hist: HistoContainer[V, NBINS, SIZE, S, I, NHISTS],
    wmin: Scalar[V],
    wmax: Scalar[V],
):
    """Iterate over bins containing all values in window wmin, wmax."""
    var bs = hist.bin(wmin)
    var be = hist.bin(wmax)
    debug_assert(be >= bs)

    var pj = hist.begin(bs.cast[DType.uint32]())
    while pj < hist.end(be.cast[DType.uint32]()):
        func(pj[])
        pj += 1


struct HistoContainer[
    T: DType,  # the type of the discretized input values
    NBINS: UInt32,  # number of bins
    SIZE: UInt32,  # max number of elements
    S: UInt32 = UInt32(
        size_of[T]() * 8
    ),  # number of significant bits in T
    I: DType = DType.uint32,  # type stored in the container (usually an index in a vector of the input values)
    NHISTS: UInt32 = 1,  # number of histos stored
](Defaultable, Movable, Sized, Typeable):
    comptime Counter = UInt32
    comptime CountersOnly = HistoContainer[Self.T, Self.NBINS, 0, Self.S, Self.I, Self.NHISTS]
    comptime IndexType = Self.I

    comptime D = Scalar[Self.T]
    comptime UT = signed_to_unsigned[Self.T]()
    comptime UD = Scalar[Self.UT]

    # Heap-backed, not InlineArray: as an inline struct this is 101 kB for the
    # rechit Hist and 448 kB for TrackSoA's HitContainer, and an
    # OwnedPointer of such a struct costs superlinearly in the number of
    # derefs at compile time (doc §12). Capacity and indexing are unchanged.
    var off: List[Self.Counter]
    var psws: Int32
    var bins: List[Scalar[Self.IndexType]]

    @staticmethod
    def ilog2(var v: UInt32) -> UInt32:
        comptime b: InlineArray[UInt32, 5] = [
            0x2,
            0xC,
            0xF0,
            0xFF00,
            0xFFFF0000,
        ]
        comptime s: InlineArray[UInt32, 5] = [1, 2, 4, 8, 16]

        var r: UInt32 = 0
        comptime for i in range(4, -1, -1):
            comptime bi = b[i]
            comptime si = s[i]
            if v & bi:
                v >>= si
                r |= si
        return r

    @staticmethod
    @always_inline
    def sizeT() -> UInt32:
        return Self.S

    @staticmethod
    @always_inline
    def nbins() -> UInt32:
        return Self.NBINS

    @staticmethod
    @always_inline
    def nhists() -> UInt32:
        return Self.NHISTS

    @staticmethod
    @always_inline
    def totbins() -> UInt32:
        return Self.NHISTS * Self.NBINS + 1

    @staticmethod
    @always_inline
    def nbits() -> UInt32:
        return Self.ilog2(Self.NBINS - 1) + 1

    @staticmethod
    @always_inline
    def capacity() -> UInt32:
        return Self.SIZE

    @staticmethod
    @always_inline
    def histOff(nh: UInt32) -> UInt32:
        return Self.NBINS * nh

    @staticmethod
    @always_inline
    def bin(t: Self.D) -> Self.UD:
        comptime shift: UInt32 = Self.sizeT() - Self.nbits()
        comptime mask: UInt32 = (1 << Self.nbits()) - 1
        return ((t.cast[DType.uint32]() >> shift) & mask).cast[Self.UT]()

    @always_inline
    def __init__(out self):
        self.off = List[Self.Counter](length=Int(Self.totbins()), fill=0)
        self.psws = 0
        self.bins = List[Scalar[Self.IndexType]](
            length=Int(Self.capacity()), fill=0
        )

    @always_inline
    def __len__(self) -> Int:
        return Int(self.size())

    @always_inline
    def zero(mut self):
        memset(self.off.unsafe_ptr(), 0, Int(Self.totbins()))

    @always_inline
    def add(mut self, ref co: Self.CountersOnly):
        comptime for i in range(Self.totbins()):
            self.off[i] += co.off[i]

    @always_inline
    def countDirect(mut self, b: Self.D):
        debug_assert(b.cast[DType.uint32]() < Self.nbins())
        self.off[b] += 1

    @always_inline
    def fillDirect(mut self, b: Self.D, j: Scalar[Self.IndexType]):
        debug_assert(b.cast[DType.uint32]() < Self.nbins())
        var w = self.off[b]
        self.off[b] -= 1
        debug_assert(w > 0)
        self.bins[w - 1] = j

    @always_inline
    def bulkFill(
        mut self,
        mut apc: AtomicPairCounter,
        v: Span[Scalar[Self.IndexType], _],
        n: UInt32,
    ) -> Int32:
        var c = apc.add(n)
        if c[1] >= Self.nbins():
            return -Int32(c[1])

        self.off[c[1]] = c[0]
        for j in range(n):
            self.bins[c[0] + j] = v[j]

        return Int32(c[1])

    @always_inline
    def bulkFinalize(mut self, ref apc: AtomicPairCounter):
        self.off[apc.get()[1]] = apc.get()[0]

    @always_inline
    def bulkFinalizeFill(mut self, ref apc: AtomicPairCounter):
        var m = apc.get()[1]
        var n = apc.get()[0]

        if m >= Self.nbins():  # overflow
            self.off[Self.nbins()] = UInt32(self.off[Self.nbins() - 1])
            return

        for i in range(m, Self.totbins()):
            self.off[i] = n

    @always_inline
    def count(mut self, t: Self.D):
        var b = Self.bin(t).cast[DType.uint32]()
        debug_assert(b < Self.nbins())
        self.off[b] += 1

    @always_inline
    def fill(mut self, t: Self.D, j: Scalar[Self.IndexType]):
        var b = Self.bin(t).cast[DType.uint32]()
        debug_assert(b < Self.nbins())
        var w = self.off[b]
        self.off[b] -= 1
        debug_assert(w > 0)
        self.bins[w - 1] = j

    @always_inline
    def count(mut self, t: Self.D, nh: UInt32):
        var b = Self.bin(t).cast[DType.uint32]()
        debug_assert(b < Self.nbins())
        b += Self.histOff(nh)
        debug_assert(b < Self.totbins())
        self.off[b] += 1

    @always_inline
    def fill(mut self, t: Self.D, j: Scalar[Self.IndexType], nh: UInt32):
        var b = Self.bin(t).cast[DType.uint32]()
        debug_assert(b < Self.nbins())
        b += Self.histOff(nh)
        debug_assert(b < Self.totbins())
        var w = self.off[b]
        self.off[b] -= 1
        debug_assert(w > 0)
        self.bins[w - 1] = j

    @always_inline
    def finalize(mut self):
        debug_assert(self.off[Self.totbins() - 1] == 0)
        blockPrefixScan(Span(self.off), Self.totbins())
        debug_assert(
            self.off[Self.totbins() - 1] == self.off[Self.totbins() - 2]
        )

    @always_inline
    def size(self) -> UInt32:
        return UInt32(self.off[Self.totbins() - 1])

    @always_inline
    def size(self, b: UInt32) -> UInt32:
        return UInt32(self.off[b + 1] - self.off[b])

    def begin(self) -> Pointer[Scalar[Self.IndexType], origin_of(self.bins)]:
        return self.bins.unsafe_ptr()

    def end(self) -> Pointer[Scalar[Self.IndexType], origin_of(self.bins)]:
        return self.begin() + self.size()

    def begin(
        self, b: UInt32
    ) -> Pointer[Scalar[Self.IndexType], origin_of(self.bins)]:
        return self.bins.unsafe_ptr() + self.off[b]

    def end(
        self, b: UInt32
    ) -> Pointer[Scalar[Self.IndexType], origin_of(self.bins)]:
        return self.bins.unsafe_ptr() + self.off[b + 1]

    @always_inline
    @staticmethod
    def dtype() -> String:
        return (
            "HistoContainer["
            + String(Self.T)
            + ", "
            + String(Self.NBINS)
            + ", "
            + String(Self.SIZE)
            + ", "
            + String(Self.S)
            + ", "
            + String(Self.I)
            + ", "
            + String(Self.NHISTS)
            + "]"
        )


comptime OneToManyAssoc[
    I: DType,  # type stored in the container (usually an index in a vector of the input values)
    MAXONES: UInt32,  # max number of "ones"
    MAXMANYS: UInt32,  # max number of "manys"
] = HistoContainer[
    DType.uint32,
    MAXONES,
    MAXMANYS,
    UInt32(size_of[DType.uint32]() * 8),
    I,
    1,
]
