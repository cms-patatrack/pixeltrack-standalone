from std.collections import Span

from MojoSerial.MojoBridge.DTypes import SizeType, Typeable


@fieldwise_init
struct SiPixelDigisSoA(Copyable, Defaultable, Movable, Sized, Typeable):
    var _pdigi: List[UInt32]
    var _rawIdArr: List[UInt32]
    var _adc: List[UInt16]
    var _clus: List[Int32]

    # default constructor
    @always_inline
    def __init__(out self):
        self._pdigi = List[UInt32]()
        self._rawIdArr = List[UInt32]()
        self._adc = List[UInt16]()
        self._clus = List[Int32]()

    # C++ takes four `T const*` here; the columns are array bases, so they are
    # Spans (doc §5) and the copy is a plain indexed loop.
    def __init__(
        out self,
        var nDigis: SizeType,
        pdigi: Span[UInt32, _],
        rawIdArr: Span[UInt32, _],
        adc: Span[UInt16, _],
        clus: Span[Int32, _],
    ):
        var n = Int(nDigis)
        self._pdigi = List[UInt32](length=n, fill=0)
        self._rawIdArr = List[UInt32](length=n, fill=0)
        self._adc = List[UInt16](length=n, fill=0)
        self._clus = List[Int32](length=n, fill=0)
        for i in range(n):
            self._pdigi[i] = pdigi[i]
            self._rawIdArr[i] = rawIdArr[i]
            self._adc[i] = adc[i]
            self._clus[i] = clus[i]
        debug_assert(self._pdigi.__len__() == n)

    @always_inline
    def __len__(self) -> Int:
        return self._pdigi.__len__()

    @always_inline
    def pdigi(self, var i: SizeType) -> UInt32:
        return self._pdigi[i]

    @always_inline
    def rawIdArr(self, var i: SizeType) -> UInt32:
        return self._rawIdArr[i]

    @always_inline
    def adc(self, var i: SizeType) -> UInt16:
        return self._adc[i]

    @always_inline
    def clus(self, var i: SizeType) -> Int32:
        return self._clus[i]

    @always_inline
    def pdigiList(self) -> ref [self._pdigi] List[UInt32]:
        return self._pdigi

    @always_inline
    def rawIdArrList(self) -> ref [self._rawIdArr] List[UInt32]:
        return self._rawIdArr

    @always_inline
    def adcList(self) -> ref [self._adc] List[UInt16]:
        return self._adc

    @always_inline
    def clusList(self) -> ref [self._clus] List[Int32]:
        return self._clus

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelDigisSoA"
