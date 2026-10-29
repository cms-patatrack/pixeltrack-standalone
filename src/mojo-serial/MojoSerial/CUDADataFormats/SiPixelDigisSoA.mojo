from std.collections import Span

from MojoSerial.MojoBridge.DTypes import SizeType, Typeable


struct SiPixelDigisSoA(Defaultable, Movable, Typeable):
    # Plain Lists: a List is already a heap-backed handle, so an OwnedPointer
    # around one is a second allocation and a second indirection (doc §12).
    var xx_d: List[UInt16]  # local coordinates of each pixel
    var yy_d: List[UInt16]
    var adc_d: List[UInt16]  # ADC of each pixel
    var moduleInd_d: List[UInt16]  # module id of each pixel
    var clus_d: List[Int32]  # cluster id of each pixel

    # These are for CPU output; should we (eventually) place them to a
    # separate product?
    var pdigi_d: List[UInt32]
    var rawIdArr_d: List[UInt32]

    var nModules_h: UInt32
    var nDigis_h: UInt32

    @always_inline
    def __init__(out self):
        self.xx_d = List[UInt16]()
        self.yy_d = List[UInt16]()
        self.adc_d = List[UInt16]()
        self.moduleInd_d = List[UInt16]()
        self.clus_d = List[Int32]()

        self.pdigi_d = List[UInt32]()
        self.rawIdArr_d = List[UInt32]()

        self.nModules_h = 0
        self.nDigis_h = 0

    @always_inline
    def __init__(out self, maxFedWords: SizeType):
        self.xx_d = List[UInt16](length=Int(maxFedWords), fill=0)
        self.yy_d = List[UInt16](length=Int(maxFedWords), fill=0)
        self.adc_d = List[UInt16](length=Int(maxFedWords), fill=0)
        self.moduleInd_d = List[UInt16](length=Int(maxFedWords), fill=0)
        self.clus_d = List[Int32](length=Int(maxFedWords), fill=0)

        self.pdigi_d = List[UInt32](length=Int(maxFedWords), fill=0)
        self.rawIdArr_d = List[UInt32](length=Int(maxFedWords), fill=0)

        self.nModules_h = 0
        self.nDigis_h = 0

    @always_inline
    def setNModulesDigis(mut self, nModules: UInt32, nDigis: UInt32):
        self.nModules_h = nModules
        self.nDigis_h = nDigis

    @always_inline
    def nModules(self) -> UInt32:
        return self.nModules_h

    @always_inline
    def nDigis(self) -> UInt32:
        return self.nDigis_h

    # C++ DeviceConstView element accessors, folded onto the owner.
    @always_inline
    def xx(self, i: Int) -> UInt16:
        return self.xx_d[i]

    @always_inline
    def yy(self, i: Int) -> UInt16:
        return self.yy_d[i]

    @always_inline
    def adc(self, i: Int) -> UInt16:
        return self.adc_d[i]

    @always_inline
    def moduleInd(self, i: Int) -> UInt16:
        return self.moduleInd_d[i]

    @always_inline
    def clus(self, i: Int) -> Int32:
        return self.clus_d[i]

    # C++ non-const accessors; exclusive borrow, one live at a time.
    @always_inline
    def xx(ref self) -> Span[UInt16, origin_of(self.xx_d)]:
        return Span(self.xx_d)

    @always_inline
    def yy(ref self) -> Span[UInt16, origin_of(self.yy_d)]:
        return Span(self.yy_d)

    @always_inline
    def adc(ref self) -> Span[UInt16, origin_of(self.adc_d)]:
        return Span(self.adc_d)

    @always_inline
    def moduleInd(ref self) -> Span[UInt16, origin_of(self.moduleInd_d)]:
        return Span(self.moduleInd_d)

    @always_inline
    def clus(ref self) -> Span[Int32, origin_of(self.clus_d)]:
        return Span(self.clus_d)

    @always_inline
    def pdigi(ref self) -> Span[UInt32, origin_of(self.pdigi_d)]:
        return Span(self.pdigi_d)

    @always_inline
    def rawIdArr(ref self) -> Span[UInt32, origin_of(self.rawIdArr_d)]:
        return Span(self.rawIdArr_d)

    # C++ c_* const accessors; shared borrow, several may be live.
    @always_inline
    def c_xx(self) -> Span[UInt16, origin_of(self.xx_d)].Immutable:
        return Span(self.xx_d)

    @always_inline
    def c_yy(self) -> Span[UInt16, origin_of(self.yy_d)].Immutable:
        return Span(self.yy_d)

    @always_inline
    def c_adc(self) -> Span[UInt16, origin_of(self.adc_d)].Immutable:
        return Span(self.adc_d)

    @always_inline
    def c_moduleInd(
        self,
    ) -> Span[UInt16, origin_of(self.moduleInd_d)].Immutable:
        return Span(self.moduleInd_d)

    @always_inline
    def c_clus(self) -> Span[Int32, origin_of(self.clus_d)].Immutable:
        return Span(self.clus_d)

    @always_inline
    def c_pdigi(self) -> Span[UInt32, origin_of(self.pdigi_d)].Immutable:
        return Span(self.pdigi_d)

    @always_inline
    def c_rawIdArr(
        self,
    ) -> Span[UInt32, origin_of(self.rawIdArr_d)].Immutable:
        return Span(self.rawIdArr_d)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelDigisSoA"
