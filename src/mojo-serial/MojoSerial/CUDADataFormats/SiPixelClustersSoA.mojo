from std.collections import Span
from MojoSerial.MojoBridge.DTypes import SizeType, Typeable


struct SiPixelClustersSoA(Defaultable, Movable, Typeable):
    # Plain Lists: a List is already a heap-backed handle, so an OwnedPointer
    # around one is a second allocation and a second indirection (doc §12).
    var moduleStart_d: List[UInt32]  # index of the first pixel of each module
    var clusInModule_d: List[UInt32]  # number of clusters found in each module
    var moduleId_d: List[UInt32]  # module id of each module
    var clusModuleStart_d: List[UInt32]  # index of the first cluster of each module
    var nClusters_h: UInt32

    def __init__(out self):
        self.moduleStart_d = List[UInt32]()
        self.clusInModule_d = List[UInt32]()
        self.moduleId_d = List[UInt32]()
        self.clusModuleStart_d = List[UInt32]()
        self.nClusters_h = 0

    def __init__(out self, maxClusters: SizeType):
        debug_assert(maxClusters >= 0)
        self.moduleStart_d = List[UInt32](length=Int(maxClusters) + 1, fill=0)
        self.clusInModule_d = List[UInt32](length=Int(maxClusters), fill=0)
        self.moduleId_d = List[UInt32](length=Int(maxClusters), fill=0)
        self.clusModuleStart_d = List[UInt32](
            length=Int(maxClusters) + 1, fill=0
        )
        self.nClusters_h = 0

    @always_inline
    def nClusters(self) -> UInt32:
        return self.nClusters_h

    @always_inline
    def setNClusters(mut self, nClusters: UInt32):
        self.nClusters_h = nClusters

    # C++ DeviceConstView element accessors, folded onto the owner.
    @always_inline
    def moduleStart(self, i: Int) -> UInt32:
        return self.moduleStart_d[i]

    @always_inline
    def clusInModule(self, i: Int) -> UInt32:
        return self.clusInModule_d[i]

    @always_inline
    def moduleId(self, i: Int) -> UInt32:
        return self.moduleId_d[i]

    @always_inline
    def clusModuleStart(self, i: Int) -> UInt32:
        return self.clusModuleStart_d[i]

    # C++ non-const accessors; exclusive borrow, one live at a time.
    @always_inline
    def moduleStart(ref self) -> Span[UInt32, origin_of(self.moduleStart_d)]:
        return Span(self.moduleStart_d)

    @always_inline
    def clusInModule(
        ref self,
    ) -> Span[UInt32, origin_of(self.clusInModule_d)]:
        return Span(self.clusInModule_d)

    @always_inline
    def moduleId(ref self) -> Span[UInt32, origin_of(self.moduleId_d)]:
        return Span(self.moduleId_d)

    @always_inline
    def clusModuleStart(
        ref self,
    ) -> Span[UInt32, origin_of(self.clusModuleStart_d)]:
        return Span(self.clusModuleStart_d)

    # C++ c_* const accessors; shared borrow, several may be live.
    @always_inline
    def c_moduleStart(
        self,
    ) -> Span[UInt32, origin_of(self.moduleStart_d)].Immutable:
        return Span(self.moduleStart_d)

    @always_inline
    def c_clusInModule(
        self,
    ) -> Span[UInt32, origin_of(self.clusInModule_d)].Immutable:
        return Span(self.clusInModule_d)

    @always_inline
    def c_moduleId(self) -> Span[UInt32, origin_of(self.moduleId_d)].Immutable:
        return Span(self.moduleId_d)

    @always_inline
    def c_clusModuleStart(
        self,
    ) -> Span[UInt32, origin_of(self.clusModuleStart_d)].Immutable:
        return Span(self.clusModuleStart_d)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelClustersSoA"
