from std.collections import Span
from std.sys import size_of

from MojoSerial.CUDACore.CUDACompat import CUDAStreamType, cudaStreamDefault
from MojoSerial.CUDACore.HistoContainer import HistoContainer
from MojoSerial.CUDADataFormats.GPUClusteringConstants import (
    GPUClusteringConstants,
)
from MojoSerial.Geometry.Phase1PixelTopology import (
    Phase1PixelTopology,
    AverageGeometry,
)
from MojoSerial.MojoBridge.DTypes import Float, Typeable

comptime Hist = HistoContainer[
    DType.int16,
    128,
    GPUClusteringConstants.MaxNumClusters,
    UInt32(8 * size_of[UInt16]()),
    DType.int16,
    10,
]


struct TrackingRecHit2DHeterogeneous(Defaultable, Movable, Typeable):
    """Hit SoA. C++ packs every column into m_store16/m_store32 and hands out a
    TrackingRecHit2DSOAView of pointers into them; this port stores each column
    as its own typed buffer and folds the view's accessors onto the owner.
    """

    comptime HIndexType = UInt16  # if maxHits is <= 2^16

    @staticmethod
    @always_inline
    def maxHits() -> UInt32:
        return GPUClusteringConstants.MaxNumClusters

    # Columns are plain Lists: a List is already a heap-backed handle, so an
    # OwnedPointer around one is a second allocation and a second indirection
    # that the backend reloads every iteration (doc §12).
    #
    # local coord
    var m_xl_d: List[Float]
    var m_yl_d: List[Float]
    var m_xerr_d: List[Float]
    var m_yerr_d: List[Float]

    # global coord
    var m_xg_d: List[Float]
    var m_yg_d: List[Float]
    var m_zg_d: List[Float]
    var m_rg_d: List[Float]
    var m_iphi_d: List[Int16]

    # cluster properties
    var m_charge_d: List[Int32]
    var m_xsize_d: List[Int16]
    var m_ysize_d: List[Int16]
    var m_detInd_d: List[UInt16]

    # supporting objects, owned
    var m_AverageGeometryStore: AverageGeometry
    var m_HistStore: Hist
    var m_hitsLayerStart_d: List[UInt32]

    # forwarded from clusters, copied in
    var m_hitsModuleStart_d: List[UInt32]

    var m_nHits: UInt32

    @always_inline
    def __init__(out self):
        self.m_xl_d = List[Float]()
        self.m_yl_d = List[Float]()
        self.m_xerr_d = List[Float]()
        self.m_yerr_d = List[Float]()

        self.m_xg_d = List[Float]()
        self.m_yg_d = List[Float]()
        self.m_zg_d = List[Float]()
        self.m_rg_d = List[Float]()
        self.m_iphi_d = List[Int16]()

        self.m_charge_d = List[Int32]()
        self.m_xsize_d = List[Int16]()
        self.m_ysize_d = List[Int16]()
        self.m_detInd_d = List[UInt16]()

        self.m_AverageGeometryStore = AverageGeometry()
        self.m_HistStore = Hist()
        self.m_hitsLayerStart_d = List[UInt32]()

        self.m_hitsModuleStart_d = List[UInt32]()

        self.m_nHits = 0

    def __init__(
        out self,
        nHits: UInt32,
        hitsModuleStart: Span[UInt32, _],
        stream: CUDAStreamType = cudaStreamDefault,
    ):
        self.m_nHits = nHits

        self.m_hitsModuleStart_d = List[UInt32]()
        for i in range(len(hitsModuleStart)):
            self.m_hitsModuleStart_d.append(hitsModuleStart[i])

        self.m_AverageGeometryStore = AverageGeometry()
        self.m_HistStore = Hist()

        # C++ reserves numberOfLayers + 1 uint32 at the tail of m_store32
        self.m_hitsLayerStart_d = List[UInt32](
            length=Int(Phase1PixelTopology.numberOfLayers) + 1, fill=0
        )

        var n = Int(nHits)
        self.m_xl_d = List[Float](length=n, fill=0)
        self.m_yl_d = List[Float](length=n, fill=0)
        self.m_xerr_d = List[Float](length=n, fill=0)
        self.m_yerr_d = List[Float](length=n, fill=0)

        self.m_xg_d = List[Float](length=n, fill=0)
        self.m_yg_d = List[Float](length=n, fill=0)
        self.m_zg_d = List[Float](length=n, fill=0)
        self.m_rg_d = List[Float](length=n, fill=0)
        self.m_iphi_d = List[Int16](length=n, fill=0)

        self.m_charge_d = List[Int32](length=n, fill=0)
        self.m_xsize_d = List[Int16](length=n, fill=0)
        self.m_ysize_d = List[Int16](length=n, fill=0)
        self.m_detInd_d = List[UInt16](length=n, fill=0)

    @always_inline
    def nHits(self) -> UInt32:
        return self.m_nHits

    # ---- per-hit accessors (were TrackingRecHit2DSOAView's) ----

    @always_inline
    def xLocal(ref self, i: Int) -> ref [self.m_xl_d[i]] Float:
        return self.m_xl_d[i]

    @always_inline
    def yLocal(ref self, i: Int) -> ref [self.m_yl_d[i]] Float:
        return self.m_yl_d[i]

    @always_inline
    def xerrLocal(ref self, i: Int) -> ref [self.m_xerr_d[i]] Float:
        return self.m_xerr_d[i]

    @always_inline
    def yerrLocal(ref self, i: Int) -> ref [self.m_yerr_d[i]] Float:
        return self.m_yerr_d[i]

    @always_inline
    def xGlobal(ref self, i: Int) -> ref [self.m_xg_d[i]] Float:
        return self.m_xg_d[i]

    @always_inline
    def yGlobal(ref self, i: Int) -> ref [self.m_yg_d[i]] Float:
        return self.m_yg_d[i]

    @always_inline
    def zGlobal(ref self, i: Int) -> ref [self.m_zg_d[i]] Float:
        return self.m_zg_d[i]

    @always_inline
    def rGlobal(ref self, i: Int) -> ref [self.m_rg_d[i]] Float:
        return self.m_rg_d[i]

    @always_inline
    def iphi(ref self, i: Int) -> ref [self.m_iphi_d[i]] Int16:
        return self.m_iphi_d[i]

    @always_inline
    def charge(ref self, i: Int) -> ref [self.m_charge_d[i]] Int32:
        return self.m_charge_d[i]

    @always_inline
    def clusterSizeX(ref self, i: Int) -> ref [self.m_xsize_d[i]] Int16:
        return self.m_xsize_d[i]

    @always_inline
    def clusterSizeY(ref self, i: Int) -> ref [self.m_ysize_d[i]] Int16:
        return self.m_ysize_d[i]

    @always_inline
    def detectorIndex(ref self, i: Int) -> ref [self.m_detInd_d[i]] UInt16:
        return self.m_detInd_d[i]

    @always_inline
    def hitsModuleStart(self, i: Int) -> UInt32:
        return self.m_hitsModuleStart_d[i]

    # ---- supporting objects ----

    @always_inline
    def phiBinner(ref self) -> ref [self.m_HistStore] Hist:
        return self.m_HistStore

    @always_inline
    def averageGeometry(
        ref self,
    ) -> ref [self.m_AverageGeometryStore] AverageGeometry:
        return self.m_AverageGeometryStore

    # ---- whole-column views ----

    @always_inline
    def hitsModuleStart(
        self,
    ) -> Span[UInt32, origin_of(self.m_hitsModuleStart_d)].Immutable:
        return Span(self.m_hitsModuleStart_d)

    @always_inline
    def hitsLayerStart(
        ref self,
    ) -> Span[UInt32, origin_of(self.m_hitsLayerStart_d)]:
        return Span(self.m_hitsLayerStart_d)

    @always_inline
    def iphi(ref self) -> Span[Int16, origin_of(self.m_iphi_d)]:
        return Span(self.m_iphi_d)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "TrackingRecHit2DHeterogeneous"


comptime TrackingRecHit2DCPU = TrackingRecHit2DHeterogeneous