from MojoSerial.CUDACore.CUDACompat import CUDACompat
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import (
    PixelTrack as pixelTrack,
    TrackQuality,
)
from MojoSerial.CUDADataFormats.ZVertexHeterogeneous import ZVertexHeterogeneous
from MojoSerial.CUDADataFormats.ZVertexSoA import ZVertexSoA
from MojoSerial.MojoBridge.DTypes import Float, Typeable
from MojoSerial.plugin_PixelVertexFinding.gpuClusterTracksByDensity import (
    clusterTracksByDensity,
)
from MojoSerial.plugin_PixelVertexFinding.gpuFitVertices import fitVertices
from MojoSerial.plugin_PixelVertexFinding.gpuSortByPt2 import sortByPt2
from MojoSerial.plugin_PixelVertexFinding.gpuSplitVertices import splitVertices

comptime ZVertices = ZVertexSoA
comptime TkSoA = pixelTrack.TrackSoA


# workspace used in the vertex reco algos
#
# Columns are Lists, not inline arrays, for the reason given on ZVertexSoA: as
# an inline struct this is ~600 kB, and every deref of an OwnedPointer[WorkSpace]
# materializes it.
@fieldwise_init
struct WorkSpace(Copyable, Defaultable, Movable, Typeable):
    comptime MAXTRACKS = ZVertexSoA.MAXTRACKS
    comptime MAXVTX = ZVertexSoA.MAXVTX

    var ntrks: UInt32  # number of "selected tracks"
    var itrk: List[UInt16]  # index of original track
    var zt: List[Float]  # input track z at bs
    var ezt2: List[Float]  # input error^2 on the above
    var ptt2: List[Float]  # input pt^2 on the above
    var izt: List[UInt8]  # interized z-position of input tracks
    var iv: List[Int32]  # vertex index for each associated track

    var nvIntermediate: UInt32  # the number of vertices after splitting pruning etc.

    @always_inline
    def __init__(out self):
        self.ntrks = 0
        self.itrk = List[UInt16](length=Int(Self.MAXTRACKS), fill=0)
        self.zt = List[Float](length=Int(Self.MAXTRACKS), fill=0.0)
        self.ezt2 = List[Float](length=Int(Self.MAXTRACKS), fill=0.0)
        self.ptt2 = List[Float](length=Int(Self.MAXTRACKS), fill=0.0)
        self.izt = List[UInt8](length=Int(Self.MAXTRACKS), fill=0)
        self.iv = List[Int32](length=Int(Self.MAXTRACKS), fill=0)
        self.nvIntermediate = 0

    @always_inline
    def init(mut self):
        self.ntrks = 0
        self.nvIntermediate = 0

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "WorkSpace"


@always_inline
def init(mut data: ZVertexSoA, mut ws: WorkSpace):
    data.init()
    ws.init()


@always_inline
def loadTracks(
    tracks: TkSoA,
    mut soa: ZVertexSoA,
    mut ws: WorkSpace,
    ptMin: Float,
):
    # C++ asserts ptracks/soa are non-null; a borrow cannot be null.
    ref fit = tracks.stateAtBS
    var quality = tracks.qualityData()

    for idx in range(TkSoA.stride()):
        var nHits = tracks.nHits(idx)
        if nHits == 0:
            break  # this is a guard: maybe we need to move to nTracks...

        # initialize soa...
        soa.idv[Int(idx)] = -1

        if nHits < 4:
            continue  # no triplets
        if quality[Int(idx)] != TrackQuality.loose:
            continue

        var pt = tracks.pt[Int(idx)]

        if pt < ptMin:
            continue

        var it = CUDACompat.atomicAdd(ws.ntrks, UInt32(1))
        ws.itrk[Int(it)] = UInt16(idx)
        ws.zt[Int(it)] = tracks.zip(idx)
        ws.ezt2[Int(it)] = rebind[Scalar[DType.float32]](
            fit.covariance[idx][14, 0]
        )
        ws.ptt2[Int(it)] = pt * pt


struct Producer(Typeable):
    comptime ZVertices = ZVertexSoA
    comptime WorkSpace = WorkSpace
    comptime TkSoA = pixelTrack.TrackSoA

    var oneKernel_: Bool
    var useDensity_: Bool
    var useDBSCAN_: Bool
    var useIterative_: Bool

    var minT: Int32  # min number of neighbours to be "core"
    var eps: Float  # max absolute distance to cluster
    var errmax: Float  # max error to be "seed"
    var chi2max: Float  # max normalized distance to cluster

    @always_inline
    def __init__(
        out self,
        oneKernel: Bool,
        useDensity: Bool,
        useDBSCAN: Bool,
        useIterative: Bool,
        iminT: Int32,  # min number of neighbours to be "core"
        ieps: Float,  # max absolute distance to cluster
        ierrmax: Float,  # max error to be "seed"
        ichi2max: Float,  # max normalized distance to cluster
    ):
        self.oneKernel_ = oneKernel and not (useDBSCAN or useIterative)
        self.useDensity_ = useDensity
        self.useDBSCAN_ = useDBSCAN
        self.useIterative_ = useIterative
        self.minT = iminT
        self.eps = ieps
        self.errmax = ierrmax
        self.chi2max = ichi2max

    def make(
        self, tksoa: Self.TkSoA, ptMin: Float
    ) raises -> ZVertexHeterogeneous:
        var vertices: ZVertexHeterogeneous = ZVertexHeterogeneous(ZVertexSoA())
        # C++ asserts tksoa/soa are non-null; borrows cannot be null.
        ref soa = vertices[]

        # C++ heap-allocates this (std::make_unique) because its columns are
        # inline arrays; here they are Lists, so WorkSpace is 160 B and a local
        # is cheaper than a box.
        var ws_d = WorkSpace()

        init(soa, ws_d)
        loadTracks(tksoa, soa, ws_d, ptMin)

        if self.useDensity_:
            clusterTracksByDensity(
                soa, ws_d, self.minT, self.eps, self.errmax, self.chi2max
            )
        elif self.useDBSCAN_:
            raise "NotImplementedError: clusterTracksDBSCAN is not yet ported"
        elif self.useIterative_:
            raise "NotImplementedError: clusterTracksIterative is not yet ported"

        fitVertices(soa, ws_d, 50.0)
        # one block per vertex!
        splitVertices(soa, ws_d, 9.0)
        fitVertices(soa, ws_d, 5000.0)
        sortByPt2(soa, ws_d)

        return vertices^

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "Producer"
