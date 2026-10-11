from MojoSerial.CUDACore.HistoContainer import OneToManyAssoc
from MojoSerial.CUDACore.EigenSoA import ScalarSoA
from MojoSerial.CUDADataFormats.HeterogeneousSoA import HeterogeneousSoA
from MojoSerial.CUDADataFormats.TrajectoryStateSoA import TrajectoryStateSoA
from MojoSerial.MojoBridge.Matrix import Vector, Matrix
from MojoSerial.MojoBridge.DTypes import Float, Typeable
struct TrackQuality:
    comptime bad: UInt8 = 0
    comptime dup: UInt8 = 1
    comptime loose: UInt8 = 2
    comptime strict: UInt8 = 3
    comptime tight: UInt8 = 4
    comptime highPurity: UInt8 = 5


# WARNING: THIS STRUCT IS 128-ALIGNED (ScalarSoA)
@fieldwise_init
struct TrackSoAT[S: Int32](Defaultable, Movable, Typeable):
    @staticmethod
    @always_inline
    def stride() -> Int32:
        return Self.S

    comptime Quality = UInt8
    comptime HIndexType = DType.uint16
    comptime HitContainer = OneToManyAssoc[
        Self.HIndexType,
        Self.S.cast[DType.uint32](),
        5 * Self.S.cast[DType.uint32](),
    ]

    var m_quality: ScalarSoA[DType.uint8, Int(Self.S)]

    # this is chi2/ndof as not necessarely all hits are used in the fit
    var chi2: ScalarSoA[DType.float32, Int(Self.S)]

    # State at the Beam spot
    # phi,tip,1/pt,cotan(theta),zip
    var stateAtBS: TrajectoryStateSoA[Self.S]
    var eta: ScalarSoA[DType.float32, Int(Self.S)]
    var pt: ScalarSoA[DType.float32, Int(Self.S)]

    # state at the detector of the outermost hit
    var hitIndices: Self.HitContainer
    var detIndices: Self.HitContainer

    # total number of tracks (including those not fitted)
    var m_nTracks: UInt32

    @always_inline
    def __init__(out self):
        self.m_quality = ScalarSoA[DType.uint8, Int(Self.S)]()

        self.chi2 = ScalarSoA[DType.float32, Int(Self.S)]()

        self.stateAtBS = TrajectoryStateSoA[Self.S]()
        self.eta = ScalarSoA[DType.float32, Int(Self.S)]()
        self.pt = ScalarSoA[DType.float32, Int(Self.S)]()

        self.hitIndices = Self.HitContainer()
        self.detIndices = Self.HitContainer()
        self.m_nTracks = 0

    @always_inline
    def quality(ref self, i: Int) -> ref [self.m_quality._data] Self.Quality:
        return self.m_quality[i]

    @always_inline
    def qualityData(
        ref self,
    ) -> Span[Self.Quality, origin_of(self.m_quality._data)]:
        return Span(self.m_quality._data)

    @always_inline
    def nHits(self, i: Int32) -> Int32:
        return self.detIndices.size(i.cast[DType.uint32]()).cast[DType.int32]()

    @always_inline
    def charge(self, i: Int32) -> Float:
        return Float(1.0) if self.stateAtBS.state[i][2, 0] >= 0 else Float(-1.0)

    @always_inline
    def phi(self, i: Int32) -> Float:
        return rebind[Scalar[DType.float32]](self.stateAtBS.state[i][0, 0])

    @always_inline
    def tip(self, i: Int32) -> Float:
        return rebind[Scalar[DType.float32]](self.stateAtBS.state[i][1, 0])

    @always_inline
    def zip(self, i: Int32) -> Float:
        return rebind[Scalar[DType.float32]](self.stateAtBS.state[i][4, 0])

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "TrackSoAT[" + String(Self.S) + "]"
struct PixelTrack:
    @staticmethod
    @always_inline
    def maxNumber() -> UInt32:
        return 32 * 1024

    comptime TrackSoA = TrackSoAT[Self.maxNumber().cast[DType.int32]()]
    comptime TrajectoryState = TrajectoryStateSoA[
        Self.maxNumber().cast[DType.int32]()
    ]
    comptime HitContainer = Self.TrackSoA.HitContainer
    comptime Quality = UInt8


comptime PixelTrackHeterogeneous = HeterogeneousSoA[PixelTrack.TrackSoA]
