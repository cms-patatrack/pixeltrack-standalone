from std.atomic import Atomic
from std.memory import ArcPointer

from MojoSerial.Framework.EDProducer import EDProducer
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.EDGetToken import EDGetTokenT
from MojoSerial.DataFormats.DigiClusterCount import DigiClusterCount
from MojoSerial.DataFormats.TrackCount import TrackCount
from MojoSerial.DataFormats.VertexCount import VertexCount
from MojoSerial.CUDADataFormats.SiPixelDigisSoA import SiPixelDigisSoA
from MojoSerial.CUDADataFormats.SiPixelClustersSoA import SiPixelClustersSoA
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import PixelTrackHeterogeneous
from MojoSerial.CUDADataFormats.ZVertexHeterogeneous import ZVertexHeterogeneous

from MojoSerial.MojoBridge.DTypes import Typeable


# C++: the counters are globals in an anonymous namespace, updated by the
# CountValidator of every stream. Mojo has no global var: one of these is
# shared by all the streams through PluginGlobals (bin/Plugins.mojo).
# The fields are updated directly, not through `mut self` methods: those could
# update a copy of the struct (see doc §12).
struct CountValidatorCounters(Movable):
    var allEvents: Atomic[DType.int32]
    var goodEvents: Atomic[DType.int32]
    var sumVertexDifference: Atomic[DType.int32]
    # C++ protects the sum with a mutex
    var sumTrackDifference: Atomic[DType.float32]
    # The number of streams that have not reached endJob yet; the last one
    # reports the results, once all the events have been processed.
    var pending: Atomic[DType.int32]

    def __init__(out self, streams: Int):
        self.allEvents = Atomic[DType.int32](0)
        self.goodEvents = Atomic[DType.int32](0)
        self.sumVertexDifference = Atomic[DType.int32](0)
        self.sumTrackDifference = Atomic[DType.float32](0)
        self.pending = Atomic[DType.int32](Int32(streams))


struct CountValidator (
    Defaultable, EDProducer, Movable, Typeable
):

    var digiClusterCountToken_: EDGetTokenT[DigiClusterCount]
    var trackCountToken_: EDGetTokenT[TrackCount]
    var vertexCountToken_: EDGetTokenT[VertexCount]

    var digiToken_: EDGetTokenT[SiPixelDigisSoA]
    var clusterToken_: EDGetTokenT[SiPixelClustersSoA]
    var trackToken_: EDGetTokenT[PixelTrackHeterogeneous]
    var vertexToken_: EDGetTokenT[ZVertexHeterogeneous]

    var counters_: ArcPointer[CountValidatorCounters]

    def __init__(out self):
        self.digiClusterCountToken_ = EDGetTokenT[DigiClusterCount]()
        self.trackCountToken_ = EDGetTokenT[TrackCount]()
        self.vertexCountToken_ = EDGetTokenT[VertexCount]()

        self.digiToken_ = EDGetTokenT[SiPixelDigisSoA]()
        self.clusterToken_ = EDGetTokenT[SiPixelClustersSoA]()
        self.trackToken_ = EDGetTokenT[PixelTrackHeterogeneous]()
        self.vertexToken_ = EDGetTokenT[ZVertexHeterogeneous]()

        self.counters_ = ArcPointer(CountValidatorCounters(1))

    # A CountValidator with its own counters, for a single stream.
    def __init__(out self, mut reg: ProductRegistry):
        self = Self(reg, ArcPointer(CountValidatorCounters(1)))

    # A CountValidator updating the counters shared by all the streams.
    def __init__(
        out self,
        mut reg: ProductRegistry,
        counters: ArcPointer[CountValidatorCounters],
    ):
        self.counters_ = counters
        try:
            self.digiClusterCountToken_ = reg.consumes[DigiClusterCount]()
            self.trackCountToken_ = reg.consumes[TrackCount]()
            self.vertexCountToken_ = reg.consumes[VertexCount]()
            self.digiToken_ = reg.consumes[SiPixelDigisSoA]()
            self.clusterToken_ = reg.consumes[SiPixelClustersSoA]()
            self.trackToken_ = reg.consumes[PixelTrackHeterogeneous]()
            self.vertexToken_ = reg.consumes[ZVertexHeterogeneous]()
        except e:
            print("Handled exception in CountValidator: ", e)
            return Self()

    @staticmethod
    def strformat[Ty: Writable](str: StringSlice, arg: Ty) -> String:
        # Simple implementation of format function
        try:
            return str.format(arg)
        except e:
            return str + " <== Format error: " + String(e)

    @staticmethod
    def strformat[Ty: Writable](str: StringSlice, arg: Ty, arg1: Ty) -> String:
        # Simple implementation of format function
        try:
            return str.format(arg, arg1)
        except e:
            return str + " <== Format error: " + String(e)

    def produce(mut self, mut iEvent: Event, ref iSetup: EventSetup):
        var trackTolerance: Float32 = 0.012  # in 200 runs of 1k events all events are withing this tolerance
        var vertexTolerance: Int32 = 1

        ref counters = self.counters_[]

        var errorMsg: String = Self.strformat("Event {} ", iEvent.eventID())
        var ok: Bool = True

        ref count = iEvent.get(self.digiClusterCountToken_)
        ref digis = iEvent.get(self.digiToken_)
        ref clusters = iEvent.get(self.clusterToken_)

        if digis.nModules() != count.nModules():
            errorMsg += Self.strformat("\n N(modules) is {} expected {}", digis.nModules(), count.nModules())
            ok = False

        if digis.nDigis() != count.nDigis():
            errorMsg += Self.strformat("\n N(digis) is {} expected {}", digis.nDigis(), count.nDigis())
            ok = False
        if clusters.nClusters() != count.nClusters():
            errorMsg += Self.strformat("\n N(clusters) is {} expected {}", clusters.nClusters(), count.nClusters())
            ok = False

        ref trackCount = iEvent.get(self.trackCountToken_)
        ref trackWrapper = iEvent.get(self.trackToken_)
        ref tracks = trackWrapper[]

        var nTracks: Int32 = 0
        for i in range(tracks.stride()):
            if tracks.nHits(Int32(i)) > 0:
                nTracks += 1

        var rel = abs(
            Float32(nTracks - Int32(trackCount.nTracks()))
            / Float32(trackCount.nTracks())
        )
        if UInt32(nTracks) != trackCount.nTracks():
            _ = counters.sumTrackDifference.fetch_add(rel)
        if rel >= trackTolerance:
            errorMsg += Self.strformat(
                "\n N(tracks) is {} expected {}",
                UInt32(nTracks),
                trackCount.nTracks(),
            )
            errorMsg += Self.strformat(", relative difference {} is outside tolerance {}", rel, trackTolerance)
            ok = False

        ref vertexCount = iEvent.get(self.vertexCountToken_)
        ref vertexWrapper = iEvent.get(self.vertexToken_)
        ref vertices = vertexWrapper[]

        var diff: Int32 = abs(Int32(vertices.nvFinal) - Int32(vertexCount.nVertices()))
        if diff != 0:
            _ = counters.sumVertexDifference.fetch_add(diff)
        if diff > vertexTolerance:
            errorMsg += Self.strformat(
                "\n N(vertices) is {} expected {}",
                vertices.nvFinal,
                vertexCount.nVertices(),
            )
            errorMsg += Self.strformat(", difference {} is outside tolerance {}", diff, vertexTolerance)
            ok = False

        _ = counters.allEvents.fetch_add(1)
        if ok:
            _ = counters.goodEvents.fetch_add(1)
        else:
            print(errorMsg)

    # C++ calls endJob once, on the first stream, after all the streams are
    # done; here every stream calls it, and the last one reports the results.
    def endJob(mut self) raises:
        ref counters = self.counters_[]
        if counters.pending.fetch_sub(1) != 1:
            return
        var all = counters.allEvents.load()
        var good = counters.goodEvents.load()
        var sumTrackDifference = counters.sumTrackDifference.load()
        var sumVertexDifference = counters.sumVertexDifference.load()
        if all == good:
            print(Self.strformat("CountValidator: all {} events validated successfully!", all))
            if sumTrackDifference > 0:
                print(
                    Self.strformat(
                        " Average relative track difference: {} (all within tolerance)",
                        sumTrackDifference / Float32(all)
                    )
                )
            if sumVertexDifference > 0:
                print(
                    Self.strformat(
                        " Average absolute vertex difference: {} (all within tolerance)",
                        Float32(sumVertexDifference) / Float32(all)
                    )
                )
        else:
            print(
                Self.strformat(
                    "CountValidator: {} events failed validation (see details above)",
                    all - good
                )
            )
            raise Error("CountValidator failed")

    @staticmethod
    def dtype() -> String:
        return "CountValidator"
