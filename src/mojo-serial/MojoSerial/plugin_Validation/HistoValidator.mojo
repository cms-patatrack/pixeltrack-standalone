from std.atomic import Atomic
from std.memory import ArcPointer

from MojoSerial.Framework.EDProducer import EDProducer
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.EDGetToken import EDGetTokenT
from MojoSerial.CUDADataFormats.SiPixelDigisSoA import SiPixelDigisSoA
from MojoSerial.CUDADataFormats.SiPixelClustersSoA import SiPixelClustersSoA
from MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DCPU,
)
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import (
    PixelTrackHeterogeneous,
    TrackQuality,
)
from MojoSerial.CUDADataFormats.ZVertexHeterogeneous import ZVertexHeterogeneous
from MojoSerial.MojoBridge.DTypes import Typeable
from MojoSerial.plugin_Validation.SimpleAtomicHisto import SimpleAtomicHisto


# C++: `static std::map<std::string, SimpleAtomicHisto> HistoValidator::histos`,
# filled by the HistoValidator of every stream. Mojo has no global var: one of
# these is shared by all the streams through PluginGlobals (bin/Plugins.mojo).
# The histograms are booked in the order of their names, which is the order in
# which std::map iterates over them, and are addressed by index.
struct HistoValidatorHistos(Movable):
    comptime cluster_n = 0
    comptime cluster_per_module_n = 1
    comptime digi_adc = 2
    comptime digi_n = 3
    comptime hit_charge = 4
    comptime hit_gr = 5
    comptime hit_gx = 6
    comptime hit_gy = 7
    comptime hit_gz = 8
    comptime hit_lex = 9
    comptime hit_ley = 10
    comptime hit_lx = 11
    comptime hit_ly = 12
    comptime hit_n = 13
    comptime hit_sizex = 14
    comptime hit_sizey = 15
    comptime module_n = 16
    comptime track_chi2 = 17
    comptime track_eta = 18
    comptime track_n = 19
    comptime track_nhits = 20
    comptime track_phi = 21
    comptime track_pt = 22
    comptime track_quality = 23
    comptime track_tip = 24
    comptime track_tip_zoom = 25
    comptime track_zip = 26
    comptime track_zip_zoom = 27
    comptime vertex_chi2 = 28
    comptime vertex_n = 29
    comptime vertex_ndof = 30
    comptime vertex_pt2 = 31
    comptime vertex_z = 32

    var names: List[String]
    var histos: List[SimpleAtomicHisto]
    # The number of streams that have not reached endJob yet; the last one
    # writes the histograms, once all the events have been processed.
    var pending_: Atomic[DType.int32]

    def __init__(out self, streams: Int):
        self.names = []
        self.histos = []
        self.pending_ = Atomic[DType.int32](Int32(streams))

        self._book(Self.cluster_n, "cluster_n", 200, 5000, 25000)
        self._book(
            Self.cluster_per_module_n, "cluster_per_module_n", 110, 0, 110
        )
        self._book(Self.digi_adc, "digi_adc", 250, 0, 5e4)
        self._book(Self.digi_n, "digi_n", 100, 0, 1e5)
        self._book(Self.hit_charge, "hit_charge", 400, 0, 4e6)
        self._book(Self.hit_gr, "hit_gr", 200, 0, 20)
        self._book(Self.hit_gx, "hit_gx", 200, -20, 20)
        self._book(Self.hit_gy, "hit_gy", 200, -20, 20)
        self._book(Self.hit_gz, "hit_gz", 600, -60, 60)
        self._book(Self.hit_lex, "hit_lex", 100, 0, 5e-5)
        self._book(Self.hit_ley, "hit_ley", 100, 0, 1e-4)
        self._book(Self.hit_lx, "hit_lx", 200, -1, 1)
        self._book(Self.hit_ly, "hit_ly", 800, -4, 4)
        self._book(Self.hit_n, "hit_n", 200, 5000, 25000)
        self._book(Self.hit_sizex, "hit_sizex", 800, 0, 800)
        self._book(Self.hit_sizey, "hit_sizey", 800, 0, 800)
        self._book(Self.module_n, "module_n", 100, 1500, 2000)
        self._book(Self.track_chi2, "track_chi2", 100, 0, 40)
        self._book(Self.track_eta, "track_eta", 100, -3, 3)
        self._book(Self.track_n, "track_n", 150, 0, 15000)
        self._book(Self.track_nhits, "track_nhits", 3, 3, 6)
        self._book(Self.track_phi, "track_phi", 100, -3.15, 3.15)
        self._book(Self.track_pt, "track_pt", 400, 0, 400)
        self._book(Self.track_quality, "track_quality", 6, 0, 6)
        self._book(Self.track_tip, "track_tip", 100, -1, 1)
        self._book(Self.track_tip_zoom, "track_tip_zoom", 100, -0.05, 0.05)
        self._book(Self.track_zip, "track_zip", 100, -15, 15)
        self._book(Self.track_zip_zoom, "track_zip_zoom", 100, -0.1, 0.1)
        self._book(Self.vertex_chi2, "vertex_chi2", 100, 0, 40)
        self._book(Self.vertex_n, "vertex_n", 60, 0, 60)
        self._book(Self.vertex_ndof, "vertex_ndof", 170, 0, 170)
        self._book(Self.vertex_pt2, "vertex_pt2", 100, 0, 4000)
        self._book(Self.vertex_z, "vertex_z", 100, -15, 15)

    def _book(
        mut self,
        index: Int,
        var name: String,
        nbins: Int,
        min: Float32,
        max: Float32,
    ):
        debug_assert(index == len(self.histos), "histogram booked out of order")
        if self.names:
            debug_assert(
                self.names[len(self.names) - 1] < name,
                "histograms not sorted by name",
            )
        self.names.append(name^)
        self.histos.append(SimpleAtomicHisto(nbins, min, max))

    @always_inline
    def fill(self, index: Int, value: Float32) raises:
        self.histos[index].fill(value)


struct HistoValidator(Defaultable, EDProducer, Movable, Typeable):
    comptime H = HistoValidatorHistos

    var digiToken_: EDGetTokenT[SiPixelDigisSoA]
    var clusterToken_: EDGetTokenT[SiPixelClustersSoA]
    var hitToken_: EDGetTokenT[TrackingRecHit2DCPU]
    var trackToken_: EDGetTokenT[PixelTrackHeterogeneous]
    var vertexToken_: EDGetTokenT[ZVertexHeterogeneous]

    var histos_: ArcPointer[HistoValidatorHistos]

    def __init__(out self):
        self.digiToken_ = EDGetTokenT[SiPixelDigisSoA]()
        self.clusterToken_ = EDGetTokenT[SiPixelClustersSoA]()
        self.hitToken_ = EDGetTokenT[TrackingRecHit2DCPU]()
        self.trackToken_ = EDGetTokenT[PixelTrackHeterogeneous]()
        self.vertexToken_ = EDGetTokenT[ZVertexHeterogeneous]()
        self.histos_ = ArcPointer(HistoValidatorHistos(1))

    # A HistoValidator with its own histograms, for a single stream.
    def __init__(out self, mut reg: ProductRegistry):
        self = Self(reg, ArcPointer(HistoValidatorHistos(1)))

    # A HistoValidator filling the histograms shared by all the streams.
    def __init__(
        out self,
        mut reg: ProductRegistry,
        histos: ArcPointer[HistoValidatorHistos],
    ):
        self.histos_ = histos
        try:
            self.digiToken_ = reg.consumes[SiPixelDigisSoA]()
            self.clusterToken_ = reg.consumes[SiPixelClustersSoA]()
            self.hitToken_ = reg.consumes[TrackingRecHit2DCPU]()
            self.trackToken_ = reg.consumes[PixelTrackHeterogeneous]()
            self.vertexToken_ = reg.consumes[ZVertexHeterogeneous]()
        except e:
            print("Handled exception in HistoValidator: ", e)
            return Self()

    def produce(mut self, mut iEvent: Event, ref iSetup: EventSetup):
        try:
            self._produce(iEvent)
        except e:
            print("Error during produce in HistoValidator, ", e)

    def _produce(self, mut iEvent: Event) raises:
        ref histos = self.histos_[]

        ref digis = iEvent.get(self.digiToken_)
        ref clusters = iEvent.get(self.clusterToken_)

        var nDigis = digis.nDigis()
        var nModules = digis.nModules()

        var nClusters = clusters.nClusters()

        ref hits = iEvent.get(self.hitToken_)

        histos.fill(Self.H.digi_n, Float32(nDigis))
        for i in range(Int(nDigis)):
            histos.fill(Self.H.digi_adc, Float32(digis.adc(i)))
        histos.fill(Self.H.module_n, Float32(nModules))

        histos.fill(Self.H.cluster_n, Float32(nClusters))
        for i in range(Int(nModules)):
            histos.fill(
                Self.H.cluster_per_module_n, Float32(clusters.clusInModule(i))
            )

        var nHits = hits.nHits()
        histos.fill(Self.H.hit_n, Float32(nHits))
        for i in range(Int(nHits)):
            histos.fill(Self.H.hit_lx, hits.xLocal(i))
            histos.fill(Self.H.hit_ly, hits.yLocal(i))
            histos.fill(Self.H.hit_lex, hits.xerrLocal(i))
            histos.fill(Self.H.hit_ley, hits.yerrLocal(i))
            histos.fill(Self.H.hit_gx, hits.xGlobal(i))
            histos.fill(Self.H.hit_gy, hits.yGlobal(i))
            histos.fill(Self.H.hit_gz, hits.zGlobal(i))
            histos.fill(Self.H.hit_gr, hits.rGlobal(i))
            histos.fill(Self.H.hit_charge, Float32(hits.charge(i)))
            histos.fill(Self.H.hit_sizex, Float32(hits.clusterSizeX(i)))
            histos.fill(Self.H.hit_sizey, Float32(hits.clusterSizeY(i)))

        ref tracks = iEvent.get(self.trackToken_)[]

        var nTracks = 0
        for i in range(Int(tracks.stride())):
            var nHitsTrack = tracks.nHits(Int32(i))
            if nHitsTrack > 0 and tracks.quality(i) >= TrackQuality.loose:
                nTracks += 1
                histos.fill(Self.H.track_nhits, Float32(nHitsTrack))
                histos.fill(Self.H.track_chi2, tracks.chi2[i])
                histos.fill(Self.H.track_pt, tracks.pt[i])
                histos.fill(Self.H.track_eta, tracks.eta[i])
                histos.fill(Self.H.track_phi, tracks.phi(Int32(i)))
                histos.fill(Self.H.track_tip, tracks.tip(Int32(i)))
                histos.fill(Self.H.track_tip_zoom, tracks.tip(Int32(i)))
                histos.fill(Self.H.track_zip, tracks.zip(Int32(i)))
                histos.fill(Self.H.track_zip_zoom, tracks.zip(Int32(i)))
                histos.fill(Self.H.track_quality, Float32(tracks.quality(i)))

        histos.fill(Self.H.track_n, Float32(nTracks))

        ref vertices = iEvent.get(self.vertexToken_)[]

        histos.fill(Self.H.vertex_n, Float32(vertices.nvFinal))
        for i in range(Int(vertices.nvFinal)):
            histos.fill(Self.H.vertex_z, vertices.zv[i])
            histos.fill(Self.H.vertex_chi2, vertices.chi2[i])
            histos.fill(Self.H.vertex_ndof, Float32(vertices.ndof[i]))
            histos.fill(Self.H.vertex_pt2, vertices.ptv2[i])

    # C++ calls endJob once, on the first stream, after all the streams are
    # done; here every stream calls it, and the last one writes the histograms.
    def endJob(mut self) raises:
        ref histos = self.histos_[]
        # through the field, not a `mut self` method: that could update a copy
        # of the struct (see doc §12)
        if histos.pending_.fetch_sub(1) != 1:
            return
        with open("histograms_mojo-serial.txt", "w") as out:
            for i in range(len(histos.histos)):
                out.write(histos.names[i], " ", histos.histos[i], "\n")

    @staticmethod
    def dtype() -> String:
        return "HistoValidator"
