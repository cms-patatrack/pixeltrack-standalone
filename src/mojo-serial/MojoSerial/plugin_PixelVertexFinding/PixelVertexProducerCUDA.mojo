from MojoSerial.CUDADataFormats.ZVertexHeterogeneous import ZVertexHeterogeneous
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import (
    PixelTrackHeterogeneous,
)
from MojoSerial.Framework.EDGetToken import EDGetTokenT
from MojoSerial.Framework.EDProducer import EDProducer
from MojoSerial.Framework.EDPutToken import EDPutTokenT
from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.MojoBridge.DTypes import Float, Typeable
from MojoSerial.plugin_PixelVertexFinding.gpuVertexFinder import Producer, TkSoA


struct PixelVertexProducerCUDA(Defaultable, EDProducer, Typeable):
    var _tokenCPUTrack: EDGetTokenT[PixelTrackHeterogeneous]
    var _tokenCPUVertex: EDPutTokenT[ZVertexHeterogeneous]

    var _gpuAlgo: Producer

    # Tracking cuts before sending tracks to vertex algo
    var _ptMin: Float

    @always_inline
    def __init__(out self):
        self._tokenCPUTrack = EDGetTokenT[PixelTrackHeterogeneous]()
        self._tokenCPUVertex = EDPutTokenT[ZVertexHeterogeneous]()
        self._gpuAlgo = Producer(
            True,  # oneKernel
            True,  # useDensity
            False,  # useDBSCAN
            False,  # useIterative
            2,  # minT
            0.07,  # eps
            0.01,  # errmax
            9.0,  # chi2max
        )
        self._ptMin = 0.5  # 0.5 GeV

    @always_inline
    def __init__(out self, mut reg: ProductRegistry):
        try:
            self._tokenCPUTrack = reg.consumes[PixelTrackHeterogeneous]()
            self._tokenCPUVertex = reg.produces[ZVertexHeterogeneous]()

            self._gpuAlgo = Producer(
                True,  # oneKernel
                True,  # useDensity
                False,  # useDBSCAN
                False,  # useIterative
                2,  # minT
                0.07,  # eps
                0.01,  # errmax
                9.0,  # chi2max
            )
            self._ptMin = 0.5  # 0.5 GeV
        except e:
            print("Handled exception in PixelVertexProducerCUDA, ", e)
            return Self()

    def produce(mut self, mut iEvent: Event, ref iSetup: EventSetup):
        try:
            ref tracks = iEvent.get[PixelTrackHeterogeneous](
                self._tokenCPUTrack
            )
            # C++ asserts tksoa is non-null; a borrow cannot be null.
            var vertices = self._gpuAlgo.make(tracks[], self._ptMin)
            iEvent.put[ZVertexHeterogeneous](self._tokenCPUVertex, vertices^)
        except e:
            print("Error during produce in PixelVertexProducerCUDA, ", e)

    def endJob(mut self) raises:
        pass

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "PixelVertexProducerCUDA"
