from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EDProducer import EDProducer
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.EDGetToken import EDGetTokenT
from MojoSerial.Framework.EDPutToken import EDPutTokenT
from MojoSerial.Framework.ProductRegistry import ProductRegistry

from MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DCPU,
)
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import (
    PixelTrackHeterogeneous,
)
from MojoSerial.plugin_PixelTriplets.CAHitNtupletGeneratorOnGPU import (
    CAHitNtupletGeneratorOnGPU,
)
from MojoSerial.CondFormats.PixelCPEFast import PixelCPEFast
from MojoSerial.MojoBridge.DTypes import Typeable


struct CAHitNtupletCUDA(Defaultable, EDProducer, Typeable):
    var tokenHitCPU_: EDGetTokenT[TrackingRecHit2DCPU]
    var tokenTrackCPU_: EDPutTokenT[PixelTrackHeterogeneous]

    var gpuAlgo_: CAHitNtupletGeneratorOnGPU

    def __init__(out self):
        self.tokenHitCPU_ = EDGetTokenT[TrackingRecHit2DCPU]()
        self.tokenTrackCPU_ = EDPutTokenT[PixelTrackHeterogeneous]()
        var dummy_reg = ProductRegistry()
        self.gpuAlgo_ = CAHitNtupletGeneratorOnGPU(dummy_reg)

    def __init__(out self, mut reg: ProductRegistry):
        try:
            self.tokenHitCPU_ = reg.consumes[TrackingRecHit2DCPU]()
            self.tokenTrackCPU_ = reg.produces[PixelTrackHeterogeneous]()
        except e:
            print("Handled exception in CAHitNtupletCUDA, ", e)
            self.tokenHitCPU_ = EDGetTokenT[TrackingRecHit2DCPU]()
            self.tokenTrackCPU_ = EDPutTokenT[PixelTrackHeterogeneous]()

        self.gpuAlgo_ = CAHitNtupletGeneratorOnGPU(reg)

    def produce(mut self, mut iEvent: Event, ref es: EventSetup):
        var bf: Float32 = 0.0114256972711507  # 1/fieldInGeV
        ref hits = iEvent.get(self.tokenHitCPU_)

        try:
            # C++ gets the CPE params from the hits view; see make_tuples.
            iEvent.put(
                self.tokenTrackCPU_,
                self.gpuAlgo_.make_tuples(hits, es.get[PixelCPEFast](), bf),
            )
        except e:
            print("Error during produce in CAHitNtupletCUDA, ", e)

    def endJob(mut self) raises:
        pass

    @staticmethod
    def dtype() -> String:
        return "CAHitNtupletCUDA"
