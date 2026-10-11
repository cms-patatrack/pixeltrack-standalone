from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EDProducer import EDProducer
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.Framework.EDPutToken import EDPutTokenT

from MojoSerial.DataFormats.BeamSpotPOD import BeamSpotPOD
from MojoSerial.MojoBridge.DTypes import Typeable


struct BeamSpotToPOD(Defaultable, EDProducer, Typeable):
    var bsPutToken_: EDPutTokenT[BeamSpotPOD]

    def __init__(out self):
        self.bsPutToken_ = EDPutTokenT[BeamSpotPOD]()

    def __init__(out self, mut reg: ProductRegistry):
        try:
            self.bsPutToken_ = reg.produces[BeamSpotPOD]()
        except e:
            print("Handled exception in BeamSpotToPOD, ", e)
            return Self()

    def produce(mut self, mut iEvent: Event, ref iSetup: EventSetup):
        try:
            iEvent.put[BeamSpotPOD](
                self.bsPutToken_,
                iSetup.get[BeamSpotPOD](),
            )
        except e:
            print("Error during produce in BeamSpotToPOD, ", e)

    def endJob(mut self) raises:
        pass

    @staticmethod
    def dtype() -> String:
        return "BeamSpotToPOD"
