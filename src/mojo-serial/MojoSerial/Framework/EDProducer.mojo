from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.ProductRegistry import ProductRegistry


trait EDProducer(Defaultable):
    # this cannot raise
    def __init__(out self, mut reg: ProductRegistry):
        ...

    def produce(mut self, mut event: Event, ref eventSetup: EventSetup):
        ...

    def endJob(mut self) raises:
        ...
