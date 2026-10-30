from std.memory import ArcPointer

from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.MojoBridge.DTypes import Typeable


# The name returned by dtype() is the name of the plugin.
trait EDProducer(Defaultable, Movable, Typeable):
    # this cannot raise
    def __init__(out self, mut reg: ProductRegistry):
        ...

    def produce(mut self, mut event: Event, ref eventSetup: EventSetup):
        ...

    def endJob(mut self) raises:
        ...


# An EDProducer whose C++ version keeps some state in static or global
# variables, shared by the instances of all the streams. Mojo has no global
# variables: the plugin factory creates one `Globals` when the plugin is loaded,
# and passes it to each instance it creates.
trait EDProducerWithGlobals(EDProducer):
    comptime Globals: Movable & Deinitable

    # `streams` is the number of streams, each with its own instance.
    @staticmethod
    def makeGlobals(streams: Int) -> Self.Globals:
        ...

    # this cannot raise
    def __init__(
        out self, mut reg: ProductRegistry, globals: ArcPointer[Self.Globals]
    ):
        ...
