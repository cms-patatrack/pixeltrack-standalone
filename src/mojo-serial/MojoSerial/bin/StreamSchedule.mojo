from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.MojoBridge.DTypes import Typeable
from MojoSerial.Framework.PluginFactory import Worker
from MojoSerial.bin.PluginManager import PluginManager
from MojoSerial.bin.Source import Source


struct StreamSchedule(Defaultable, Movable, Typeable):
    # C++ stores Source*/EventSetup* pointing at sibling fields of its owner;
    # passed to run() instead so the struct is not self-referential.
    var _path: List[Worker]
    var _streamId: Int32

    @always_inline
    def __init__(out self):
        self._path = []
        self._streamId = 0

    def __init__(
        out self,
        mut reg: ProductRegistry,
        ref pluginManager: PluginManager,
        path: List[String],
        streamId: Int32 = 0,
    ):
        try:
            self._streamId = streamId
            # C++ StreamSchedule.cc:23 -- construct in the given order. Each
            # ctor calls reg.consumes/produces, which is what beginModuleConstruction
            # records; the order itself comes from `path`, not from the graph.
            self._path = List[Worker](capacity=len(path))
            for i in range(len(path)):
                reg.beginModuleConstruction(Int32(i) + 1)
                self._path.append(pluginManager.create(path[i], reg))
        except e:
            print("Error occurred in bin/StreamSchedule.mojo,", e)
            return Self()

    def run(
        mut self,
        ref reg: ProductRegistry,
        mut source: Source,
        ref eventSetup: EventSetup,
    ):
        var next = source.produce(self._streamId, reg)
        while next:
            var event = next.take()
            for i in range(self._path.__len__()):
                self._path[i].produce(event, eventSetup)
            next = source.produce(self._streamId, reg)

    def endJob(mut self) raises:
        for i in range(self._path.__len__()):
            self._path[i].endJob()

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "StreamSchedule"
