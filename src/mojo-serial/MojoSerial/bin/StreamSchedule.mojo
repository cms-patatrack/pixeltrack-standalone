from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.MojoBridge.DTypes import Typeable
from MojoSerial.bin.Plugins import EDPlugin, ed_end_job, ed_produce, make_ed
from MojoSerial.bin.Source import Source


struct StreamSchedule(Defaultable, Movable, Typeable):
    # C++ stores Source*/EventSetup* pointing at sibling fields of its owner;
    # passed to run() instead so the struct is not self-referential.
    var _path: List[EDPlugin]
    var _streamId: Int32

    @always_inline
    def __init__(out self):
        self._path = []
        self._streamId = 0

    def __init__(
        out self,
        mut reg: ProductRegistry,
        path: List[String],
        streamId: Int32 = 0,
    ):
        try:
            self._streamId = streamId
            debug_assert(len(path) > 0)

            # C++ StreamSchedule.cc:23 -- construct in the given order. Each
            # ctor calls reg.consumes/produces, which is what beginModuleConstruction
            # records; the order itself comes from `path`, not from the graph.
            self._path = List[EDPlugin](capacity=len(path))
            for i in range(len(path)):
                reg.beginModuleConstruction(Int32(i) + 1)
                self._path.append(make_ed(path[i], reg))
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
                ed_produce(self._path[i], event, eventSetup)
            next = source.produce(self._streamId, reg)

    def endJob(mut self) raises:
        for i in range(self._path.__len__()):
            ed_end_job(self._path[i])

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "StreamSchedule"
