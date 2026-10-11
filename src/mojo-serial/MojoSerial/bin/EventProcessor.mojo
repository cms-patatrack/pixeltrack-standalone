from std.pathlib import Path

from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.MojoBridge.DTypes import Typeable
from MojoSerial.bin.Plugins import ed_path, es_path, es_produce, make_es
from MojoSerial.bin.Source import Source
from MojoSerial.bin.StreamSchedule import StreamSchedule


struct EventProcessor(Defaultable, Typeable):
    # no pluginmanager
    var _registry: ProductRegistry
    var _source: Source
    var _eventSetup: EventSetup
    var _schedule: StreamSchedule
    var _warmupEvents: Int32
    var _startEvent: Int32
    var _endEvent: Int32
    var _runForMinutes: Int32

    @always_inline
    def __init__(out self):
        self._registry = ProductRegistry()
        self._source = Source()
        self._eventSetup = EventSetup()
        self._schedule = StreamSchedule()
        self._warmupEvents = 0
        self._startEvent = 0
        self._endEvent = 0
        self._runForMinutes = 0

    def __init__(
        out self,
        var warmupEvents: Int,
        var startEvent: Int,
        var endEvent: Int,
        var runForMinutes: Int,
        var path: Path,
        var validation: Bool,
    ):
        try:
            self._registry = ProductRegistry()
            self._source = Source(
                Int32(startEvent),
                Int32(endEvent),
                Int32(runForMinutes),
                self._registry,
                path,
                validation,
            )
            self._eventSetup = EventSetup()
            self._warmupEvents = Int32(warmupEvents)
            self._startEvent = Int32(startEvent)
            self._endEvent = Int32(endEvent)
            self._runForMinutes = Int32(runForMinutes)

            var esp_names = es_path()
            for i in range(len(esp_names)):
                var esp = make_es(esp_names[i], path)
                es_produce(esp, self._eventSetup)

            self._schedule = StreamSchedule(
                self._registry, ed_path(validation)
            )
        except e:
            print("Error occurred in bin/EventProcessor.mojo,", e)
            return Self()

    @always_inline
    def warmUp(mut self):
        if self._warmupEvents <= 0:
            return

        self._source.reconfigure(
            self._startEvent,
            self._startEvent + self._warmupEvents,
            -1,
        )
        self.process()

    @always_inline
    def runToCompletion(mut self):
        self._source.reconfigure(
            self._startEvent,
            self._endEvent,
            self._runForMinutes,
        )
        self.process()

    @always_inline
    def process(mut self):
        self._source.startProcessing()
        self._schedule.run(self._registry, self._source, self._eventSetup)

    @always_inline
    def endJob(mut self) raises:
        self._schedule.endJob()

    @always_inline
    def processedEvents(self) -> Int32:
        return self._source.processedEvents()

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "EventProcessor"
