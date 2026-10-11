from std.time import perf_counter_ns
from std.pathlib import Path

from MojoSerial.DataFormats.FEDRawDataCollection import FEDRawDataCollection
from MojoSerial.DataFormats.DigiClusterCount import DigiClusterCount
from MojoSerial.DataFormats.TrackCount import TrackCount
from MojoSerial.DataFormats.VertexCount import VertexCount
from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.Framework.EDPutToken import EDPutTokenT
from MojoSerial.MojoBridge.DTypes import Typeable
from MojoSerial.MojoBridge.File import read_simd, read_simd_eof


def readRaw(mut file: FileHandle, nfeds: UInt32) raises -> FEDRawDataCollection:
    var rawCollection = FEDRawDataCollection()
    for _ in range(nfeds):
        var fedId = read_simd[DType.uint32](file)
        var fedSize = read_simd[DType.uint32](file)
        rawCollection.FEDData(Int(fedId))._data = file.read_bytes(Int(fedSize))
    return rawCollection^


struct Source(Defaultable, Movable, Typeable):
    var _startEvent: Int32
    var _endEvent: Int32

    var _runForMinutes: Int32
    var _startTime: Int
    # don't need a mutex
    var _numEventsTimeLastCheck: Int32
    var _shouldStop: Bool

    var _numEvents: Int32
    var _rawToken: EDPutTokenT[FEDRawDataCollection]
    var _digiClusterToken: EDPutTokenT[DigiClusterCount]
    var _trackToken: EDPutTokenT[TrackCount]
    var _vertexToken: EDPutTokenT[VertexCount]
    var _raw: List[FEDRawDataCollection]
    var _digiclusters: List[DigiClusterCount]
    var _tracks: List[TrackCount]
    var _vertices: List[VertexCount]
    var _validation: Bool

    @always_inline
    def __init__(out self):
        self._startEvent = 0
        self._endEvent = 0

        self._runForMinutes = 0
        self._startTime = 0
        self._numEventsTimeLastCheck = 0
        self._shouldStop = False

        self._numEvents = 0
        self._rawToken = EDPutTokenT[FEDRawDataCollection]()
        self._digiClusterToken = EDPutTokenT[DigiClusterCount]()
        self._trackToken = EDPutTokenT[TrackCount]()
        self._vertexToken = EDPutTokenT[VertexCount]()
        self._raw = []
        self._digiclusters = []
        self._tracks = []
        self._vertices = []
        self._validation = False

    def __init__(
        out self,
        var startEvent: Int32,
        var endEvent: Int32,
        var runForMinutes: Int32,
        mut reg: ProductRegistry,
        path: Path,
        var validation: Bool,
    ):
        try:
            self._startEvent = Int32(startEvent)
            self._endEvent = Int32(endEvent)

            self._runForMinutes = runForMinutes
            self._startTime = 0
            self._numEventsTimeLastCheck = 0
            self._shouldStop = False

            self._numEvents = 0
            self._rawToken = reg.produces[FEDRawDataCollection]()
            self._validation = validation
            self._raw = []
            self._digiclusters = []
            self._tracks = []
            self._vertices = []

            if self._validation:
                self._digiClusterToken = reg.produces[DigiClusterCount]()
                self._trackToken = reg.produces[TrackCount]()
                self._vertexToken = reg.produces[VertexCount]()
            else:
                self._digiClusterToken = EDPutTokenT[DigiClusterCount]()
                self._trackToken = EDPutTokenT[TrackCount]()
                self._vertexToken = EDPutTokenT[VertexCount]()

            var in_digiclusters = FileHandle()
            var in_tracks = FileHandle()
            var in_vertices = FileHandle()

            with open(path / "raw.bin", "r") as in_raw:
                if self._validation:
                    in_digiclusters = open(path / "digicluster.bin", "r")
                    in_tracks = open(path / "tracks.bin", "r")
                    in_vertices = open(path / "vertices.bin", "r")

                var nfeds = read_simd[DType.uint32](in_raw)
                while True:
                    self._raw.append(readRaw(in_raw, nfeds))
                    if self._validation:
                        var nm = read_simd[DType.uint32](in_digiclusters)
                        var nd = read_simd[DType.uint32](in_digiclusters)
                        var nc = read_simd[DType.uint32](in_digiclusters)
                        var nt = read_simd[DType.uint32](in_tracks)
                        var nv = read_simd[DType.uint32](in_vertices)
                        self._digiclusters.append(DigiClusterCount(nm, nd, nc))
                        self._tracks.append(TrackCount(nt))
                        self._vertices.append(VertexCount(nv))
                    var eofEvent = read_simd_eof[DType.uint32](in_raw)
                    if eofEvent[0]:
                        break
                    else:
                        nfeds = eofEvent[1]

                if self._validation:
                    in_digiclusters.close()
                    in_tracks.close()
                    in_vertices.close()

            if self._validation:
                debug_assert(
                    self._raw.__len__() == self._digiclusters.__len__()
                )
                debug_assert(self._raw.__len__() == self._tracks.__len__())
                debug_assert(self._raw.__len__() == self._vertices.__len__())

        except e:
            print("Error occurred in bin/Source.mojo,", e)
            return Self()

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self._startEvent = move._startEvent
        self._endEvent = move._endEvent

        self._runForMinutes = move._runForMinutes
        self._startTime = move._startTime
        self._numEventsTimeLastCheck = move._numEventsTimeLastCheck
        self._shouldStop = move._shouldStop

        self._numEvents = move._numEvents
        self._rawToken = move._rawToken
        self._digiClusterToken = move._digiClusterToken
        self._trackToken = move._trackToken
        self._vertexToken = move._vertexToken
        self._raw = move._raw^
        self._digiclusters = move._digiclusters^
        self._tracks = move._tracks^
        self._vertices = move._vertices^
        self._validation = move._validation

    @always_inline
    def reconfigure(
        mut self,
        var startEvent: Int32,
        var endEvent: Int32,
        var runForMinutes: Int32,
    ):
        self._startEvent = startEvent
        self._endEvent = endEvent
        self._runForMinutes = runForMinutes
        self._numEventsTimeLastCheck = 0
        self._shouldStop = False
        self._numEvents = 0

    @always_inline
    def startProcessing(mut self):
        if self._runForMinutes >= 0:
            self._startTime = perf_counter_ns()

    @always_inline
    def processedEvents(self) -> Int32:
        return self._numEvents

    def produce(
        mut self, streamId: Int32, ref reg: ProductRegistry
    ) -> Optional[Event]:
        """C++ returns unique_ptr<Event>; None is its nullptr."""
        if self._shouldStop:
            return None

        var old = self._numEvents
        var globalEvent = self._startEvent + old

        if self._runForMinutes < 0:
            if self._endEvent >= 0 and globalEvent >= self._endEvent:
                self._shouldStop = True
                return None
        else:
            self._numEvents += 1
            if (
                Int(self._numEvents - self._numEventsTimeLastCheck)
                > self._raw.__len__()
            ):
                # this is in nanoseconds
                var processingTime: Int = perf_counter_ns() - self._startTime
                if (processingTime // (6 * 10**10)) >= Int(
                    self._runForMinutes
                ):
                    self._shouldStop = True
                self._numEventsTimeLastCheck = Int32(
                    (Int(self._numEvents) // self._raw.__len__())
                    * self._raw.__len__()
                )
            if self._shouldStop:
                self._numEvents -= 1
                return None
            old = self._numEvents - 1
            globalEvent = self._startEvent + old

        if self._runForMinutes < 0:
            self._numEvents += 1

        var iev = globalEvent + 1
        var ev = Event(streamId, iev, reg)
        var index = Int(globalEvent) % self._raw.__len__()

        ev.put[FEDRawDataCollection](self._rawToken, self._raw[index].copy())
        if self._validation:
            ev.put[DigiClusterCount](
                self._digiClusterToken, self._digiclusters[index]
            )
            ev.put[TrackCount](self._trackToken, self._tracks[index])
            ev.put[VertexCount](self._vertexToken, self._vertices[index])
        return Optional[Event](ev^)

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "Source"
