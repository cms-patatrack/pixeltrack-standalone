from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct TrackCount(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    var _tracks: UInt32

    @always_inline
    def __init__(out self):
        self._tracks = 0

    @always_inline
    def nTracks(self) -> UInt32:
        return self._tracks

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "TrackCount"
