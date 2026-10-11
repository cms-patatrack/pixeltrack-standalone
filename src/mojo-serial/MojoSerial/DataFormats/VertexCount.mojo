from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct VertexCount(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    var _vertices: UInt32

    @always_inline
    def __init__(out self):
        self._vertices = 0

    @always_inline
    def nVertices(self) -> UInt32:
        return self._vertices

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "VertexCount"
