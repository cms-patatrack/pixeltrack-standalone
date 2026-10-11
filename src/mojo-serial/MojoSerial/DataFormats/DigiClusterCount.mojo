from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct DigiClusterCount(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    var _modules: UInt32
    var _digis: UInt32
    var _clusters: UInt32

    @always_inline
    def __init__(out self):
        self._modules = 0
        self._digis = 0
        self._clusters = 0

    @always_inline
    def nModules(self) -> UInt32:
        return self._modules

    @always_inline
    def nDigis(self) -> UInt32:
        return self._digis

    @always_inline
    def nClusters(self) -> UInt32:
        return self._clusters

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "DigiClusterCount"
