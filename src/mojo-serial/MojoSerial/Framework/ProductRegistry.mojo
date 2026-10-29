from std.collections import Set

from MojoSerial.Framework.EDGetToken import EDGetTokenT
from MojoSerial.Framework.EDPutToken import EDPutTokenT
from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct Indices(Copyable, Movable, Typeable, TrivialRegisterPassable):
    var _moduleIndex: UInt
    var _productIndex: UInt

    @always_inline
    def moduleIndex(self) -> UInt:
        return self._moduleIndex

    @always_inline
    def productIndex(self) -> UInt:
        return self._productIndex

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "Indices"


struct ProductRegistry(Movable, Sized, Typeable):
    comptime kSourceIndex: Int = 0
    var _currentModuleIndex: Int32
    var _consumedModules: Set[UInt]
    var _typeToIndex: Dict[String, Indices]

    @always_inline
    def __init__(out self):
        self._currentModuleIndex = Int32(Self.kSourceIndex)
        self._consumedModules = Set[UInt]()
        self._typeToIndex = Dict[String, Indices]()

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self._currentModuleIndex = move._currentModuleIndex
        self._consumedModules = move._consumedModules^
        self._typeToIndex = move._typeToIndex^

    def produces[T: Typeable](mut self) raises -> EDPutTokenT[T]:
        if T.dtype() in self._typeToIndex:
            raise "RuntimeError: Product of type " + T.dtype() + " already exists."
        var ind = UInt(self.__len__())
        self._typeToIndex[T.dtype()] = Indices(
            UInt(self._currentModuleIndex), ind
        )
        return EDPutTokenT[T].__init__[Self](ind)

    def consumes[T: Typeable](mut self) raises -> EDGetTokenT[T]:
        if T.dtype() not in self._typeToIndex:
            raise "RuntimeError: Product of type " + T.dtype() + " is not produced."
        var item = self._typeToIndex[T.dtype()]
        self._consumedModules.add(item.moduleIndex())
        return EDGetTokenT[T].__init__[Self](item.productIndex())

    # internal interface
    def beginModuleConstruction(mut self, i: Int32):
        self._currentModuleIndex = i
        self._consumedModules.clear()

    def consumedModules(self) -> ref [self._consumedModules] Set[UInt]:
        return self._consumedModules

    def __len__(self) -> Int:
        return self._typeToIndex.__len__()

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "ProductRegistry"
