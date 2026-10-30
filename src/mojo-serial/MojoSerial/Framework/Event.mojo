from MojoSerial.Framework.EDGetToken import EDGetTokenT
from MojoSerial.Framework.EDPutToken import EDPutTokenT
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.Framework.Wrapper import Wrapper
from MojoSerial.MojoBridge.DTypes import Typeable

comptime StreamID = Int32


# C++: vector<unique_ptr<WrapperBase>>, with the products that have not been put
# yet left null. The types of the products are erased, so that the framework
# does not depend on the plugins that produce them.
struct Event(Defaultable, Movable, Typeable):
    var _streamId: StreamID
    var _eventId: Int32
    var _products: List[Optional[Wrapper]]

    @always_inline
    def __init__(out self):
        self._streamId = 0
        self._eventId = 0
        self._products = []

    @always_inline
    def __init__(
        out self,
        var streamId: Int32,
        var eventId: Int32,
        ref reg: ProductRegistry,
    ):
        self._streamId = streamId
        self._eventId = eventId
        self._products = List[Optional[Wrapper]](capacity=reg.__len__())
        for _ in range(reg.__len__()):
            self._products.append(None)

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self._streamId = move._streamId
        self._eventId = move._eventId
        self._products = move._products^

    @always_inline
    def streamID(self) -> StreamID:
        return self._streamId

    @always_inline
    def eventID(self) -> Int32:
        return self._eventId

    def get[
        T: Typeable & Movable
    ](self, ref token: EDGetTokenT[T]) -> ref [
        self._products[token.index()].value()
    ] T:
        return self._products[token.index()].value().product[T]()

    # emplace is not possible due to failure in binding the constructor at compile time, so we provide put instead

    def put[
        T: Typeable & Movable & Deinitable
    ](mut self, ref token: EDPutTokenT[T], var prod: T):
        self._products[token.index()] = Wrapper(prod^)

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "Event"
