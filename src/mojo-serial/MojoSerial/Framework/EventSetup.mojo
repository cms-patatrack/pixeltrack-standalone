from MojoSerial.Framework.Wrapper import Wrapper
from MojoSerial.MojoBridge.DTypes import Typeable


# C++: map<type_index, unique_ptr<ESWrapperBase>>. The types of the products are
# erased, so that the framework does not depend on the plugins that produce
# them; the type names stand for the type_index.
struct EventSetup(Defaultable, Movable, Typeable):
    var _typeToProduct: Dict[String, Wrapper]

    @always_inline
    def __init__(out self):
        self._typeToProduct = Dict[String, Wrapper]()

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self._typeToProduct = move._typeToProduct^

    def put[
        T: Typeable & Movable & Deinitable
    ](mut self, var prod: T) raises:
        if T.dtype() in self._typeToProduct:
            raise "RuntimeError: Product of type " + T.dtype() + " already exists."
        self._typeToProduct[T.dtype()] = Wrapper(prod^)

    def get[
        T: Typeable & Movable & Deinitable
    ](self) raises -> ref [self._typeToProduct[T.dtype()]] T:
        if T.dtype() not in self._typeToProduct:
            raise "RuntimeError: Product of type " + T.dtype() + " is not produced."
        return self._typeToProduct[T.dtype()].product[T]()

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "EventSetup"
