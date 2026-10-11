from MojoSerial.MojoBridge.DTypes import Typeable


struct EDPutTokenT[T: Typeable](Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    comptime s_uninitializedValue = 0xFFFFFFFF
    var m_value: UInt

    def __init__(out self):
        self.m_value = Self.s_uninitializedValue

    @always_inline
    def __init__(out self, var iOther: EDPutToken):
        self.m_value = iOther.m_value

    @always_inline
    def __init__[O: Typeable](out self, iValue: UInt):
        comptime assert (
            O.dtype() == "ProductRegistry"
        ), "Only the product registry can hand tokens"
        self.m_value = iValue

    @always_inline
    def index(self) -> UInt:
        return self.m_value

    @always_inline
    def isUninitialized(self) -> Bool:
        return self.m_value == Self.s_uninitializedValue

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "EDPutTokenT[" + Self.T.dtype() + "]"


struct EDPutToken(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    comptime s_uninitializedValue = 0xFFFFFFFF
    var m_value: UInt

    @always_inline
    def __init__(out self):
        self.m_value = Self.s_uninitializedValue

    @always_inline
    def __init__[T: Typeable, //](out self, var iOther: EDPutTokenT[T]):
        self.m_value = iOther.m_value

    @always_inline
    def __init__[O: Typeable](out self, iValue: UInt):
        comptime assert (
            O.dtype() == "ProductRegistry"
        ), "Only the product registry can hand tokens"
        self.m_value = iValue

    @always_inline
    def index(self) -> UInt:
        return self.m_value

    @always_inline
    def isUninitialized(self) -> Bool:
        return self.m_value == Self.s_uninitializedValue

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "EDPutToken"
