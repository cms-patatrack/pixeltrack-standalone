from std.collections.list import _ListIter

from MojoSerial.MojoBridge.DTypes import SizeType, Typeable, UChar


struct FEDRawData(Copyable, Defaultable, Movable, Sized, Typeable):
    """
    Class representing the raw data for one FED.
    The raw data is owned as a binary buffer. It is required that the
    length of the data is a multiple of the S-Link64 word length (8 byte).
    The FED data should include the standard FED header and trailer.
    """

    comptime Data = List[UChar]
    comptime Iterator[origin: Origin] = _ListIter[UChar, origin]
    var _data: Self.Data

    @always_inline
    def __init__(out self):
        self._data = []

    @always_inline
    def __init__(out self, newsize: SizeType):
        debug_assert(
            newsize % 8 == 0,
            "FEDRawData::resize: "
            + String(newsize)
            + " is not a multiple of 8 bytes.",
        )

        self._data = Self.Data(length=Int(newsize), fill=0)

    @always_inline
    def __init__(out self, *, copy: Self):
        self._data = copy._data.copy()

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self._data = move._data^

    @always_inline
    def data(ref self) -> Pointer[UInt8, origin_of(self._data)]:
        return self._data.unsafe_ptr()

    @always_inline
    def size(self) -> SizeType:
        return SizeType(self._data.__len__())

    @always_inline
    def __len__(self) -> Int:
        return self._data.__len__()

    @always_inline
    def resize(mut self, newsize: SizeType):
        debug_assert(
            newsize % 8 == 0,
            "FEDRawData::resize: "
            + String(newsize)
            + " is not a multiple of 8 bytes.",
        )

        if self.size() == newsize:
            return

        self._data.resize(Int(newsize), 0)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "FEDRawData"
