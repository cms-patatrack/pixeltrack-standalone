from std.sys import size_of
from layout import Layout, LayoutTensor, IntTuple

from MojoSerial.MojoBridge.DTypes import Typeable


def isPowerOf2(v: Int) -> Bool:
    return v and not (v & (v - 1))


# WARNING: THIS STRUCT IS 128-ALIGNED
struct ScalarSoA[T: DType, S: Int](
    Copyable, Defaultable, Movable, Sized, Typeable
):
    comptime Scalar = Scalar[Self.T]
    var _data: InlineArray[Self.Scalar, Self.S]

    @always_inline
    def __init__(out self):
        comptime assert isPowerOf2(Self.S), "SoA stride not a power of 2"
        comptime assert (
            Self.S * size_of[Self.T]() % 128 == 0
        ), "SoA size not a multiple of 128"
        self._data = InlineArray[Self.Scalar, Self.S](fill=0)

    @always_inline
    def __init__(out self, var list: InlineArray[Self.Scalar, Self.S]):
        comptime assert isPowerOf2(Self.S), "SoA stride not a power of 2"
        comptime assert (
            Self.S * size_of[Self.T]() % 128 == 0
        ), "SoA size not a multiple of 128"
        self._data = list^

    # Was UnsafePointer + a `cp` flag; a Span borrow can only copy.
    @always_inline
    def __init__(out self, src: Span[Self.Scalar, _]):
        comptime assert isPowerOf2(Self.S), "SoA stride not a power of 2"
        comptime assert (
            Self.S * size_of[Self.T]() % 128 == 0
        ), "SoA size not a multiple of 128"
        debug_assert(len(src) >= Self.S)

        self._data = InlineArray[Self.Scalar, Self.S](uninitialized=True)

        for i in range(Self.S):
            self._data[i] = src[i]

    @always_inline
    def __len__(self) -> Int:
        return Self.S

    # C++: Scalar* data() / Scalar const* data() -- one span, origin carries const.
    @always_inline
    def data(ref self) -> Span[Self.Scalar, origin_of(self._data)]:
        return Span(self._data)

    @always_inline
    def __getitem__(ref self, i: Int) -> ref [self._data] Self.Scalar:
        return self._data[i]

    @always_inline
    def __setitem__(mut self, i: Int, v: Self.Scalar):
        self._data[i] = v

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "ScalarSoA[" + repr(Self.T) + ", " + String(Self.S) + "]"


# WARNING: THIS STRUCT IS 128-ALIGNED
struct MatrixSoA[T: DType, R: Int, C: Int, S: Int](
    Copyable, Defaultable, Movable, Sized, Typeable
):
    comptime Scalar = Scalar[Self.T]
    comptime _D = InlineArray[Self.Scalar, Self.S * Self.R * Self.C]
    # stride in C++ is coln, row
    comptime Map = Layout(IntTuple(Self.R, Self.C), IntTuple(Self.S, Self.R * Self.S))
    var _data: Self._D

    @always_inline
    def __init__(out self):
        comptime assert isPowerOf2(Self.S), "SoA stride not a power of 2"
        comptime assert (
            Self.R * Self.C * Self.S * size_of[Self.T]() % 128 == 0
        ), "SoA size not a multiple of 128"
        self._data = Self._D(fill=0)

    @always_inline
    def __init__(out self, var list: Self._D):
        comptime assert isPowerOf2(Self.S), "SoA stride not a power of 2"
        comptime assert (
            Self.R * Self.C * Self.S * size_of[Self.T]() % 128 == 0
        ), "SoA size not a multiple of 128"
        self._data = list^

    # Was UnsafePointer + a `cp` flag; a Span borrow can only copy.
    @always_inline
    def __init__(out self, src: Span[Self.Scalar, _]):
        comptime assert isPowerOf2(Self.S), "SoA stride not a power of 2"
        comptime assert (
            Self.R * Self.C * Self.S * size_of[Self.T]() % 128 == 0
        ), "SoA size not a multiple of 128"
        debug_assert(len(src) >= Self.R * Self.C * Self.S)

        self._data = Self._D(uninitialized=True)

        for i in range(Self.R * Self.C * Self.S):
            self._data[i] = src[i]

    @always_inline
    def __len__(self) -> Int:
        return Self.R * Self.C * Self.S

    # C++: Map operator[](int32_t i) { return Map(data_ + i); }
    # Origin must name the field, not `self`, or the Span will not unify.
    @always_inline
    def __getitem__(ref self, i: Int32) -> LayoutTensor[
        mut = origin_of(self._data).mut,
        Self.T,
        Self.Map,
        origin_of(self._data),
    ]:
        return LayoutTensor[
            mut = origin_of(self._data).mut,
            Self.T,
            Self.Map,
            origin_of(self._data),
        ](Span(self._data)[Int(i) :])

    @always_inline
    def __setitem__(mut self, idx: Int32, val: LayoutTensor):
        var dest_slice = self[idx]

        var rows = val.layout.shape[0].value()
        var colns = val.layout.shape[1].value()
        for i in range(rows):
            for j in range(colns):
                dest_slice[i, j] = rebind[Scalar[Self.T]](val[i, j].cast[Self.T]())

    @always_inline
    def __setitem__(
        mut self,
        idx: Int32,
        first: LayoutTensor,
        second: LayoutTensor,
    ):
        """Eigen::Map::operator<<."""
        var dest_slice = self[idx]

        var dest_rows = dest_slice.layout.shape[0].value()
        var dest_cols = dest_slice.layout.shape[1].value()
        var first_rows = first.layout.shape[0].value()
        var first_cols = first.layout.shape[1].value()
        var second_rows = second.layout.shape[0].value()
        var second_cols = second.layout.shape[1].value()

        var i_first = 0
        var j_first = 0
        var i_second = 0
        var j_second = 0

        for j_dest in range(dest_cols):
            for i_dest in range(dest_rows):
                if j_first < first_cols:
                    dest_slice[i_dest, j_dest] = rebind[Scalar[Self.T]](
                        first[i_first, j_first].cast[Self.T]()
                    )

                    i_first += 1
                    if i_first == first_rows:
                        i_first = 0
                        j_first += 1

                elif j_second < second_cols:
                    dest_slice[i_dest, j_dest] = rebind[Scalar[Self.T]](
                        second[i_second, j_second].cast[Self.T]()
                    )

                    i_second += 1
                    if i_second == second_rows:
                        i_second = 0
                        j_second += 1
                else:
                    return

    @always_inline
    @staticmethod
    def dtype() -> String:
        return (
            "MatrixSoA["
            + repr(Self.T)
            + ", "
            + String(Self.R)
            + ", "
            + String(Self.C)
            + ", "
            + String(Self.S)
            + "]"
        )
