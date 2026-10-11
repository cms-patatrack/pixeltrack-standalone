from std.sys import align_of, is_gpu
from std.bit import pop_count
from std.math import Ceilable, CeilDivable, Floorable, Truncable, sqrt
from std.utils.numerics import max_finite as _max_finite
from std.utils.numerics import max_or_inf as _max_or_inf
from std.utils.numerics import min_finite as _min_finite
from std.utils.numerics import min_or_neg_inf as _min_or_neg_inf
from std.hashlib.hasher import Hasher
from layout import Layout, LayoutTensor

from MojoSerial.MojoBridge.DTypes import Double, Typeable
from MojoSerial.MojoBridge.Vector import Vector


@fieldwise_init
struct _MatIterator[
    W: DType,
    rows: Int,
    colns: Int,
    mat_origin: Origin,
    forward: Bool = True,
    row_wise: Bool = True,
](Copyable, Iterator, Movable, Typeable):
    comptime mat_type = Matrix[Self.W, Self.rows, Self.colns]
    comptime T = Scalar[Self.W]
    comptime Element = Self.T

    var index: Int
    var src: Pointer[Self.mat_type, Self.mat_origin]

    def __next_ref__(mut self) -> Self.T:
        comptime if Self.forward:
            self.index += 1
            return self.src[][self.index - 1, Self.row_wise]
        else:
            self.index -= 1
            return self.src[][self.index, Self.row_wise]

    @always_inline
    def __next__(mut self) -> Self.T:
        return self.__next_ref__()

    @always_inline
    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    def __iter__(self) -> Self:
        return self.copy()

    def __len__(self) -> Int:
        comptime if Self.forward:
            return len(self.src[]) - self.index
        else:
            return self.index

    @always_inline
    @staticmethod
    def dtype() -> String:
        return (
            "_MatIterator["
            + repr(Self.W)
            + ", "
            + String(Self.rows)
            + ", "
            + String(Self.colns)
            + ", Origin["
            + String(Self.mat_origin.mut)
            + "], "
            + String(Self.row_wise)
            + "]"
        )


# Common interface satisfied by both Matrix (owns its storage) and Map (a
# strided view over external storage), so generic code can accept either.
trait MatrixLike:
    comptime ElemType: DType
    comptime Rows: Int

    def __getitem__(self, i: Int, j: Int) -> Scalar[Self.ElemType]:
        ...

    def __setitem__(mut self, i: Int, j: Int, val: Scalar[Self.ElemType]):
        ...

    def __getitem__(self, i: Int) -> Scalar[Self.ElemType]:
        ...

    def __setitem__(mut self, i: Int, val: Scalar[Self.ElemType]):
        ...

    def num_rows(self) -> Int:
        ...

    def block[
        br: Int, bc: Int
    ](self, row: Int, col: Int) -> Matrix[Self.ElemType, br, bc]:
        ...

    def head[n: Int](self) -> Matrix[Self.ElemType, n, 1]:
        ...

    def col(self, c: Int) -> Vector[Self.ElemType, Self.Rows]:
        ...

    @staticmethod
    def ColsAtCompileTime() -> Int:
        ...


# A comment about this implementation: it is probably the speediest, but arguably not of the best memory efficiency (?)
# Handling rows in a SIMD structure does give rows immense advantage over columns, it also simplfies implementation... but we are still using InlineArray for memory
# TODO: Is implementing a matrix as an inline array of vectors faster or slower than a direct memory implementation using an unsafe pointer?
struct Matrix[T: DType, rows: Int, colns: Int](
    Absable,
    CeilDivable,
    Ceilable,
    Copyable,
    Defaultable,
    Floorable,
    Hashable,
    ImplicitlyCopyable,
    MatrixLike,
    Movable,
    Roundable,
    Sized,
    Truncable,
    Typeable,
    Writable,
):
    comptime ElemType = Self.T
    comptime Rows = Self.rows
    comptime _L = List[List[Scalar[Self.T]]]
    comptime _LS = InlineArray[InlineArray[Scalar[Self.T], Self.colns], Self.rows]
    comptime _R = Vector[Self.T, Self.colns]
    comptime _D = Scalar[Self.T]
    comptime _DC = InlineArray[Vector[Self.T, Self.colns], Self.rows]
    comptime _DB = InlineArray[Vector[DType.bool, Self.colns], Self.rows]
    comptime _Mask = Matrix[DType.bool, Self.rows, Self.colns]
    var _data: Self._DC

    comptime MAX = Self(_max_or_inf[Self.T]())
    comptime MIN = Self(_min_or_neg_inf[Self.T]())
    comptime MAX_FINITE = Self(_max_finite[Self.T]())
    comptime MIN_FINITE = Self(_min_finite[Self.T]())

    comptime _default_alignment = align_of[Self._D]() if is_gpu() else 1

    # Lifecycle methods
    @always_inline
    def __init__(out self):
        """Default constructor."""
        self._data = Self._DC(fill=Self._R())

    @always_inline
    def __init__(out self, *, uninitialized: Bool):
        """Default unsafe constructor."""
        self._data = Self._DC(uninitialized=uninitialized)

    # InlineArray is not ImplicitlyCopyable, so this cannot be synthesized.
    @always_inline
    def __init__(out self, *, copy: Self):
        self._data = Self._DC(uninitialized=True)

        comptime for i in range(Self.rows):
            self._data[i] = copy._data[i]

    @always_inline
    def copy(self) -> Self:
        """Explicitly construct a copy of self."""
        return Self(copy=self)

    @always_inline
    def __init__[U: DType, //](out self, *, var row: Vector[U, Self.colns]):
        """Initialize a matrix from a Vector row object of the same coln-size, splattered across all rows.
        """
        self._data = Self._DC(fill=Self._R(row))

    @always_inline
    def __init__[U: DType, //](out self, *, var coln: Vector[U, Self.colns]):
        """Initialize a matrix from a Vector coln object of the same row-size, splattered across all columns.
        """
        self._data = Self._DC(uninitialized=True)

        comptime for i in range(Self.rows):

            comptime for j in range(Self.colns):
                self[i, j] = coln[i].cast[Self.T]()

    @always_inline
    def __init__[U: DType, //](out self, val: Scalar[U], /):
        """Initializes a matrix with a scalar.
        The scalar is splatted across all the elements of the matrix."""
        self._data = Self._DC(fill=Self._R(val))

    @always_inline
    def __init__(out self, val: Int, /):
        """Initializes a matrix with a signed integer.
        The signed integer is splatted across all the elements of the matrix."""
        self._data = Self._DC(fill=Self._R(val))

    @always_inline
    def __init__(out self, val: UInt, /):
        """Initializes a matrix with a unsigned integer.
        The unsigned integer is splatted across all the elements of the matrix.
        """
        self._data = Self._DC(fill=Self._R(val))

    @always_inline
    @implicit
    def __init__(out self, val: IntLiteral, /):
        """Initializes a matrix with an integer literal (implicit).
        The integer literal is splatted across all the elements of the matrix.
        """
        self._data = Self._DC(fill=Self._R(val))

    @always_inline
    @implicit
    def __init__(out self, *values: Self._D, __list_literal__: () = ()):
        """Constructs a matrix via a variadic list of values in a literal format (implicit).
        """
        self._data = Self._DC(uninitialized=True)
        for i in range(values.__len__()):
            self[i] = values[i]

    @implicit
    def __init__(out self, mat: Self._L):
        """Constructs a matrix via a matrix list representation (implicit)."""
        self._data = Self._DC(uninitialized=True)
        for i in range(min(Self.rows, mat.__len__())):
            for j in range(min(Self.colns, mat[0].__len__())):
                self[i, j] = mat[i][j]

    @implicit
    def __init__(out self, mat: Self._LS):
        """Constructs a matrix via a matrix inline array representation (implicit).
        """
        self._data = Self._DC(uninitialized=True)

        comptime for i in range(Self.rows):

            comptime for j in range(Self.colns):
                self[i, j] = mat[i][j]

    @implicit
    def __init__(out self, var data: Self._DC):
        """Constructs a matrix via a matrix inline array internal data object (implicit).
        """
        self._data = data^

    def __init__[
        vrows: Int, vcolns: Int, //
    ](out self, mat: Matrix[Self.T, vrows, vcolns]):
        """Initialize a matrix from an arbitrary matrix. Might cause data loss.
        """
        self._data = Self._DC(uninitialized=True)

        comptime for i in range(min(Self.rows * Self.colns, vrows * vcolns)):
            self[i] = mat[i]

    def __init__[U: DType, //](out self, mat: Matrix[U, Self.rows, Self.colns]):
        """Initialize a matrix from a matrix of the same size of a different data type.
        """
        self._data = Self._DC(uninitialized=True)

        comptime for i in range(Self.rows * Self.colns):
            self[i] = mat[i].cast[Self.T]()

    def __init__[
        *, row_offset: Int, coln_offset: Int
    ](out self, mat: Matrix[Self.T, ...]):
        """Initializes a matrix as a slice of another matrix with specified output size and offset.
        """
        self._data = Self._DC(uninitialized=True)
        var u = 0

        comptime for i in range(row_offset, Self.rows + row_offset):
            var v = 0

            comptime for j in range(coln_offset, Self.colns + coln_offset):
                self[u, v] = mat[i, j]
                v += 1
            u += 1

    # Compatibility with V1 Matrices

    def __init__[vsize: Int, //](out self, vec: Vector[Self.T, vsize]):
        """Initialize a matrix from an arbitrary vector (V1 format). Might cause data loss.
        """
        self._data = Self._DC(uninitialized=True)

        comptime for i in range(min(Self.rows * Self.colns, vsize)):
            self[i] = vec[i]

    @implicit
    def __init__(out self, values: List[Self._D], /):
        """Initialize a matrix from a list of values. Might cause data loss."""
        self._data = Self._DC(uninitialized=True)
        for i in range(min(self.__len__(), values.__len__())):
            self[i] = values[i]

    @always_inline
    def __getitem__(self, i: Int) -> Self._D:
        return self._data[i // Self.colns][i % Self.colns]

    @always_inline
    def __getitem__(self, i: Int, row_wise: Bool) -> Self._D:
        if row_wise:
            return self._data[i // Self.colns][i % Self.colns]
        else:
            return self._data[i % Self.rows][i // Self.rows]

    @always_inline
    def __setitem__(mut self, i: Int, val: Self._D):
        self._data[i // Self.colns][i % Self.colns] = val

    @always_inline
    def __setitem__(mut self, i: Int, row_wise: Bool, val: Self._D):
        if row_wise:
            self._data[i // Self.colns][i % Self.colns] = val
        else:
            self._data[i % Self.rows][i // Self.rows] = val

    @always_inline
    def __len__(self) -> Int:
        return Self.rows * Self.colns

    @always_inline
    @staticmethod
    def Zero() -> Self:
        return Self()

    @always_inline
    @staticmethod
    def Constant(val: Self._D) -> Self:
        var res = Self(uninitialized=True)
        comptime for i in range(Self.rows * Self.colns):
            res[i] = val
        return res

    @staticmethod
    def RowsAtCompileTime() -> Int:
        return Self.rows

    @staticmethod
    def ColsAtCompileTime() -> Int:
        return Self.colns

    @always_inline
    def num_rows(self) -> Int:
        return Self.rows

    @always_inline
    def cols(self) -> Int:
        return Self.colns

    # Operators

    @always_inline
    def __getitem__(self, i: Int, j: Int) -> Self._D:
        return self._data[i][j]

    @always_inline
    def __setitem__(mut self, i: Int, j: Int, val: Self._D):
        self._data[i][j] = val

    def __iter__(
        ref self,
    ) -> _MatIterator[Self.T, Self.rows, Self.colns, origin_of(self)]:
        return _MatIterator[Self.T, Self.rows, Self.colns, origin_of(self)](
            0, Pointer(to=self)
        )

    @always_inline
    def __contains__(self, value: Self._D) -> Bool:
        var res = False

        comptime for i in range(Self.rows):
            res = res and self._data[i].__contains__(value)
            if res:
                return res
        return res

    @always_inline
    def __add__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        var res = self

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i] + rhs._data[i]
        return res

    @always_inline
    def __sub__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        var res = self

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i] - rhs._data[i]
        return res

    @always_inline
    def __mul__[trp: Int, //](self, rhs: Matrix[Self.T, Self.colns, trp]) -> Matrix[Self.T, Self.rows, trp]:
        """Matrix product (rows x colns) @ (colns x trp), matching Eigen's operator*."""
        return self @ rhs

    @always_inline
    def __mul__(self, scalar: Self._D) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        var res = self
        # Splat explicitly: relying on implicit Scalar->Vector conversion in
        # `res._data[i] * scalar` silently only fills lane 0 in this Mojo build.
        var splatted = Self._R(scalar)

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i] * splatted
        return res

    @always_inline
    def __rmul__(self, scalar: Self._D) -> Self:
        return self * scalar

    def __matmul__[
        trp: Int, //
    ](self, rhs: Matrix[Self.T, Self.colns, trp]) -> Matrix[Self.T, Self.rows, trp]:
        var res = Matrix[Self.T, Self.rows, trp](uninitialized=True)

        comptime for i in range(Self.rows):

            comptime for j in range(trp):
                res[i, j] = self._row_by_coln(rhs, i, j)
        return res

    @always_inline
    def __truediv__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        var res = self

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i] / rhs._data[i]
        return res

    @always_inline
    def __truediv__(self, scalar: Self._D) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        var res = self
        var splatted = Self._R(scalar)

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i] / splatted
        return res

    @always_inline
    def __floordiv__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        var res = self

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i] // rhs._data[i]
        return res

    @always_inline
    def __mod__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        var res = self

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i] % rhs._data[i]
        return res

    def __pow__(
        self: Matrix[Self.T, Self.rows, Self.colns], exp: Int
    ) -> Matrix[Self.T, Self.rows, Self.colns]:
        comptime assert (
            Self.rows == Self.colns
        ), "Can only calculate power of a square matrix"
        comptime sq = Matrix[Self.T, Self.rows, Self.rows]

        if exp < 0:
            return ~(self**-exp)
        elif exp == 0:
            return Matrix[Self.T, Self.rows, Self.colns].identity()
        elif exp == 1:
            return self
        var res: Matrix[Self.T, Self.rows, Self.colns] = self
        for _ in range(2, exp + 1):
            res = rebind[sq](self) @ res
        return res

    @always_inline
    def __lt__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i] < rhs._data[i]
        return res

    @always_inline
    def __le__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i] <= rhs._data[i]
        return res

    @always_inline
    def __eq__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i] == rhs._data[i]
        return res

    @always_inline
    def __ne__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i] != rhs._data[i]
        return res

    @always_inline
    def __gt__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i] > rhs._data[i]
        return res

    @always_inline
    def __ge__(self, rhs: Self) -> Self._Mask:
        var res = Self._Mask()

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i] >= rhs._data[i]
        return res

    @always_inline
    def __pos__(self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return self

    @always_inline
    def __neg__(self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        var res = self

        comptime for i in range(Self.rows):
            res[i] = -res[i]
        return res

    @always_inline
    def __and__(self, rhs: Self) -> Self:
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType must be an integral or bool type"
        var res = self

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i] & rhs._data[i]
        return res

    @always_inline
    def __xor__(self, rhs: Self) -> Self:
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType must be an integral or bool type"
        var res = self

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i] ^ rhs._data[i]
        return res

    @always_inline
    def __or__(self, rhs: Self) -> Self:
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType must be an integral or bool type"
        var res = self

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i] | rhs._data[i]
        return res

    @always_inline
    def __lshift__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_integral(), "DType must be an integral type"
        var res = self

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i] << rhs._data[i]
        return res

    @always_inline
    def __rshift__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_integral(), "DType must be an integral type"
        var res = self

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i] >> rhs._data[i]
        return res

    def __invert__[
        W: DType, *, protect: Bool = False
    ](self: Matrix[Self.T, Self.rows, Self.colns]) -> Matrix[W, Self.rows, Self.colns]:
        comptime assert (
            Self.rows == Self.colns
        ), "Can only find inverse of a square matrix"
        debug_assert(
            abs(self.det[DType.float64]()) > 1e-9, "Matrix is not invertible"
        )
        # if this assert fails, we'll return a weird value
        comptime n = Self.rows

        var mat = self.cast[DType.float64]()
        var idn = Matrix[DType.float64, Self.rows, Self.colns].identity()

        comptime for i in range(n):
            var pivot = i

            comptime for j in range(i, n):
                if abs(mat[j, i]) > abs(mat[pivot, i]):
                    pivot = j

            mat._data[i], mat._data[pivot] = mat._data[pivot], mat._data[i]
            idn._data[i], idn._data[pivot] = idn._data[pivot], idn._data[i]

            if abs(mat[i, i]) < 1e-9:
                return idn.cast[W]()

            var div = mat[i, i]

            comptime for j in range(n):
                mat[i, j] /= div
                idn[i, j] /= div

            comptime for j in range(n):
                if i != j:
                    var mult = mat[j, i]

                    comptime for k in range(n):
                        mat[j, k] -= mult * mat[i, k]
                        idn[j, k] -= mult * idn[i, k]

        comptime if protect:

            comptime if W in (
                DType.uint8,
                DType.uint16,
                DType.uint32,
                DType.uint64,
                DType.uint128,
                DType.uint256,
            ):

                comptime for i in range(n * n):
                    if idn[i] < 1e-9:
                        idn[i] = 0
        return idn.cast[W]()

    @always_inline
    def __invert__(self) -> Self:
        return self.__invert__[Self.T]()

    # In place operations

    @always_inline("nodebug")
    def __iadd__(mut self, rhs: Self):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        self = self + rhs

    @always_inline("nodebug")
    def __isub__(mut self, rhs: Self):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        self = self - rhs

    @always_inline("nodebug")
    def __imul__(mut self, rhs: Self):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        comptime assert (
            Self.rows == Self.colns
        ), "In-place matrix multiply requires a square matrix"
        var result = rebind[Matrix[Self.T, Self.colns, Self.colns]](self) @ rebind[Matrix[Self.T, Self.colns, Self.colns]](rhs)
        self = rebind[Self](result)

    @always_inline("nodebug")
    def __imul__(mut self, scalar: Self._D):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        self = self * scalar

    @always_inline("nodebug")
    def __itruediv__(mut self, rhs: Self):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        self = self / rhs

    @always_inline("nodebug")
    def __itruediv__(mut self, scalar: Self._D):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        self = self / scalar

    @always_inline("nodebug")
    def __ifloordiv__(mut self, rhs: Self):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        self = self // rhs

    @always_inline("nodebug")
    def __imod__(mut self, rhs: Self):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        self = self.__mod__(rhs)

    @always_inline("nodebug")
    def __ipow__(mut self: Matrix[Self.T, Self.rows, Self.colns], rhs: Int):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        comptime assert (
            Self.rows == Self.colns
        ), "Can only calculate power of a square matrix"
        self = self.__pow__(rhs)

    @always_inline("nodebug")
    def __iand__(mut self, rhs: Self):
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType must be an integral or bool type"
        self = self & rhs

    @always_inline("nodebug")
    def __ixor__(mut self, rhs: Self):
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType must be an integral or bool type"
        self = self ^ rhs

    @always_inline("nodebug")
    def __ior__(mut self, rhs: Self):
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType must be an integral or bool type"
        self = self | rhs

    @always_inline("nodebug")
    def __ilshift__(mut self, rhs: Self):
        comptime assert Self.T.is_integral(), "DType must be an integral type"
        self = self << rhs

    @always_inline("nodebug")
    def __irshift__(mut self, rhs: Self):
        comptime assert Self.T.is_integral(), "DType must be an integral type"
        self = self >> rhs

    @always_inline("nodebug")
    def __iinvert__(mut self):
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType must be an integral or bool type"
        self = ~self

    # Reversed operations

    @always_inline
    def __radd__(self, value: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return value + self

    @always_inline
    def __rsub__(self, value: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return value - self

    @always_inline
    def __rfloordiv__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return rhs // self

    @always_inline
    def __rtruediv__(self, value: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return value / self

    @always_inline
    def __rmod__(self, value: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return value % self

    @always_inline
    def __rand__(self, value: Self) -> Self:
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType be an integral or bool type"
        return value & self

    @always_inline
    def __rxor__(self, value: Self) -> Self:
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType be an integral or bool type"
        return value ^ self

    @always_inline
    def __ror__(self, value: Self) -> Self:
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType be an integral or bool type"
        return value | self

    @always_inline
    def __rlshift__(self, value: Self) -> Self:
        comptime assert Self.T.is_integral(), "DType be an integral type"
        return value << self

    @always_inline
    def __rrshift__(self, value: Self) -> Self:
        comptime assert Self.T.is_integral(), "DType be an integral type"
        return value >> self

    # Trait conformance

    @always_inline
    @staticmethod
    def dtype() -> String:
        return (
            "Matrix["
            + repr(Self.T)
            + ", "
            + String(Self.rows)
            + ", "
            + String(Self.colns)
            + "]"
        )

    @always_inline
    def __str__(self) -> String:
        return String(self)

    def __repr__(self) -> String:
        var output = String()
        output.write(
            "Matrix[" + repr(Self.T) + ", ", Self.rows, ", ", Self.colns, "]("
        )
        for i in range(self.__len__()):
            output.write(self[i])
            if i < self.__len__() - 1:
                output.write(", ")
        output.write(")")
        return output^

    @always_inline
    def __floor__(self) -> Self:
        var res = Self(uninitialized=True)

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i].__floor__()
        return res

    @always_inline
    def __ceil__(self) -> Self:
        var res = Self(uninitialized=True)

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i].__ceil__()
        return res

    @always_inline
    def __trunc__(self) -> Self:
        var res = Self(uninitialized=True)

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i].__trunc__()
        return res

    @always_inline
    def __abs__(self) -> Self:
        var res = Self(uninitialized=True)

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i].__abs__()
        return res

    @always_inline
    def __round__(self) -> Self:
        var res = Self(uninitialized=True)

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i].__round__()
        return res

    @always_inline
    def __round__(self, ndigits: Int) -> Self:
        var res = Self(uninitialized=True)

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i].__round__(ndigits)
        return res

    @always_inline
    def __ceildiv__(self, denominator: Self) -> Self:
        return self.__truediv__(denominator).__round__()

    def __hash__[H: Hasher](self, mut hasher: H):
        comptime for i in range(Self.rows):
            self._data[i].__hash__[H](hasher)
        hasher._update_with_simd(Scalar[DType.uint64](37))

    # Methods

    @always_inline("nodebug")
    def _refine[
        _T: DType = Self.T, _rows: Int = Self.rows, _colns: Int = Self.colns
    ](self) -> Matrix[_T, _rows, _colns]:
        return rebind[Matrix[_T, _rows, _colns]](self)

    @always_inline
    def cast[target: DType](self) -> Matrix[target, Self.rows, Self.colns]:
        comptime if Self.T == target:
            return self._refine[target]()

        comptime if Self.T in (DType.float8_e4m3fn, DType.float8_e5m2):
            comptime assert (
                target
                in (
                    DType.bfloat16,
                    DType.float16,
                    DType.float32,
                    DType.float64,
                )
            ), String(
                (
                    "Only FP8->F64, FP8->F32, FP8->F16, and FP8->BF16"
                    " castings are implemented. "
                ),
                Self.T,
                "->",
                target,
            )

        # low level manip for efficiency
        var res = InlineArray[Vector[target, Self.colns], Self.rows](
            uninitialized=True
        )

        comptime for i in range(Self.rows):
            res[i] = self._data[i].cast[target]()
        return Matrix[target, Self.rows, Self.colns](res^)

    @always_inline
    def is_power_of_two(self) -> Self._Mask:
        comptime assert Self.T.is_integral(), "DType must be integral"
        if Self.T.is_unsigned():
            return self.pop_count() == 1
        else:
            return (self > 0) & (self & (self - 1) == 0)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("[")

        var width = 0

        comptime for i in range(Self.rows * Self.colns):
            width = max(width, String(self[i]).byte_length())

        comptime for i in range(Self.rows):
            if i != 0:
                writer.write(" ")
            writer.write("[")

            comptime for j in range(Self.colns):
                var _c = width - String(self[i, j]).byte_length()
                writer.write(
                    " " * (_c if _c > 0 else 0)
                    + String(self[i, j])
                    + (" " if j < Self.colns - 1 else "")
                )
            writer.write("]")
            if i < Self.rows - 1:
                writer.write("\n")
        writer.write("]")

    def row_iterator(
        ref self,
    ) -> _MatIterator[Self.T, Self.rows, Self.colns, origin_of(self)]:
        return _MatIterator[Self.T, Self.rows, Self.colns, origin_of(self)](
            0, Pointer(to=self)
        )

    def coln_iterator(
        ref self,
    ) -> _MatIterator[
        Self.T, Self.rows, Self.colns, origin_of(self), row_wise=False
    ]:
        return _MatIterator[
            Self.T, Self.rows, Self.colns, origin_of(self), row_wise=False
        ](0, Pointer(to=self))

    @always_inline
    def row(self, i: Int) -> Vector[Self.T, Self.colns]:
        return self._data[i]

    def coln(self, j: Int) -> Vector[Self.T, Self.rows]:
        var res = Vector[Self.T, Self.rows]()

        comptime for i in range(Self.rows):
            res[i] = self[i, j]
        return res

    @always_inline
    def col(self, c: Int) -> Vector[Self.T, Self.rows]:
        return self.coln(c)

    @always_inline
    def block[br: Int, bc: Int](self, row: Int, col: Int) -> Matrix[Self.T, br, bc]:
        var res = Matrix[Self.T, br, bc](uninitialized=True)
        comptime for r in range(br):
            comptime for c in range(bc):
                res[r, c] = self[row + r, col + c]
        return res

    @always_inline
    def set_block[
        br: Int, bc: Int
    ](mut self, row: Int, col: Int, val: Matrix[Self.T, br, bc]):
        """Writes val into self at (row, col), the write-back counterpart to
        block() (which returns a copy, unlike Eigen's reference-semantics
        .block())."""
        comptime for r in range(br):
            comptime for c in range(bc):
                self[row + r, col + c] = val[r, c]

    #this is not flattening it is just taking first col
    @always_inline
    def head[n: Int](self) -> Matrix[Self.T, n, 1]:
        comptime assert Self.colns == 1, "head() requires a column vector"
        var res = Matrix[Self.T, n, 1](uninitialized=True)
        comptime for i in range(n):
            res[i, 0] = self[i, 0]
        return res

    @always_inline
    def squaredNorm(self) -> Self._D:
        var acc: Self._D = 0
        comptime for i in range(Self.rows):
            comptime for j in range(Self.colns):
                acc += self[i, j] * self[i, j]
        return acc

    @always_inline
    def norm(self) -> Self._D:
        return sqrt(self.squaredNorm())

    @always_inline
    def dot(self, other: Self) -> Self._D:
        var acc: Self._D = 0
        comptime for i in range(Self.rows):
            comptime for j in range(Self.colns):
                acc += self[i, j] * other[i, j]
        return acc

    def _row_by_coln(
        self, other: Matrix[Self.T, Self.colns, _], row: Int, coln: Int
    ) -> Self._D:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        var sum: Self._D = 0

        comptime for i in range(Self.colns):
            sum += self[row, i] * other[i, coln]
        return sum

    def transpose(self) -> Matrix[Self.T, Self.colns, Self.rows]:
        var res = Matrix[Self.T, Self.colns, Self.rows](uninitialized=True)

        comptime for i in range(Self.colns):

            comptime for j in range(Self.rows):
                res[i, j] = self[j, i]
        return res

    @staticmethod
    def identity() -> Self:
        comptime assert Self.rows == Self.colns, "Identity can only be a square matrix"
        var res = Self()
        for i in range(Self.rows):
            res[i, i] = 1
        return res

    @always_inline
    def inverse[
        W: DType, *, protect: Bool = False
    ](self: Matrix[Self.T, Self.rows, Self.colns]) -> Matrix[W, Self.rows, Self.colns]:
        comptime assert (
            Self.rows == Self.colns
        ), "Can only find inverse of a square matrix"
        return self.__invert__[W, protect=protect]()

    @always_inline
    def inverse(self) -> Self:
        comptime assert (
            Self.rows == Self.colns
        ), "Can only find inverse of a square matrix"
        return ~self

    def det[
        W: DType, *, protect: Bool = False
    ](self: Matrix[Self.T, Self.rows, Self.colns]) -> Scalar[W]:
        comptime assert (
            Self.rows == Self.colns
        ), "Can only calculate determinant for a square matrix"
        comptime n = Self.rows

        var mat = self.cast[DType.float64]()
        var det: Double = 1.0

        comptime for i in range(n):
            var pivot = i

            comptime for j in range(i + 1, n):
                if abs(mat[j, i]) > abs(mat[pivot, i]):
                    pivot = j
            if pivot != i:
                mat._data[i], mat._data[pivot] = mat._data[pivot], mat._data[i]
                det *= -1
            if mat[i, i] == 0:
                return 0
            det *= mat[i, i]

            comptime for j in range(i + 1, n):
                var factor: Double = mat[j, i] / mat[i, i]

                comptime for k in range(i + 1, n):
                    mat[j, k] -= factor * mat[i, k]

        comptime if protect:

            comptime if W in (
                DType.uint8,
                DType.uint16,
                DType.uint32,
                DType.uint64,
                DType.uint128,
                DType.uint256,
            ):
                if det < 1e-9:
                    det = 0
        return det.cast[W]()

    @always_inline
    def det[
        *, protect: Bool = False
    ](self: Matrix[Self.T, Self.rows, Self.colns],) -> Self._D:
        return self.det[Self.T, protect=protect]()

    @always_inline
    def clamp(self, lower_bound: Self, upper_bound: Self) -> Self:
        var res = self

        comptime for i in range(Self.rows):
            res._data[i] = res._data[i].clamp(
                lower_bound._data[i], upper_bound._data[i]
            )
        return res

    @always_inline
    def fma(self, multiplier: Self, accumulator: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        var res = self
        for i in range(Self.rows):
            res._data[i] = res._data[i].fma(
                multiplier._data[i], accumulator._data[i]
            )
        return res

    def slice[
        output_rows: Int,
        output_colns: Int,
        /,
        *,
        row_offset: Int = 0,
        coln_offset: Int = 0,
    ](self) -> Matrix[Self.T, output_rows, output_colns]:
        comptime assert (
            0 <= row_offset < output_rows + row_offset <= Self.rows
        ), "Output rows must be a positive integer less than rows"
        comptime assert (
            0 <= coln_offset < output_colns + coln_offset <= Self.rows
        ), "Output colns must be a positive integer less than colns"

        comptime if output_rows == 1 and output_colns == 1:
            return self[row_offset, coln_offset]

        return Matrix[Self.T, output_rows, output_colns].__init__[
            row_offset=row_offset, coln_offset=coln_offset
        ](self)

    def insert[
        *, row_offset: Int = 0, coln_offset: Int = 0
    ](self, mat: Matrix[Self.T, ...]) -> Self:
        comptime input_rows = mat.rows
        comptime input_colns = mat.colns
        comptime assert (
            0 <= row_offset < input_rows + row_offset <= Self.rows
        ), "Insertion position must not exceed the rows of the matrix"
        comptime assert (
            0 <= coln_offset < input_colns + coln_offset <= Self.rows
        ), "Insertion position must not exceed the colns of the matrix"

        comptime if Self.rows == 1 and Self.colns == 1:
            comptime assert (
                input_rows == 1 and input_colns == 1
            ), "The input width must be 1 if the size of the matrix is 1"
            return mat[0]

        var res = self

        comptime for i in range(row_offset, Self.rows):
            res._data[i] = res._data[i].insert[offset=coln_offset](mat._data[i])
        return res

    def iinsert[
        *, row_offset: Int = 0, coln_offset: Int = 0
    ](mut self, mat: Matrix[Self.T, ...]):
        comptime input_rows = mat.rows
        comptime input_colns = mat.colns
        comptime assert (
            0 <= row_offset < input_rows + row_offset <= Self.rows
        ), "Insertion position must not exceed the rows of the matrix"
        comptime assert (
            0 <= coln_offset < input_colns + coln_offset <= Self.rows
        ), "Insertion position must not exceed the colns of the matrix"

        comptime if Self.rows == 1 and Self.colns == 1:
            comptime assert (
                input_rows == 1 and input_colns == 1
            ), "The input width must be 1 if the size of the matrix is 1"
            self[0] = mat[0]

        comptime for i in range(row_offset, Self.rows):
            self._data[i] = self._data[i].insert[offset=coln_offset](
                mat._data[i]
            )

    def row_stack[
        mcolns: Int, //
    ](self, other: Matrix[Self.T, Self.rows, mcolns]) -> Matrix[Self.T, Self.rows, Self.colns + mcolns]:
        var res = Matrix[Self.T, Self.rows, Self.colns + mcolns](uninitialized=True)

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i].join(other._data[i])
        return res

    def coln_stack[
        mrows: Int, //
    ](self, other: Matrix[Self.T, mrows, Self.colns]) -> Matrix[Self.T, Self.rows + mrows, Self.colns]:
        var res = Matrix[Self.T, Self.rows + mrows, Self.colns](uninitialized=True)

        comptime for i in range(Self.rows):
            res._data[i] = self._data[i]

        comptime for i in range(mrows):
            res._data[Self.rows + i] = other._data[i]
        return res

    def split[
        factor: Int = 2
    ](self) -> InlineArray[
        Matrix[Self.T, Self.rows // factor, Self.colns // factor], factor * factor
    ]:
        comptime assert (
            Self.rows == Self.colns and Self.rows % factor == 0
        ), "Can only do integral splits on square matrices"
        var res = InlineArray[
            Matrix[Self.T, Self.rows // factor, Self.colns // factor], factor * factor
        ](uninitialized=True)
        var i = 0

        comptime for row_offset in range(0, Self.rows, Self.rows // factor):

            comptime for coln_offset in range(0, Self.colns, Self.colns // factor):
                res[i] = self.slice[
                    Self.rows // factor,
                    Self.colns // factor,
                    row_offset=row_offset,
                    coln_offset=coln_offset,
                ]()
                i += 1
        return res^

    # Reductions

    def reduce_max(self) -> Self._D:
        comptime if Self.rows == 1 and Self.colns == 1:
            return self[0]
        var A = self._data[0].reduce_max()

        comptime for i in range(Self.rows):
            A = max(A, self._data[i].reduce_max())
        return A

    def reduce_min(self) -> Self._D:
        comptime if Self.rows == 1 and Self.colns == 1:
            return self[0]
        var A = self._data[0].reduce_min()

        comptime for i in range(Self.rows):
            A = min(A, self._data[i].reduce_min())
        return A

    def reduce_add(self) -> Self._D:
        comptime if Self.rows == 1 and Self.colns == 1:
            return self[0]
        var A = self._data[0].reduce_add()

        comptime for i in range(1, Self.rows):
            A = A + self._data[i].reduce_add()
        return A

    def reduce_mul(self) -> Self._D:
        comptime if Self.rows == 1 and Self.colns == 1:
            return self[0]
        var A = self._data[0].reduce_mul()

        comptime for i in range(1, Self.rows):
            A = A * self._data[i].reduce_mul()
        return A

    def reduce_and(self) -> Self._D:
        comptime if Self.rows == 1 and Self.colns == 1:
            return self[0]
        var A = self._data[0].reduce_and()

        comptime for i in range(Self.rows):
            A = A & self._data[i].reduce_and()
        return A

    def reduce_or(self) -> Self._D:
        comptime if Self.rows == 1 and Self.colns == 1:
            return self[0]
        var A = self._data[0].reduce_or()

        comptime for i in range(Self.rows):
            A = A | self._data[i].reduce_or()
        return A

    def reduce_bit_count(self) -> Int:
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "Expected either integral or bool type"

        comptime if Self.T == DType.bool:
            return Int(self.cast[DType.uint8]().reduce_add())
        else:
            return Int(self.pop_count().reduce_add())

    def pop_count(self) -> Self:
        var res = Self._DC(uninitialized=True)

        comptime for i in range(Self.rows):
            res[i] = Vector[Self.T, Self.colns](pop_count(self._data[i]._data))
        return Self(res^)


# A strided view over storage owned elsewhere. The origin is a real parameter,
# so the borrow checker tracks the backing buffer's lifetime; C++ never makes a
# Map-backed array const, so a mutable origin is the only kind needed. (A single
# Map generic over `mut` does not work: 1.0 rejects writes through
# `Span[T, Origin[mut=mut]]` even under `where Self.mut`.)
struct Map[
    T: DType,
    rows: Int,
    colns: Int,
    origin: MutOrigin,
    default_inner_stride: Int = 1,
](
    Copyable,
    MatrixLike,
    Movable,
    Typeable,
):
    comptime ElemType = Self.T
    comptime Rows = Self.rows
    var data: Span[Scalar[Self.T], Self.origin]
    var inner_stride: Int
    var outer_stride: Int

    @always_inline
    def __init__(out self, var data: Span[Scalar[Self.T], Self.origin]):
        self.data = data
        self.inner_stride = Self.default_inner_stride
        self.outer_stride = Self.rows * Self.default_inner_stride

    @always_inline
    def __init__(
        out self,
        var data: Span[Scalar[Self.T], Self.origin],
        inner_stride: Int,
    ):
        self.data = data
        self.inner_stride = inner_stride
        self.outer_stride = Self.rows * inner_stride

    @always_inline
    def __init__(
        out self,
        var data: Span[Scalar[Self.T], Self.origin],
        inner_stride: Int,
        outer_stride: Int,
    ):
        self.data = data
        self.inner_stride = inner_stride
        self.outer_stride = outer_stride

    @always_inline
    def __getitem__(self, r: Int, c: Int) -> Scalar[Self.T]:
        return self.data[c * self.outer_stride + r * self.inner_stride]

    @always_inline
    def __setitem__(mut self, r: Int, c: Int, val: Scalar[Self.T]):
        self.data[c * self.outer_stride + r * self.inner_stride] = val
    #this is not flattening it is just taking first col
    @always_inline
    def __getitem__(self, i: Int) -> Scalar[Self.T]:
        comptime assert Self.colns == 1, "Map 1D access requires a column vector"
        return self[i, 0]

    @always_inline
    def __setitem__(mut self, i: Int, val: Scalar[Self.T]):
        comptime assert Self.colns == 1, "Map 1D access requires a column vector"
        self[i, 0] = val

    @always_inline
    def col(self, c: Int) -> Vector[Self.T, Self.rows]:
        var res = Vector[Self.T, Self.rows]()
        comptime for r in range(Self.rows):
            res[r] = self[r, c]
        return res

    @always_inline
    def block[br: Int, bc: Int](self, row: Int, col: Int) -> Matrix[Self.T, br, bc]:
        var res = Matrix[Self.T, br, bc](uninitialized=True)
        comptime for r in range(br):
            comptime for c in range(bc):
                res[r, c] = self[row + r, col + c]
        return res

    #this is not flattening it is just taking first col
    @always_inline
    def head[n: Int](self) -> Matrix[Self.T, n, 1]:
        comptime assert Self.colns == 1, "Map head() requires a column vector"
        var res = Matrix[Self.T, n, 1](uninitialized=True)
        comptime for i in range(n):
            res[i, 0] = self[i, 0]
        return res

    @staticmethod
    def RowsAtCompileTime() -> Int:
        return Self.rows

    @staticmethod
    def ColsAtCompileTime() -> Int:
        return Self.colns

    @always_inline
    def num_rows(self) -> Int:
        return Self.rows

    @always_inline
    @staticmethod
    def dtype() -> String:
        return (
            "Map["
            + repr(Self.T)
            + ", "
            + String(Self.rows)
            + ", "
            + String(Self.colns)
            + "]"
        )


@fieldwise_init
struct MatrixXd(Defaultable, Movable, Typeable):
    var _rows: Int
    var _cols: Int
    var _data: List[Float64]

    @always_inline
    def __init__(out self):
        self._rows = 0
        self._cols = 0
        self._data = List[Float64]()

    @always_inline
    def __init__(out self, rows: Int, cols: Int, val: Float64 = 0.0):
        self._rows = rows
        self._cols = cols
        self._data = List[Float64](length=rows * cols, fill=val)

    @always_inline
    def rows(self) -> Int:
        return self._rows

    @always_inline
    def cols(self) -> Int:
        return self._cols

    @always_inline
    def __getitem__(self, r: Int, c: Int) -> Float64:
        return self._data[r + c * self._rows]

    @always_inline
    def __setitem__(mut self, r: Int, c: Int, val: Float64):
        self._data[r + c * self._rows] = val

    @staticmethod
    def Zero[rows: Int, cols: Int](
        rows_hint: IntLiteral, cols_hint: IntLiteral
    ) -> Matrix[DType.float64, rows, cols]:
        return Matrix[DType.float64, rows, cols]()

    @staticmethod
    def zero[rows: Int, cols: Int](
        rows_hint: IntLiteral, cols_hint: IntLiteral
    ) -> Matrix[DType.float64, rows, cols]:
        return Matrix[DType.float64, rows, cols]()



    @staticmethod
    def Zero(rows: Int, cols: Int) -> MatrixXd:
        return MatrixXd(rows, cols, 0.0)

    @staticmethod
    def zero(rows: Int, cols: Int) -> MatrixXd:
        return MatrixXd(rows, cols, 0.0)

    @staticmethod
    def Constant(rows: Int, cols: Int, val: Float64) -> MatrixXd:
        return MatrixXd(rows, cols, val)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "MatrixXd"


@fieldwise_init
struct VectorXd(Defaultable, Movable, Typeable):
    var _size: Int
    var _data: List[Float64]

    @always_inline
    def __init__(out self):
        self._size = 0
        self._data = List[Float64]()

    @always_inline
    def __init__(out self, size: Int, val: Float64 = 0.0):
        self._size = size
        self._data = List[Float64](length=size, fill=val)

    @always_inline
    def size(self) -> Int:
        return self._size

    @always_inline
    def rows(self) -> Int:
        return self._size

    @always_inline
    def cols(self) -> Int:
        return 1

    @always_inline
    def __getitem__(self, i: Int) -> Float64:
        return self._data[i]

    @always_inline
    def __setitem__(mut self, i: Int, val: Float64):
        self._data[i] = val

    @staticmethod
    def Zero[rows: Int](rows_hint: IntLiteral) -> Matrix[DType.float64, rows, 1]:
        return Matrix[DType.float64, rows, 1]()

    @staticmethod
    def zero[rows: Int](rows_hint: IntLiteral) -> Matrix[DType.float64, rows, 1]:
        return Matrix[DType.float64, rows, 1]()

    @staticmethod
    def Constant[rows: Int](
        rows_hint: IntLiteral, val: Float64
    ) -> Matrix[DType.float64, rows, 1]:
        return Matrix[DType.float64, rows, 1](val)

    @staticmethod
    def Zero(size: Int) -> VectorXd:
        return VectorXd(size, 0.0)

    @staticmethod
    def zero(size: Int) -> VectorXd:
        return VectorXd(size, 0.0)

    @staticmethod
    def Constant(size: Int, val: Float64) -> VectorXd:
        return VectorXd(size, val)

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "VectorXd"


# Bridges MojoBridge's Matrix[T,rows,cols] (row-major storage) to layout's
# LayoutTensor (used by TrajectoryStateSoA/EigenSoA, expects col-major
# storage for a given Layout) -- two independently-ported Eigen equivalents
# with different memory layouts, so this copies element-by-element into a
# caller-owned buffer rather than reinterpreting the pointer directly.
# The returned tensor carries `buf`'s origin, so `buf` must outlive it.
def to_layout_tensor[
    T: DType, rows: Int, cols: Int
](
    m: Matrix[T, rows, cols], mut buf: InlineArray[Scalar[T], rows * cols]
) -> LayoutTensor[
    mut=True, T, Layout.col_major(rows, cols), origin_of(buf)
]:
    for c in range(cols):
        for r in range(rows):
            buf[c * rows + r] = m[r, c]
    return LayoutTensor[
        mut=True, T, Layout.col_major(rows, cols), origin_of(buf)
    ](Span(buf))
