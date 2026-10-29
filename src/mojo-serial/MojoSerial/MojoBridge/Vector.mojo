from std.memory import bitcast
from std.math import Ceilable, CeilDivable, Floorable, Truncable
from std.sys import align_of, is_gpu, size_of
from std.bit import pop_count
from std.utils.numerics import max_finite as _max_finite
from std.utils.numerics import max_or_inf as _max_or_inf
from std.utils.numerics import min_finite as _min_finite
from std.utils.numerics import min_or_neg_inf as _min_or_neg_inf
from std.hashlib.hasher import Hasher

from MojoSerial.MojoBridge.DTypes import Typeable


@always_inline
def _pow_2[T: DType, //, n: Scalar[T]]() -> Scalar[T]:
    comptime num_bits = size_of[T]() * 8
    var result = n - 1

    comptime if num_bits > 1:
        result |= result >> 1

    comptime if num_bits > 2:
        result |= result >> 2

    comptime if num_bits > 4:
        result |= result >> 4

    comptime if num_bits > 8:
        result |= result >> 8

    comptime if num_bits > 16:
        result |= result >> 16

    comptime if num_bits > 32:
        result |= result >> 32

    comptime if num_bits > 64:
        result |= result >> 64
    return result + 1


@always_inline
def _pow_2[n: Int]() -> Int:
    var result = n - 1
    result |= result >> 1
    result |= result >> 2
    result |= result >> 4
    result |= result >> 8
    result |= result >> 16
    result |= result >> 32
    return result + 1


@fieldwise_init
struct _VecIterator[
    W: DType,
    size: Int,
    vec_origin: Origin,
    forward: Bool = True,
](Copyable, Iterator, Movable, Typeable):
    comptime vec_type = Vector[Self.W, Self.size]
    comptime T = Scalar[Self.W]
    comptime Element = Self.T

    var index: Int
    var src: Pointer[Self.vec_type, Self.vec_origin]

    def __next_ref__(mut self) -> Self.T:
        comptime if Self.forward:
            self.index += 1
            return self.src[][self.index - 1]
        else:
            self.index -= 1
            return self.src[][self.index]

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
            "_VecIterator["
            + repr(Self.W)
            + ", "
            + String(Self.size)
            + ", Origin["
            + String(Self.vec_origin.mut)
            + "], "
            + String(Self.forward)
            + "]"
        )


@fieldwise_init
struct Vector[T: DType, size: Int](
    Absable,
    CeilDivable,
    Ceilable,
    Copyable,
    Defaultable,
    Floorable,
    Hashable,
    ImplicitlyCopyable,
    Movable,
    Powable,
    Roundable,
    Sized,
    Truncable,
    Typeable,
    Writable,
    TrivialRegisterPassable,
):
    comptime psize = _pow_2[Self.size]()
    comptime _D = Scalar[Self.T]
    comptime _DC = SIMD[Self.T, Self.psize]
    comptime _Mask = Vector[DType.bool, Self.size]
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
        self._data = Self._DC()

    @always_inline
    def copy(self) -> Self:
        """Explicitly construct a copy of self."""
        return Self(copy=self)

    @implicit
    def __init__[vsize: Int, //](out self, vec: Vector[Self.T, vsize]):
        """Initialize a vector from an arbitrary vector. Might cause data loss (implicit).
        """
        self = Self()

        comptime for i in range(min(Self.size, vsize)):
            self._data[i] = vec[i]

    @implicit
    def __init__[vsize: Int, //](out self, vec: SIMD[Self.T, vsize]):
        """Initialize a vector from an arbitrary SIMD vector. Might cause data loss (implicit).
        """
        self = Self()

        comptime for i in range(min(Self.size, vsize)):
            self._data[i] = vec[i]

    @always_inline
    @implicit
    def __init__(out self, var vec: SIMD[Self.T, Self.size], /):
        """Initialize a vector from a SIMD object of the same size (implicit).
        """
        self._data = rebind[Self._DC](vec)

    @implicit
    def __init__(out self, values: List[Self._D], /):
        """Initialize a vector from a list of values. Might cause data loss (implicit).
        """
        self = Self()
        for i in range(min(Self.size, values.__len__())):
            self._data[i] = values[i]

    @always_inline
    @implicit
    def __init__(out self, *values: Self._D, __list_literal__: () = ()):
        """Constructs a vector via a variadic list of values in a literal format.
        """
        self = Self()
        for i in range(values.__len__()):
            self._data[i] = values[i]

    def __init__(out self, *values: Self._D):
        """Initialize a vector from a variadic list of values."""
        self = Self()
        for i in range(values.__len__()):
            self._data[i] = values[i]

    @always_inline
    def __init__[U: DType, //](out self, val: Scalar[U], /):
        """Initializes a vector with a scalar.
        The scalar is splatted across all the elements of the vector."""
        self._data = Self._DC(val)

    @always_inline
    def __init__(out self, val: Int, /):
        """Initializes a vector with a signed integer.
        The signed integer is splatted across all the elements of the vector."""
        self._data = Self._DC(val)

    @always_inline
    def __init__(out self, val: UInt, /):
        """Initializes a vector with an unsigned integer.
        The unsigned integer is splatted across all the elements of the vector.
        """
        self._data = Self._DC(val)

    @always_inline
    @implicit
    def __init__(out self, val: IntLiteral, /):
        """Initializes a vector with an integer literal (implicit).
        The signed integer is splatted across all the elements of the vector."""
        self._data = Self._DC(val)

    @always_inline
    def __init__[U: DType, //](out self, value: SIMD[U, Self.size], /):
        """Initializes a vector with a SIMD vector of the same size and of a different data type.
        """
        self._data = rebind[Self._DC](value.cast[Self.T]())

    @always_inline
    def __init__[U: DType, //](out self, vec: Vector[U, Self.size], /):
        """Initializes a vector with a vector of the same size and of a different data type.
        """
        self._data = rebind[Self._DC](vec._data.cast[Self.T]())

    def __init__[*, offset: Int](out self, vec: Vector[Self.T, _]):
        """Initializes a vector as a slice of another vector with specified output size and offset.
        """
        comptime output_width = Self.size

        self = Self()
        var i = 0

        comptime for j in range(offset, offset + output_width):
            self._data[i] = vec._data[j]
            i += 1

    @staticmethod
    def from_bits[U: DType, //](value: SIMD[U, Self.size]) -> Vector[U, Self.size]:
        """Initializes a vector from the bits of an integral SIMD vector."""
        comptime assert U.is_integral(), "DType must be integral"
        return Vector[U, Self.size](bitcast[U](value))

    # Operators

    @always_inline
    def __getitem__(self, idx: Int) -> Self._D:
        return self._data[idx]

    @always_inline
    def __setitem__(mut self, idx: Int, val: Self._D):
        self._data[idx] = val

    def __iter__(ref self) -> _VecIterator[Self.T, Self.size, origin_of(self)]:
        return _VecIterator[Self.T, Self.size, origin_of(self)](
            0, Pointer(to=self)
        )

    @always_inline
    def __contains__(self, value: Self._D) -> Bool:
        return self._data.__contains__(value)

    @always_inline
    def __add__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return self._data + rhs._data

    @always_inline
    def __sub__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return self._data - rhs._data

    @always_inline
    def __mul__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return self._data * rhs._data

    @always_inline
    def __matmul__(self, rhs: Self) -> Self._D:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        var res: Self._D = 0
        for i in range(Self.size):
            res += self._data[i] * rhs._data[i]
        return res

    @always_inline
    def __truediv__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return self._data / rhs._data

    @always_inline
    def __floordiv__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return self._data // rhs._data

    @always_inline
    def __mod__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return self._data % rhs._data

    @always_inline
    def __pow__(self, exp: Int) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return self._data**exp

    @always_inline
    def __pow__(self, exp: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return self._data**exp._data

    # 1.0 reserves SIMD's comparison operators for `Scalar` (they return Bool);
    # the elementwise forms are the named methods.
    @always_inline
    def __lt__(self, rhs: Self) -> Self._Mask:
        return self._data.lt(rhs._data)

    @always_inline
    def __le__(self, rhs: Self) -> Self._Mask:
        return self._data.le(rhs._data)

    @always_inline
    def __eq__(self, rhs: Self) -> Self._Mask:
        return self._data.eq(rhs._data)

    @always_inline
    def __ne__(self, rhs: Self) -> Self._Mask:
        return self._data.ne(rhs._data)

    @always_inline
    def __gt__(self, rhs: Self) -> Self._Mask:
        return self._data.gt(rhs._data)

    @always_inline
    def __ge__(self, rhs: Self) -> Self._Mask:
        return self._data.ge(rhs._data)

    @always_inline
    def __pos__(self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return self

    @always_inline
    def __neg__(self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return -self._data

    @always_inline
    def __and__(self, rhs: Self) -> Self:
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType must be an integral or bool type"
        return self._data & rhs._data

    @always_inline
    def __xor__(self, rhs: Self) -> Self:
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType must be an integral or bool type"
        return self._data ^ rhs._data

    @always_inline
    def __or__(self, rhs: Self) -> Self:
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType must be an integral or bool type"
        return self._data | rhs._data

    @always_inline
    def __lshift__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_integral(), "DType must be an integral type"
        return self._data << rhs._data

    @always_inline
    def __rshift__(self, rhs: Self) -> Self:
        comptime assert Self.T.is_integral(), "DType must be an integral type"
        return self._data >> rhs._data

    @always_inline
    def __invert__(self) -> Self:
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "DType must be an integral or bool type"
        return ~self._data

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
        self = self * rhs

    @always_inline("nodebug")
    def __itruediv__(mut self, rhs: Self):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        self = self / rhs

    @always_inline("nodebug")
    def __ifloordiv__(mut self, rhs: Self):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        self = self // rhs

    @always_inline("nodebug")
    def __imod__(mut self, rhs: Self):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        self = self.__mod__(rhs)

    @always_inline("nodebug")
    def __ipow__(mut self, rhs: Int):
        comptime assert Self.T.is_numeric(), "DType must be numeric"
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
    def __rmul__(self, value: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return value * self

    @always_inline
    def __rmatmul__(self, rhs: Self) -> Self._D:
        return rhs @ self

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
    def __rpow__(self, base: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return base**self

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
        return "Vector[" + repr(Self.T) + ", " + String(Self.size) + "]"

    @always_inline
    def __len__(self) -> Int:
        return Self.size

    @always_inline
    def __str__(self) -> String:
        return String(self)

    def __repr__(self) -> String:
        var output = String()
        output.write("Vector[" + repr(Self.T) + ", ", Self.size, "](")
        for i in range(self.__len__()):
            output.write(self[i])
            if i < self.__len__() - 1:
                output.write(", ")
        output.write(")")
        return output^

    @always_inline
    def __floor__(self) -> Self:
        return self._data.__floor__()

    @always_inline
    def __ceil__(self) -> Self:
        return self._data.__ceil__()

    @always_inline
    def __trunc__(self) -> Self:
        return self._data.__trunc__()

    @always_inline
    def __abs__(self) -> Self:
        return self._data.__abs__()

    @always_inline
    def __round__(self) -> Self:
        return self._data.__round__()

    @always_inline
    def __round__(self, ndigits: Int) -> Self:
        return self._data.__round__(ndigits)

    @always_inline
    def __ceildiv__(self, denominator: Self) -> Self:
        return self._data.__ceildiv__(denominator._data)

    def __hash__[H: Hasher](self, mut hasher: H):
        hasher._update_with_simd(self._data)
        hasher._update_with_simd(Scalar[DType.uint64](37))

    # Methods

    @always_inline("nodebug")
    def _refine[
        _T: DType = Self.T, _size: Int = Self.size
    ](self) -> Vector[_T, _size]:
        return rebind[Vector[_T, _size]](self)

    @always_inline
    def cast[target: DType](self) -> Vector[target, Self.size]:
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

        return self._data.cast[target]()

    @always_inline
    def is_power_of_two(self) -> Self._Mask:
        comptime assert Self.T.is_integral(), "DType must be integral"

        comptime if Self.T.is_unsigned():
            return Self._Mask(pop_count(self._data) == 1)
        else:
            return (self > 0) & (self & (self - 1) == 0)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("[")
        for i in range(self.__len__()):
            writer.write(self[i])
            if i < self.__len__() - 1:
                writer.write(", ")
        writer.write("]")

    @always_inline
    def clamp(self, lower_bound: Self, upper_bound: Self) -> Self:
        return self._data.clamp(lower_bound._data, upper_bound._data)

    @always_inline
    def fma(self, multiplier: Self, accumulator: Self) -> Self:
        comptime assert Self.T.is_numeric(), "DType must be numeric"
        return self._data.fma(multiplier._data, accumulator._data)

    def slice[
        output_width: Int, /, *, offset: Int = 0
    ](self) -> Vector[Self.T, output_width]:
        comptime assert (
            0 <= offset < output_width + offset <= Self.size
        ), "Output width must be a positive integer less than size"

        comptime if output_width == 1:
            return self[offset]

        return Vector[Self.T, output_width].__init__[offset=offset](self)

    def insert[*, offset: Int = 0](self, value: Vector[Self.T, _]) -> Self:
        comptime input_width = value.size
        comptime assert (
            0 <= offset < input_width + offset <= Self.size
        ), "Insertion position must not exceed the size of the vector"

        comptime if Self.size == 1:
            comptime assert (
                input_width == 1
            ), "The input width must be 1 if the size is 1"
            return value[0]

        return self._data.insert[offset=offset](value._data)

    def iinsert[*, offset: Int = 0](mut self, value: Vector[Self.T, _]):
        comptime input_width = value.size
        comptime assert (
            0 <= offset < input_width + offset <= Self.size
        ), "Insertion position must not exceed the size of the vector"

        comptime if Self.size == 1:
            comptime assert (
                input_width == 1
            ), "The input width must be 1 if the size is 1"
            self._data[0] = value[0]

        self._data = self._data.insert[offset=offset](value._data)

    def join[
        vsize: Int, //
    ](self, other: Vector[Self.T, vsize]) -> Vector[Self.T, Self.size + vsize]:
        var res = Vector[Self.T, Self.size + vsize]()
        res.iinsert(self)
        res.iinsert[offset=Self.size](other)
        return res

    def interleave[
        vsize: Int, //
    ](self, other: Vector[Self.T, vsize]) -> Vector[Self.T, Self.size + vsize]:
        var res = Vector[Self.T, Self.size + vsize]()
        var u = 0
        var v = 0

        comptime for i in range(min(Self.size, vsize) * 2):
            if i % 2 == 0:
                res[i] = self[u]
                u += 1
            else:
                res[i] = other[v]
                v += 1

        comptime if Self.size > vsize:

            comptime for i in range(vsize * 2, vsize + Self.size):
                res[i] = self[u]
                u += 1

        comptime if vsize > Self.size:

            comptime for i in range(Self.size * 2, vsize + Self.size):
                res[i] = other[v]
                v += 1
        return res

    @always_inline
    def split(self) -> Tuple[Vector[Self.T, Self.size // 2], Vector[Self.T, Self.size // 2]]:
        comptime assert (
            Self.size % 2 == 0 and Self.size > 1
        ), "Vector size must be divisible by 2 for splitting"
        comptime half_size = Self.size // 2
        var se = self.slice[half_size]()
        var lf = self.slice[half_size, offset=half_size]()
        return se, lf

    @always_inline
    def deinterleave(self) -> Tuple[Vector[Self.T, Self.size // 2], Vector[Self.T, Self.size // 2]]:
        comptime assert (
            Self.size % 2 == 0 and Self.size > 1
        ), "Vector size must be divisible by 2 for deinterleaving"

        comptime if Self.size == 2:
            return self[0], self[1]

        var res = Vector[Self.T, Self.size // 2](), Vector[Self.T, Self.size // 2]()

        comptime for i in range(Self.size // 2):
            res[0][i] = self[2 * i]
            res[1][i] = self[2 * i + 1]
        return res[0], res[1]

    def reversed(self) -> Self:
        var res = self

        comptime for i in range(Self.size // 2):
            res[i], res[Self.size - 1 - i] = res[Self.size - 1 - i], res[i]
        return res

    def pop_count(self) -> Self:
        return pop_count(self._data)

    # Reductions

    def reduce_max(self) -> Self._D:
        comptime if self.size == 1:
            return self._data[0]
        var A = self._data[0]

        comptime for i in range(1, Self.size):
            A = max(A, self._data[i])
        return A

    def reduce_min(self) -> Self._D:
        comptime if self.size == 1:
            return self._data[0]
        var A = self._data[0]

        comptime for i in range(1, Self.size):
            A = min(A, self._data[i])
        return A

    def reduce_add(self) -> Self._D:
        comptime if self.size == 1:
            return self._data[0]
        var A = self._data[0]

        comptime for i in range(1, Self.size):
            A = A + self._data[i]
        return A

    def reduce_mul(self) -> Self._D:
        comptime if self.size == 1:
            return self._data[0]
        var A = self._data[0]

        comptime for i in range(1, Self.size):
            A = A * self._data[i]
        return A

    def reduce_and(self) -> Self._D:
        comptime if self.size == 1:
            return self._data[0]
        var A = self._data[0]

        comptime for i in range(1, Self.size):
            A = A & self._data[i]
        return A

    def reduce_or(self) -> Self._D:
        comptime if self.size == 1:
            return self._data[0]
        var A = self._data[0]

        comptime for i in range(1, Self.size):
            A = A | self._data[i]
        return A

    def reduce_bit_count(self) -> Int:
        comptime assert (
            Self.T.is_integral() or Self.T == DType.bool
        ), "Expected either integral or bool type"

        comptime if Self.T == DType.bool:
            return Int(self.cast[DType.uint8]().reduce_add())
        else:
            return Int(Vector[Self.T, Self.size](pop_count(self._data)).reduce_add())
