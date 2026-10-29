from std.atomic import Atomic, Ordering
from std.math import floor, log10


struct SimpleAtomicHisto(Movable, Writable):
    # C++ std::vector<std::atomic<int>>: the bins are a heap buffer shared by
    # all the streams, and are only ever updated with atomic operations.
    var data_: List[Int32]
    var min_: Float32
    var max_: Float32

    def __init__(out self, nbins: Int, min: Float32, max: Float32):
        self.data_ = List[Int32](length=nbins + 2, fill=0)
        self.min_ = min
        self.max_ = max

    # thread safe
    # `self` is read-only: a `mut self` of a struct of this size can be copied
    # in and back out (see doc §12), which would write the struct concurrently
    # from all the streams. Only the bins are modified, through a pointer.
    def fill(self, value: Float32) raises:
        var size = len(self.data_)
        var i: Int
        if value < self.min_:
            i = 0
        elif value >= self.max_:
            i = size - 1
        else:
            i = Int(
                (value - self.min_)
                / (self.max_ - self.min_)
                * Float32(size - 2)
            )
            # handle rounding near maximum
            if i == size - 2:
                i = size - 3
            if not (i >= 0 and i < size - 2):
                raise Error(
                    "SimpleAtomicHisto::fill(",
                    value,
                    "): i ",
                    i,
                    " min ",
                    self.min_,
                    " max ",
                    self.max_,
                    " nbins ",
                    size - 2,
                )
            i += 1
        debug_assert(i >= 0 and i < size)
        _ = Atomic.fetch_add[ordering=Ordering.SEQUENTIAL](
            self.data_.unsafe_ptr().unsafe_mut_cast[True]().unsafe_offset(i),
            Int32(1),
        )

    def dump[W: Writer](self, mut writer: W):
        writer.write(len(self.data_), " ")
        _write_float(writer, self.min_)
        writer.write(" ")
        _write_float(writer, self.max_)
        for item in self.data_:
            writer.write(" ", item)

    def write_to[W: Writer](self, mut writer: W):
        self.dump(writer)


# Write a float as the default std::ostream formatting does, i.e. as printf
# "%g": 6 significant digits, without trailing zeros, in scientific notation
# if the exponent is below -4 or at least 6. Mojo writes floats in the shortest
# representation instead (100000.0 rather than 100000, 4000000.0 rather than
# 4e+06), which would not match the output of the C++ back-ends.
def _write_float[W: Writer](mut writer: W, value: Float32):
    comptime precision = 6
    var v = Float64(value)
    if v == 0:
        writer.write("0")
        return
    if v < 0:
        writer.write("-")
        v = -v

    # the significant digits, as an integer in [10^(precision-1), 10^precision)
    var exponent = Int(floor(log10(v)))
    var digits = Int(round(v * 10.0 ** Float64(precision - 1 - exponent)))
    if digits >= 10**precision:
        exponent += 1
        digits = Int(round(v * 10.0 ** Float64(precision - 1 - exponent)))
    elif digits < 10 ** (precision - 1):
        exponent -= 1
        digits = Int(round(v * 10.0 ** Float64(precision - 1 - exponent)))
    var significand = String(digits)

    if exponent < -4 or exponent >= precision:
        var fraction = String(significand[byte=1:]).rstrip("0")
        writer.write(significand[byte=0:1])
        if fraction:
            writer.write(".", fraction)
        writer.write("e", "-" if exponent < 0 else "+")
        if abs(exponent) < 10:
            writer.write("0")
        writer.write(abs(exponent))
    elif exponent < 0:
        writer.write(
            "0.", String("0") * (-exponent - 1), significand.rstrip("0")
        )
    else:
        var fraction = String(significand[byte = exponent + 1 :]).rstrip("0")
        writer.write(significand[byte = 0 : exponent + 1])
        if fraction:
            writer.write(".", fraction)
