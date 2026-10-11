# Translated from pixeltrack-standalone/src/serial/plugin-PixelTriplets/choleskyInversion.h
# Explicit Cholesky-based inverses for small positive-definite matrices.

import std.math as math
from MojoSerial.MojoBridge.Matrix import Matrix
from MojoSerial.MojoBridge.DTypes import DType

# NOTE: src/dst are plain Matrix[T, rows, cols] (matching invert()'s own
# signature below) rather than a generic AnyType/trait bound, since every
# call site here only ever passes concrete Matrix values.


#
#fully inlined specialized code to perform the inversion of a
#positive defined matrix of rank up to 6.
#
#adapted from ROOT::Math::CholeskyDecomp
#originally by
#@author Manuel Schiller
#@date Aug 29 2008
#
#
#

def invert11[T: DType, rows: Int, cols: Int](src: Matrix[T, rows, cols], mut dst: Matrix[T, rows, cols]):
    var inv = 1.0 / src[0, 0]
    dst[0, 0] = inv



def invert22[T: DType, rows: Int, cols: Int](src: Matrix[T, rows, cols], mut dst: Matrix[T, rows, cols]):
    var luc0 = 1.0 / src[0, 0]
    var luc1 = src[1, 0] * src[1, 0] * luc0
    var luc2 = 1.0 / (src[1, 1] - luc1)

    var li21 = luc1 * luc0 * luc2

    dst[0, 0] = li21 + luc0
    dst[1, 0] = -src[1, 0] * luc0 * luc2
    dst[1, 1] = luc2



def invert33[T: DType, rows: Int, cols: Int](src: Matrix[T, rows, cols], mut dst: Matrix[T, rows, cols]):
    var luc0 = 1.0 / src[0, 0]
    var luc1 = src[1, 0]
    var luc2 = 1.0 / (src[1, 1] - luc0 * luc1 * luc1)
    var luc3 = src[2, 0]
    var luc4 = src[2, 1] - luc0 * luc1 * luc3
    var luc5 = 1.0 / (src[2, 2] - (luc0 * luc3 * luc3 + (luc2 * luc4) * luc4))

    var li21 = -luc0 * luc1
    var li32 = -(luc2 * luc4)
    var li31 = (luc1 * (luc2 * luc4) - luc3) * luc0

    dst[0, 0] = luc5 * li31 * li31 + li21 * li21 * luc2 + luc0
    dst[1, 0] = luc5 * li31 * li32 + li21 * luc2
    dst[1, 1] = luc5 * li32 * li32 + luc2
    dst[2, 0] = luc5 * li31
    dst[2, 1] = luc5 * li32
    dst[2, 2] = luc5



def invert44[T: DType, rows: Int, cols: Int](src: Matrix[T, rows, cols], mut dst: Matrix[T, rows, cols]):
    var luc0 = 1.0 / src[0, 0]
    var luc1 = src[1, 0]
    var luc2 = 1.0 / (src[1, 1] - luc0 * luc1 * luc1)
    var luc3 = src[2, 0]
    var luc4 = src[2, 1] - luc0 * luc1 * luc3
    var luc5 = 1.0 / (src[2, 2] - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4))
    var luc6 = src[3, 0]
    var luc7 = src[3, 1] - luc0 * luc1 * luc6
    var luc8 = src[3, 2] - luc0 * luc3 * luc6 - luc2 * luc4 * luc7
    var luc9 = 1.0 / (src[3, 3] - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * (luc8 * luc5)))

    var li21 = -luc1 * luc0
    var li32 = -luc2 * luc4
    var li31 = (luc1 * (luc2 * luc4) - luc3) * luc0
    var li43 = -(luc8 * luc5)
    var li42 = (luc4 * luc8 * luc5 - luc7) * luc2
    var li41 = (-luc1 * (luc2 * luc4) * (luc8 * luc5) + luc1 * (luc2 * luc7) + luc3 * (luc8 * luc5) - luc6) * luc0

    dst[0, 0] = luc9 * li41 * li41 + luc5 * li31 * li31 + luc2 * li21 * li21 + luc0
    dst[1, 0] = luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21
    dst[1, 1] = luc9 * li42 * li42 + luc5 * li32 * li32 + luc2
    dst[2, 0] = luc9 * li41 * li43 + luc5 * li31
    dst[2, 1] = luc9 * li42 * li43 + luc5 * li32
    dst[2, 2] = luc9 * li43 * li43 + luc5
    dst[3, 0] = luc9 * li41
    dst[3, 1] = luc9 * li42
    dst[3, 2] = luc9 * li43
    dst[3, 3] = luc9



def invert55[T: DType, rows: Int, cols: Int](src: Matrix[T, rows, cols], mut dst: Matrix[T, rows, cols]):
    var luc0 = 1.0 / src[0, 0]
    var luc1 = src[1, 0]
    var luc2 = 1.0 / (src[1, 1] - luc0 * luc1 * luc1)
    var luc3 = src[2, 0]
    var luc4 = src[2, 1] - luc0 * luc1 * luc3
    var luc5 = 1.0 / (src[2, 2] - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4))
    var luc6 = src[3, 0]
    var luc7 = src[3, 1] - luc0 * luc1 * luc6
    var luc8 = src[3, 2] - luc0 * luc3 * luc6 - luc2 * luc4 * luc7
    var luc9 = 1.0 / (src[3, 3] - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * (luc8 * luc5)))
    var luc10 = src[4, 0]
    var luc11 = src[4, 1] - luc0 * luc1 * luc10
    var luc12 = src[4, 2] - luc0 * luc3 * luc10 - luc2 * luc4 * luc11
    var luc13 = src[4, 3] - luc0 * luc6 * luc10 - luc2 * luc7 * luc11 - luc5 * luc8 * luc12
    var luc14 = 1.0 / (
        src[4, 4]
        - (luc0 * luc10 * luc10 + luc2 * luc11 * luc11 + luc5 * luc12 * luc12 + luc9 * luc13 * luc13)
    )

    var li21 = -luc1 * luc0
    var li32 = -luc2 * luc4
    var li31 = (luc1 * (luc2 * luc4) - luc3) * luc0
    var li43 = -(luc8 * luc5)
    var li42 = (luc4 * luc8 * luc5 - luc7) * luc2
    var li41 = (-luc1 * (luc2 * luc4) * (luc8 * luc5) + luc1 * (luc2 * luc7) + luc3 * (luc8 * luc5) - luc6) * luc0
    var li54 = -luc13 * luc9
    var li53 = (luc13 * luc8 * luc9 - luc12) * luc5
    var li52 = (-luc4 * luc8 * luc13 * luc5 * luc9 + luc4 * luc12 * luc5 + luc7 * luc13 * luc9 - luc11) * luc2
    var li51 = (
        luc1 * luc4 * luc8 * luc13 * luc2 * luc5 * luc9
        - luc13 * luc8 * luc3 * luc9 * luc5
        - luc12 * luc4 * luc1 * luc2 * luc5
        - luc13 * luc7 * luc1 * luc9 * luc2
        + luc11 * luc1 * luc2
        + luc12 * luc3 * luc5
        + luc13 * luc6 * luc9
        - luc10
    ) * luc0

    dst[0, 0] = luc14 * li51 * li51 + luc9 * li41 * li41 + luc5 * li31 * li31 + luc2 * li21 * li21 + luc0
    dst[1, 0] = luc14 * li51 * li52 + luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21
    dst[1, 1] = luc14 * li52 * li52 + luc9 * li42 * li42 + luc5 * li32 * li32 + luc2
    dst[2, 0] = luc14 * li51 * li53 + luc9 * li41 * li43 + luc5 * li31
    dst[2, 1] = luc14 * li52 * li53 + luc9 * li42 * li43 + luc5 * li32
    dst[2, 2] = luc14 * li53 * li53 + luc9 * li43 * li43 + luc5
    dst[3, 0] = luc14 * li51 * li54 + luc9 * li41
    dst[3, 1] = luc14 * li52 * li54 + luc9 * li42
    dst[3, 2] = luc14 * li53 * li54 + luc9 * li43
    dst[3, 3] = luc14 * li54 * li54 + luc9
    dst[4, 0] = luc14 * li51
    dst[4, 1] = luc14 * li52
    dst[4, 2] = luc14 * li53
    dst[4, 3] = luc14 * li54
    dst[4, 4] = luc14



def invert66[T: DType, rows: Int, cols: Int](src: Matrix[T, rows, cols], mut dst: Matrix[T, rows, cols]):
    var luc0 = 1.0 / src[0, 0]
    var luc1 = src[1, 0]
    var luc2 = 1.0 / (src[1, 1] - luc0 * luc1 * luc1)
    var luc3 = src[2, 0]
    var luc4 = src[2, 1] - luc0 * luc1 * luc3
    var luc5 = 1.0 / (src[2, 2] - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4))
    var luc6 = src[3, 0]
    var luc7 = src[3, 1] - luc0 * luc1 * luc6
    var luc8 = src[3, 2] - luc0 * luc3 * luc6 - luc2 * luc4 * luc7
    var luc9 = 1.0 / (src[3, 3] - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * (luc8 * luc5)))
    var luc10 = src[4, 0]
    var luc11 = src[4, 1] - luc0 * luc1 * luc10
    var luc12 = src[4, 2] - luc0 * luc3 * luc10 - luc2 * luc4 * luc11
    var luc13 = src[4, 3] - luc0 * luc6 * luc10 - luc2 * luc7 * luc11 - luc5 * luc8 * luc12
    var luc14 = 1.0 / (
        src[4, 4]
        - (luc0 * luc10 * luc10 + luc2 * luc11 * luc11 + luc5 * luc12 * luc12 + luc9 * luc13 * luc13)
    )
    var luc15 = src[5, 0]
    var luc16 = src[5, 1] - luc0 * luc1 * luc15
    var luc17 = src[5, 2] - luc0 * luc3 * luc15 - luc2 * luc4 * luc16
    var luc18 = src[5, 3] - luc0 * luc6 * luc15 - luc2 * luc7 * luc16 - luc5 * luc8 * luc17
    var luc19 = (
        src[5, 4]
        - luc0 * luc10 * luc15 - luc2 * luc11 * luc16 - luc5 * luc12 * luc17 - luc9 * luc13 * luc18
    )
    var luc20 = 1.0 / (
        src[5, 5]
        - (luc0 * luc15 * luc15 + luc2 * luc16 * luc16 + luc5 * luc17 * luc17 + luc9 * luc18 * luc18 + luc14 * luc19 * luc19)
    )

    var li21 = -luc1 * luc0
    var li32 = -luc2 * luc4
    var li31 = (luc1 * (luc2 * luc4) - luc3) * luc0
    var li43 = -(luc8 * luc5)
    var li42 = (luc4 * luc8 * luc5 - luc7) * luc2
    var li41 = (-luc1 * (luc2 * luc4) * (luc8 * luc5) + luc1 * (luc2 * luc7) + luc3 * (luc8 * luc5) - luc6) * luc0
    var li54 = -luc13 * luc9
    var li53 = (luc13 * luc8 * luc9 - luc12) * luc5
    var li52 = (-luc4 * luc8 * luc13 * luc5 * luc9 + luc4 * luc12 * luc5 + luc7 * luc13 * luc9 - luc11) * luc2
    var li51 = (
        luc1 * luc4 * luc8 * luc13 * luc2 * luc5 * luc9
        - luc13 * luc8 * luc3 * luc9 * luc5
        - luc12 * luc4 * luc1 * luc2 * luc5
        - luc13 * luc7 * luc1 * luc9 * luc2
        + luc11 * luc1 * luc2
        + luc12 * luc3 * luc5
        + luc13 * luc6 * luc9
        - luc10
    ) * luc0

    var li65 = -luc19 * luc14
    var li64 = (luc19 * luc14 * luc13 - luc18) * luc9
    var li63 = (
        -luc8 * luc13 * (luc19 * luc14) * luc9
        + luc8 * luc9 * luc18
        + luc12 * (luc19 * luc14)
        - luc17
    ) * luc5
    var li62 = (
        luc4 * (luc8 * luc9) * luc13 * luc5 * (luc19 * luc14)
        - luc18 * luc4 * (luc8 * luc9) * luc5
        - luc19 * luc12 * luc4 * luc14 * luc5
        - luc19 * luc13 * luc7 * luc14 * luc9
        + luc17 * luc4 * luc5
        + luc18 * luc7 * luc9
        + luc19 * luc11 * luc14
        - luc16
    ) * luc2
    var li61 = (
        -luc19 * luc13 * luc8 * luc4 * luc1 * luc2 * luc5 * luc9 * luc14
        + luc18 * luc8 * luc4 * luc1 * luc2 * luc5 * luc9
        + luc19 * luc12 * luc4 * luc1 * luc2 * luc5 * luc14
        + luc19 * luc13 * luc7 * luc1 * luc2 * luc9 * luc14
        + luc19 * luc13 * luc8 * luc3 * luc5 * luc9 * luc14
        - luc17 * luc4 * luc1 * luc2 * luc5
        - luc18 * luc7 * luc1 * luc2 * luc9
        - luc19 * luc11 * luc1 * luc2 * luc14
        - luc18 * luc8 * luc3 * luc5 * luc9
        - luc19 * luc12 * luc3 * luc5 * luc14
        - luc19 * luc13 * luc6 * luc9 * luc14
        + luc16 * luc1 * luc2
        + luc17 * luc3 * luc5
        + luc18 * luc6 * luc9
        + luc19 * luc10 * luc14
        - luc15
    ) * luc0

    dst[0, 0] = luc20 * li61 * li61 + luc14 * li51 * li51 + luc9 * li41 * li41 + luc5 * li31 * li31 + luc2 * li21 * li21 + luc0
    dst[1, 0] = luc20 * li61 * li62 + luc14 * li51 * li52 + luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21
    dst[1, 1] = luc20 * li62 * li62 + luc14 * li52 * li52 + luc9 * li42 * li42 + luc5 * li32 * li32 + luc2
    dst[2, 0] = luc20 * li61 * li63 + luc14 * li51 * li53 + luc9 * li41 * li43 + luc5 * li31
    dst[2, 1] = luc20 * li62 * li63 + luc14 * li52 * li53 + luc9 * li42 * li43 + luc5 * li32
    dst[2, 2] = luc20 * li63 * li63 + luc14 * li53 * li53 + luc9 * li43 * li43 + luc5
    dst[3, 0] = luc20 * li61 * li64 + luc14 * li51 * li54 + luc9 * li41
    dst[3, 1] = luc20 * li62 * li64 + luc14 * li52 * li54 + luc9 * li42
    dst[3, 2] = luc20 * li63 * li64 + luc14 * li53 * li54 + luc9 * li43
    dst[3, 3] = luc20 * li64 * li64 + luc14 * li54 * li54 + luc9
    dst[4, 0] = luc20 * li61 * li65 + luc14 * li51
    dst[4, 1] = luc20 * li62 * li65 + luc14 * li52
    dst[4, 2] = luc20 * li63 * li65 + luc14 * li53
    dst[4, 3] = luc20 * li64 * li65 + luc14 * li54
    dst[4, 4] = luc20 * li65 * li65 + luc14
    dst[5, 0] = luc20 * li61
    dst[5, 1] = luc20 * li62
    dst[5, 2] = luc20 * li63
    dst[5, 3] = luc20 * li64
    dst[5, 4] = luc20 * li65
    dst[5, 5] = luc20



def symmetrize11[T: DType, rows: Int, cols: Int](mut dst: Matrix[T, rows, cols]):
    pass



def symmetrize22[T: DType, rows: Int, cols: Int](mut dst: Matrix[T, rows, cols]):
    dst[0, 1] = dst[1, 0]



def symmetrize33[T: DType, rows: Int, cols: Int](mut dst: Matrix[T, rows, cols]):
    symmetrize22(dst)
    dst[0, 2] = dst[2, 0]
    dst[1, 2] = dst[2, 1]



def symmetrize44[T: DType, rows: Int, cols: Int](mut dst: Matrix[T, rows, cols]):
    symmetrize33(dst)
    dst[0, 3] = dst[3, 0]
    dst[1, 3] = dst[3, 1]
    dst[2, 3] = dst[3, 2]



def symmetrize55[T: DType, rows: Int, cols: Int](mut dst: Matrix[T, rows, cols]):
    symmetrize44(dst)
    dst[0, 4] = dst[4, 0]
    dst[1, 4] = dst[4, 1]
    dst[2, 4] = dst[4, 2]
    dst[3, 4] = dst[4, 3]



def symmetrize66[T: DType, rows: Int, cols: Int](mut dst: Matrix[T, rows, cols]):
    symmetrize55(dst)
    dst[0, 5] = dst[5, 0]
    dst[1, 5] = dst[5, 1]
    dst[2, 5] = dst[5, 2]
    dst[3, 5] = dst[5, 3]
    dst[4, 5] = dst[5, 4]


struct Inverter[T: DType, rows: Int, cols: Int, N: Int]:
    @staticmethod
    @always_inline
    def eval(src: Matrix[Self.T, Self.rows, Self.cols], mut dst: Matrix[Self.T, Self.rows, Self.cols]):
        comptime if Self.N == 1:
            invert11(src, dst)
        elif Self.N == 2:
            invert22(src, dst)
            symmetrize22(dst)
        elif Self.N == 3:
            invert33(src, dst)
            symmetrize33(dst)
        elif Self.N == 4:
            invert44(src, dst)
            symmetrize44(dst)
        elif Self.N == 5:
            invert55(src, dst)
            symmetrize55(dst)
        elif Self.N == 6:
            invert66(src, dst)
            symmetrize66(dst)
        else:
            dst = src.inverse()



def invert[T: DType, rows: Int, cols: Int](
    src: Matrix[T, rows, cols], mut dst: Matrix[T, rows, cols]
):
    Inverter[
        T,
        rows,
        cols,
        Matrix[T, rows, cols].ColsAtCompileTime(),
    ].eval(src, dst)
