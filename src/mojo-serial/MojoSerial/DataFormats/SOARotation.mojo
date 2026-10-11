from std.collections import Span

from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct TkRotation[T: DType](Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    @always_inline
    @staticmethod
    def dtype() -> String:
        return "TkRotation"


@fieldwise_init
struct SOARotation[T: DType](Copyable, Defaultable, Movable, TrivialRegisterPassable):
    var R11: Scalar[Self.T]
    var R12: Scalar[Self.T]
    var R13: Scalar[Self.T]
    var R21: Scalar[Self.T]
    var R22: Scalar[Self.T]
    var R23: Scalar[Self.T]
    var R31: Scalar[Self.T]
    var R32: Scalar[Self.T]
    var R33: Scalar[Self.T]

    @always_inline
    def __init__(out self):
        self.R11 = 0
        self.R12 = 0
        self.R13 = 0
        self.R21 = 0
        self.R22 = 0
        self.R23 = 0
        self.R31 = 0
        self.R32 = 0
        self.R33 = 0

    @always_inline
    def __init__(out self, var x: Scalar[Self.T], /):
        self.R11 = 1
        self.R12 = 0
        self.R13 = 0
        self.R21 = 0
        self.R22 = 1
        self.R23 = 0
        self.R31 = 0
        self.R32 = 0
        self.R33 = 1

    @always_inline
    def __init__(out self, p: Span[Scalar[Self.T], _]):
        self.R11 = p[0]
        self.R12 = p[1]
        self.R13 = p[2]
        self.R21 = p[3]
        self.R22 = p[4]
        self.R23 = p[5]
        self.R31 = p[6]
        self.R32 = p[7]
        self.R33 = p[8]

    @always_inline
    def transposed(self) -> Self:
        return Self(
            self.R11,
            self.R21,
            self.R31,
            self.R12,
            self.R22,
            self.R32,
            self.R13,
            self.R23,
            self.R33,
        )

    @always_inline
    def multiply(
        self,
        var vx: Scalar[Self.T],
        var vy: Scalar[Self.T],
        var vz: Scalar[Self.T],
        mut ux: Scalar[Self.T],
        mut uy: Scalar[Self.T],
        mut uz: Scalar[Self.T],
    ):
        ux = self.R11 * vx + self.R12 * vy + self.R13 * vz
        uy = self.R21 * vx + self.R22 * vy + self.R23 * vz
        uz = self.R31 * vx + self.R32 * vy + self.R33 * vz

    @always_inline
    def multiplyInverse(
        self,
        var vx: Scalar[Self.T],
        var vy: Scalar[Self.T],
        var vz: Scalar[Self.T],
        mut ux: Scalar[Self.T],
        mut uy: Scalar[Self.T],
        mut uz: Scalar[Self.T],
    ):
        ux = self.R11 * vx + self.R21 * vy + self.R31 * vz
        uy = self.R12 * vx + self.R22 * vy + self.R32 * vz
        uz = self.R13 * vx + self.R23 * vy + self.R33 * vz

    @always_inline
    def multiplyInverse(
        self,
        var vx: Scalar[Self.T],
        var vy: Scalar[Self.T],
        mut ux: Scalar[Self.T],
        mut uy: Scalar[Self.T],
        mut uz: Scalar[Self.T],
    ):
        ux = self.R11 * vx + self.R21 * vy
        uy = self.R12 * vx + self.R22 * vy
        uz = self.R13 * vx + self.R23 * vy

    @always_inline
    def xx(self) -> Scalar[Self.T]:
        return self.R11

    @always_inline
    def xy(self) -> Scalar[Self.T]:
        return self.R12

    @always_inline
    def xz(self) -> Scalar[Self.T]:
        return self.R13

    @always_inline
    def yx(self) -> Scalar[Self.T]:
        return self.R21

    @always_inline
    def yy(self) -> Scalar[Self.T]:
        return self.R22

    @always_inline
    def yz(self) -> Scalar[Self.T]:
        return self.R23

    @always_inline
    def zx(self) -> Scalar[Self.T]:
        return self.R31

    @always_inline
    def zy(self) -> Scalar[Self.T]:
        return self.R32

    @always_inline
    def zz(self) -> Scalar[Self.T]:
        return self.R33

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SOARotation[" + String(Self.T) + "]"


@fieldwise_init
struct SOAFrame[T: DType](Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    var px: Scalar[Self.T]
    var py: Scalar[Self.T]
    var pz: Scalar[Self.T]
    var rot: SOARotation[Self.T]

    @always_inline
    def __init__(out self):
        self.px = 0
        self.py = 0
        self.pz = 0
        self.rot = SOARotation[Self.T]()

    @always_inline
    def rotation(self) -> SOARotation[Self.T]:
        return self.rot

    @always_inline
    def toLocal(
        self,
        var vx: Scalar[Self.T],
        var vy: Scalar[Self.T],
        var vz: Scalar[Self.T],
        mut ux: Scalar[Self.T],
        mut uy: Scalar[Self.T],
        mut uz: Scalar[Self.T],
    ):
        self.rot.multiply(vx - self.px, vy - self.py, vz - self.pz, ux, uy, uz)

    @always_inline
    def toGlobal(
        self,
        var vx: Scalar[Self.T],
        var vy: Scalar[Self.T],
        var vz: Scalar[Self.T],
        mut ux: Scalar[Self.T],
        mut uy: Scalar[Self.T],
        mut uz: Scalar[Self.T],
    ):
        self.rot.multiplyInverse(vx, vy, vz, ux, uy, uz)
        ux += self.px
        uy += self.py
        uz += self.pz

    @always_inline
    def toGlobal(
        self,
        var vx: Scalar[Self.T],
        var vy: Scalar[Self.T],
        mut ux: Scalar[Self.T],
        mut uy: Scalar[Self.T],
        mut uz: Scalar[Self.T],
    ):
        self.rot.multiplyInverse(vx, vy, ux, uy, uz)
        ux += self.px
        uy += self.py
        uz += self.pz

    @always_inline
    def toGlobal(
        self,
        var cxx: Scalar[Self.T],
        var cxy: Scalar[Self.T],
        var cyy: Scalar[Self.T],
        gl: Span[mut=True, Scalar[Self.T], _],
    ):
        var r = self.rot

        gl[0] = r.xx() * (r.xx() * cxx + r.yx() * cxy) + r.yx() * (
            r.xx() * cxy + r.yx() * cyy
        )
        gl[1] = r.xx() * (r.xy() * cxx + r.yy() * cxy) + r.yx() * (
            r.xy() * cxy + r.yy() * cyy
        )
        gl[2] = r.xy() * (r.xy() * cxx + r.yy() * cxy) + r.yy() * (
            r.xy() * cxy + r.yy() * cyy
        )
        gl[3] = r.xx() * (r.xz() * cxx + r.yz() * cxy) + r.yx() * (
            r.xz() * cxy + r.yz() * cyy
        )
        gl[4] = r.xy() * (r.xz() * cxx + r.yz() * cxy) + r.yy() * (
            r.xz() * cxy + r.yz() * cyy
        )
        gl[5] = r.xz() * (r.xz() * cxx + r.yz() * cxy) + r.yz() * (
            r.xz() * cxy + r.yz() * cyy
        )

    @always_inline
    def toLocal(
        self,
        ge: Span[Scalar[Self.T], _],
        mut lxx: Scalar[Self.T],
        mut lxy: Scalar[Self.T],
        mut lyy: Scalar[Self.T],
    ):
        var r = self.rot
        var cxx = ge[0]
        var cyx = ge[1]
        var cyy = ge[2]
        var czx = ge[3]
        var czy = ge[4]
        var czz = ge[5]

        lxx = (
            r.xx() * (r.xx() * cxx + r.xy() * cyx + r.xz() * czx)
            + r.xy() * (r.xx() * cyx + r.xy() * cyy + r.xz() * czy)
            + r.xz() * (r.xx() * czx + r.xy() * czy + r.xz() * czz)
        )

        lxy = (
            r.yx() * (r.xx() * cxx + r.xy() * cyx + r.xz() * czx)
            + r.yy() * (r.xx() * cyx + r.xy() * cyy + r.xz() * czy)
            + r.yz() * (r.xx() * czx + r.xy() * czy + r.xz() * czz)
        )

        lyy = (
            r.yx() * (r.yx() * cxx + r.yy() * cyx + r.yz() * czx)
            + r.yy() * (r.yx() * cyx + r.yy() * cyy + r.yz() * czy)
            + r.yz() * (r.yx() * czx + r.yy() * czy + r.yz() * czz)
        )

    @always_inline
    def x(self) -> Scalar[Self.T]:
        return self.px

    @always_inline
    def y(self) -> Scalar[Self.T]:
        return self.py

    @always_inline
    def z(self) -> Scalar[Self.T]:
        return self.pz

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SOAFrame[" + String(Self.T) + "]"
