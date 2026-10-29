from MojoSerial.MojoBridge.DTypes import Float, Double, Typeable
from MojoSerial.MojoBridge.Vector import Vector
from MojoSerial.MojoBridge.Matrix import Matrix

from MojoSerial.CUDACore.EigenSoA import MatrixSoA
from MojoSerial.CUDADataFormats.TrajectoryStateSoA import TrajectoryStateSoA

from std.math import sqrt

comptime Vector5d = Vector[DType.float64, 5]
comptime Matrix5d = Matrix[DType.float64, 5, 5]


def loadCov(e: Vector5d) -> Matrix5d:
    var cov: Matrix5d = Matrix5d()
    for i in range(5):
        cov[i, i] = e[i] * e[i]
    for i in range(5):
        for j in range(i):
            var v: Float64 = 0.3 * sqrt(cov[i, i] * cov[j, j])
            cov[i, j] = -0.4 * v if (i + j) % 2 else 0.1 * v
            cov[j, i] = cov[i, j]
    return cov


comptime TS = TrajectoryStateSoA[128]


def testTSSoA(mut ts: TS, n: Int64):
    debug_assert(n <= 128)

    var par0: Vector5d = Vector5d(0.2, 0.1, 3.5, 0.8, 0.1)
    var e0: Vector5d = Vector5d(0.01, 0.01, 0.035, -0.03, -0.01)
    var cov0 = loadCov(e0)

    for i in range(n):
        ts.copyFromDense(par0, cov0, i.cast[DType.int32]())
        var par1: Vector5d = Vector5d()
        var cov1: Matrix5d = Matrix5d()
        ts.copyToDense(par1, cov1, i.cast[DType.int32]())
        var delV: Vector5d = par1 - par0
        var delM: Matrix5d = cov1 - cov0
        for j in range(5):
            debug_assert(abs(delV[j]) < 1e-5)
            for k in range(j, 5):
                debug_assert(cov0[k, j] == cov0[j, k])
                debug_assert(cov1[k, j] == cov1[j, k])
                debug_assert(abs(delM[k, j]) < 1.0e-5)


def main():
    var ts: TS = TS()
    testTSSoA(ts, 128)
