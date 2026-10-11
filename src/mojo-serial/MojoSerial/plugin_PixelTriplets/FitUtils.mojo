import std.math as math
from MojoSerial.MojoBridge.Matrix import Matrix, MatrixLike, MatrixXd, VectorXd
from MojoSerial.MojoBridge.DTypes import DType

@fieldwise_init
struct _Rfit_circle_fit(Copyable, Defaultable, ImplicitlyCopyable, Movable):
    var par: Matrix[DType.float64, 3, 1]  # (X0, Y0, R)
    var cov: Matrix[DType.float64, 3, 3]
    var q: Int32  # particle charge
    var chi2: Float32

    def __init__(out self):
        self.par = Matrix[DType.float64, 3, 1](uninitialized=True)
        self.cov = Matrix[DType.float64, 3, 3](uninitialized=True)
        self.q = 0
        self.chi2 = 0.0


@fieldwise_init
struct _Rfit_line_fit(Copyable, Defaultable, ImplicitlyCopyable, Movable):
    var par: Matrix[DType.float64, 2, 1]  # (cotan(theta), Zip)
    var cov: Matrix[DType.float64, 2, 2]
    var chi2: Float64

    def __init__(out self):
        self.par = Matrix[DType.float64, 2, 1](uninitialized=True)
        self.cov = Matrix[DType.float64, 2, 2](uninitialized=True)
        self.chi2 = 0.0


@fieldwise_init
struct _Rfit_helix_fit(Copyable, Defaultable, ImplicitlyCopyable, Movable):
    var par: Matrix[DType.float64, 5, 1]  # (phi, Tip, pt, cotan(theta), Zip)
    var cov: Matrix[DType.float64, 5, 5]
    var chi2_circle: Float32
    var chi2_line: Float32
    var q: Int32  # particle charge

    def __init__(out self):
        self.par = Matrix[DType.float64, 5, 1](uninitialized=True)
        self.cov = Matrix[DType.float64, 5, 5](uninitialized=True)
        self.chi2_circle = 0.0
        self.chi2_line = 0.0
        self.q = 0

struct Rfit:

    comptime d: Float64 = 1e-4

    comptime Vector2d = Matrix[DType.float64, 2, 1]
    comptime Vector3d = Matrix[DType.float64, 3, 1]
    comptime Vector4d = Matrix[DType.float64, 4, 1]
    comptime Vector5d = Matrix[DType.float64, 5, 1]
    comptime Vector6f = Matrix[DType.float64, 6, 1]

    comptime Matrix2d = Matrix[DType.float64, 2, 2]
    comptime Matrix3d = Matrix[DType.float64, 3, 3]
    comptime Matrix4d = Matrix[DType.float64, 4, 4]
    comptime Matrix5d = Matrix[DType.float64, 5, 5]
    comptime Matrix6d = Matrix[DType.float64, 6, 6]

    comptime Matrix2x3d = Matrix[DType.float64, 2, 3]
    comptime Matrix3f = Matrix[DType.float64, 3, 3]
    comptime Vector3f = Matrix[DType.float64, 3, 1]
    comptime Vector4f = Matrix[DType.float64, 4, 1]
    
    comptime MatrixXd = MatrixXd
    comptime VectorXd = VectorXd

    comptime MatrixNd[N: Int] = Matrix[DType.float64, N, N]
    comptime MatrixNplusONEd[N: Int] = Matrix[DType.float64, N + 1, N + 1]
    comptime ArrayNd[N: Int] = Matrix[DType.float64, N, N]
    comptime Matrix2Nd[N: Int] = Matrix[DType.float64, 2 * N, 2 * N]
    comptime Matrix3Nd[N: Int] = Matrix[DType.float64, 3 * N, 3 * N]
    comptime Matrix2xNd[N: Int] = Matrix[DType.float64, 2, N]
    comptime Matrix3xNd[N: Int] = Matrix[DType.float64, 3, N]
    comptime Array2xNd[N: Int] = Matrix[DType.float64, 2, N]
    comptime MatrixNx3d[N: Int] = Matrix[DType.float64, N, 3]
    comptime MatrixNx5d[N: Int] = Matrix[DType.float64, N, 5]
    comptime VectorNd[N: Int] = Matrix[DType.float64, N, 1]
    comptime VectorNplusONEd[N: Int] = Matrix[DType.float64, N + 1, 1]
    comptime Vector2Nd[N: Int] = Matrix[DType.float64, 2 * N, 1]
    comptime Vector3Nd[N: Int] = Matrix[DType.float64, 3 * N, 1]
    comptime RowVectorNd[N: Int] = Matrix[DType.float64, 1, 1]
    comptime RowVector2Nd[N: Int] = Matrix[DType.float64, 1, 2 * N]

    comptime circle_fit = _Rfit_circle_fit
    comptime line_fit = _Rfit_line_fit
    comptime helix_fit = _Rfit_helix_fit

    comptime u_int = UInt32

    # C++ takes `M const*`; a borrow is the direct equivalent and needs no
    # wrapper at the 56 call sites.
    @staticmethod
    def printIt[M: MatrixLike, RFIT_DEBUG: Bool = False](m: M, prefix: String = ""):
        comptime if RFIT_DEBUG:
            var r: Int = 0
            while r < m.num_rows():
                var c: Int = 0
                while c < M.ColsAtCompileTime():
                    print(prefix, "Matrix(", r, ",", c, ") =", m[r, c])
                    c += 1
                r += 1

    @staticmethod
    def sqr(a: Float64) -> Float64:
        return a * a

    @staticmethod
    def cross2D(a: Self.Vector2d, b: Self.Vector2d) -> Float64:
        return a[0] * b[1] - a[1] * b[0]

    @staticmethod
    def loadCovariance2D[M6xN: MatrixLike, N: Int](ge: M6xN, mut hits_cov: Matrix[DType.float64, 2 * N, 2 * N]):
        var hits_in_fit: Int = N
        var i: Int = 0
        while i < hits_in_fit:
            var ge_idx = 0
            var j = 0
            var l = 0
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            ge_idx = 2
            j = 1
            l = 1
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            ge_idx = 1
            j = 1
            l = 0
            hits_cov[i + l * hits_in_fit, i + j * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            i += 1

    @staticmethod
    def loadCovariance[M6xN: MatrixLike, N: Int](ge: M6xN, mut hits_cov: Matrix[DType.float64, 3 * N, 3 * N]):
        var hits_in_fit: Int = N
        var i: Int = 0
        while i < hits_in_fit:
            var ge_idx = 0
            var j = 0
            var l = 0
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            ge_idx = 2
            j = 1
            l = 1
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            ge_idx = 5
            j = 2
            l = 2
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            ge_idx = 1
            j = 1
            l = 0
            hits_cov[i + l * hits_in_fit, i + j * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            ge_idx = 3
            j = 2
            l = 0
            hits_cov[i + l * hits_in_fit, i + j * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            ge_idx = 4
            j = 2
            l = 1
            hits_cov[i + l * hits_in_fit, i + j * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            hits_cov[i + j * hits_in_fit, i + l * hits_in_fit] = ge.col(i)[ge_idx].cast[DType.float64]()
            i += 1

    @staticmethod
    def par_uvrtopak(mut circle: Self.circle_fit, B: Float64, error: Bool):
        var par_pak = Self.Vector3d(uninitialized=True)
        var temp0 = circle.par.head[2]().squaredNorm()
        var temp1 = math.sqrt(temp0)
        par_pak[0] = math.atan2(Float64(circle.q) * circle.par[0], -Float64(circle.q) * circle.par[1])
        par_pak[1] = Float64(circle.q) * (temp1 - circle.par[2])
        par_pak[2] = circle.par[2] * B
        if error:
            var temp2 = Self.sqr(circle.par[0]) * 1.0 / temp0
            var temp3 = 1.0 / temp1 * Float64(circle.q)
            var J4 = Self.Matrix3d(uninitialized=True)
            J4[0, 0] = -circle.par[1] * temp2 * 1.0 / Self.sqr(circle.par[0])
            J4[0, 1] = temp2 * 1.0 / circle.par[0]
            J4[0, 2] = 0.0
            J4[1, 0] = circle.par[0] * temp3
            J4[1, 1] = circle.par[1] * temp3
            J4[1, 2] = -Float64(circle.q)
            J4[2, 0] = 0.0
            J4[2, 1] = 0.0
            J4[2, 2] = B
            circle.cov = J4 * circle.cov * J4.transpose()
        circle.par = par_pak

    @staticmethod
    def fromCircleToPerigee(mut circle: Self.circle_fit):
        var par_pak = Self.Vector3d(uninitialized=True)
        var temp0 = circle.par.head[2]().squaredNorm()
        var temp1 = math.sqrt(temp0)
        par_pak[0] = math.atan2(Float64(circle.q) * circle.par[0], -Float64(circle.q) * circle.par[1])
        par_pak[1] = Float64(circle.q) * (temp1 - circle.par[2])
        par_pak[2] = Float64(circle.q) / circle.par[2]

        var temp2 = Self.sqr(circle.par[0]) * 1.0 / temp0
        var temp3 = 1.0 / temp1 * Float64(circle.q)
        var J4 = Self.Matrix3d(uninitialized=True)
        J4[0, 0] = -circle.par[1] * temp2 * 1.0 / Self.sqr(circle.par[0])
        J4[0, 1] = temp2 * 1.0 / circle.par[0]
        J4[0, 2] = 0.0
        J4[1, 0] = circle.par[0] * temp3
        J4[1, 1] = circle.par[1] * temp3
        J4[1, 2] = -Float64(circle.q)
        J4[2, 0] = 0.0
        J4[2, 1] = 0.0
        J4[2, 2] = -Float64(circle.q) / (circle.par[2] * circle.par[2])
        circle.cov = J4 * circle.cov * J4.transpose()

        circle.par = par_pak

    @staticmethod
    def transformToPerigeePlane(
        ip: Self.Vector5d,
        icov: Self.Matrix5d,
        mut op: Self.Vector5d,
        mut ocov: Self.Matrix5d,
    ):
        var sinTheta2 = 1.0 / (1.0 + ip[3] * ip[3])
        var sinTheta = math.sqrt(sinTheta2)
        var cosTheta = ip[3] * sinTheta

        op[0] = sinTheta * ip[2]
        op[1] = 0.0
        op[2] = -ip[3]
        op[3] = ip[1]
        op[4] = -ip[4]

        var J = Self.Matrix5d.Zero()

        J[0, 2] = sinTheta
        J[0, 3] = -sinTheta2 * cosTheta * ip[2]
        J[1, 0] = 1.0
        J[2, 3] = -1.0
        J[3, 1] = 1.0
        J[4, 4] = -1.0

        ocov = J * icov * J.transpose()
