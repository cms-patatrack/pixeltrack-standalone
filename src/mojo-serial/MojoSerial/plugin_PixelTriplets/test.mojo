import std.math as math
from std.sys import argv
from std.sys.terminate import exit

from MojoSerial.MojoBridge.Matrix import Matrix, MatrixLike
from MojoSerial.plugin_PixelTriplets.FitUtils import Rfit
from MojoSerial.plugin_PixelTriplets.RiemannFit import Circle_fit, Fast_fit, Line_fit


struct FitResult(Copyable, ImplicitlyCopyable):
    var fast_fit: Rfit.Vector4d
    var circle: Rfit.circle_fit
    var line: Rfit.line_fit

    def __init__(out self):
        self.fast_fit = Rfit.Vector4d()
        self.circle = Rfit.circle_fit()
        self.line = Rfit.line_fit()

    def __init__(out self, *, copy: Self):
        self.fast_fit = copy.fast_fit
        self.circle = copy.circle
        self.line = copy.line


struct ExpectedResult(Copyable, ImplicitlyCopyable):
    var fast_fit: Rfit.Vector4d
    var circle_par: Rfit.Vector3d
    var circle_cov: Rfit.Matrix3d
    var circle_chi2: Float32
    var line_par: Rfit.Vector2d
    var line_cov: Rfit.Matrix2d
    var line_chi2: Float64
    var valid: Bool

    def __init__(out self):
        self.fast_fit = Rfit.Vector4d()
        self.circle_par = Rfit.Vector3d()
        self.circle_cov = Rfit.Matrix3d()
        self.circle_chi2 = 0.0
        self.line_par = Rfit.Vector2d()
        self.line_cov = Rfit.Matrix2d()
        self.line_chi2 = 0.0
        self.valid = False

    def __init__(out self, *, copy: Self):
        self.fast_fit = copy.fast_fit
        self.circle_par = copy.circle_par
        self.circle_cov = copy.circle_cov
        self.circle_chi2 = copy.circle_chi2
        self.line_par = copy.line_par
        self.line_cov = copy.line_cov
        self.line_chi2 = copy.line_chi2
        self.valid = copy.valid


def fill_hits_and_hitscov[N: Int](
    mut hits: Rfit.Matrix3xNd[N],
    mut hits_ge: Matrix[DType.float32, 6, N],
):
    comptime if N == 5:
        var xs: InlineArray[Float64, 5] = [
            2.934787,
            6.314229,
            8.936963,
            10.360559,
            12.856387,
        ]
        var ys: InlineArray[Float64, 5] = [
            0.773211,
            1.816356,
            2.765734,
            3.330824,
            4.422212,
        ]
        var zs: InlineArray[Float64, 5] = [
            -10.980247,
            -23.162731,
            -32.759060,
            -38.061260,
            -47.518867,
        ]
        for i in range(5):
            hits[0, i] = xs[i]
            hits[1, i] = ys[i]
            hits[2, i] = zs[i]

        hits_ge[0, 0] = Float32(1.424715e-07)
        hits_ge[1, 0] = Float32(-4.996975e-07)
        hits_ge[2, 0] = Float32(1.752614e-06)
        hits_ge[3, 0] = Float32(3.660689e-11)
        hits_ge[4, 0] = Float32(1.644638e-09)
        hits_ge[5, 0] = Float32(7.346080e-05)
        hits_ge[0, 1] = Float32(6.899177e-08)
        hits_ge[1, 1] = Float32(-1.873414e-07)
        hits_ge[2, 1] = Float32(5.087101e-07)
        hits_ge[3, 1] = Float32(-2.078806e-10)
        hits_ge[4, 1] = Float32(-2.210498e-11)
        hits_ge[5, 1] = Float32(4.346079e-06)
        hits_ge[0, 2] = Float32(1.406273e-06)
        hits_ge[1, 2] = Float32(4.042467e-07)
        hits_ge[2, 2] = Float32(6.391180e-07)
        hits_ge[3, 2] = Float32(-3.141497e-07)
        hits_ge[4, 2] = Float32(6.513821e-08)
        hits_ge[5, 2] = Float32(1.163863e-07)
        hits_ge[0, 3] = Float32(1.176358e-06)
        hits_ge[1, 3] = Float32(2.154100e-07)
        hits_ge[2, 3] = Float32(5.072816e-07)
        hits_ge[3, 3] = Float32(-8.161219e-08)
        hits_ge[4, 3] = Float32(1.437878e-07)
        hits_ge[5, 3] = Float32(5.951832e-08)
        hits_ge[0, 4] = Float32(2.852843e-05)
        hits_ge[1, 4] = Float32(7.956492e-06)
        hits_ge[2, 4] = Float32(3.117701e-06)
        hits_ge[3, 4] = Float32(-1.060541e-06)
        hits_ge[4, 4] = Float32(8.777413e-09)
        hits_ge[5, 4] = Float32(1.426417e-07)
        return

    comptime if N > 3:
        var xs: InlineArray[Float64, 4] = [1.98645, 4.72598, 7.65632, 11.3151]
        var ys: InlineArray[Float64, 4] = [2.18002, 4.88864, 7.75845, 11.3134]
        var zs: InlineArray[Float64, 4] = [2.46338, 6.99838, 11.808, 17.793]
        for i in range(4):
            hits[0, i] = xs[i]
            hits[1, i] = ys[i]
            hits[2, i] = zs[i]
    else:
        var xs: InlineArray[Float64, 3] = [1.98645, 4.72598, 7.65632]
        var ys: InlineArray[Float64, 3] = [2.18002, 4.88864, 7.75845]
        var zs: InlineArray[Float64, 3] = [2.46338, 6.99838, 11.808]
        for i in range(3):
            hits[0, i] = xs[i]
            hits[1, i] = ys[i]
            hits[2, i] = zs[i]

    hits_ge[0, 0] = Float32(7.14652e-06)
    hits_ge[0, 1] = Float32(2.15789e-06)
    hits_ge[0, 2] = Float32(1.63328e-06)
    comptime if N > 3:
        hits_ge[0, 3] = Float32(6.27919e-06)

    hits_ge[2, 0] = Float32(6.10348e-06)
    hits_ge[2, 1] = Float32(2.08211e-06)
    hits_ge[2, 2] = Float32(1.61672e-06)
    comptime if N > 3:
        hits_ge[2, 3] = Float32(6.28081e-06)

    hits_ge[5, 0] = Float32(5.184e-05)
    hits_ge[5, 1] = Float32(1.444e-05)
    hits_ge[5, 2] = Float32(6.25e-06)
    comptime if N > 3:
        hits_ge[5, 3] = Float32(3.136e-05)

    hits_ge[1, 0] = Float32(-5.60077e-06)
    hits_ge[1, 1] = Float32(-1.11936e-06)
    hits_ge[1, 2] = Float32(-6.24945e-07)
    comptime if N > 3:
        hits_ge[1, 3] = Float32(-5.28e-06)


def compute_rad[N: Int](hits: Rfit.Matrix3xNd[N]) -> Rfit.VectorNd[N]:
    var rad = Rfit.VectorNd[N]()
    for i in range(N):
        var x = hits[0, i]
        var y = hits[1, i]
        rad[i] = math.sqrt(x * x + y * y)
    return rad


def run_fit[N: Int]() -> FitResult:
    var B: Float64 = 0.0113921
    var hits = Rfit.Matrix3xNd[N]()
    var hits_ge = Matrix[DType.float32, 6, N]()
    fill_hits_and_hitscov[N](hits, hits_ge)

    var fast_fit = Rfit.Vector4d()
    Fast_fit(hits, fast_fit)

    var hits_cov = Rfit.Matrix2Nd[N].Zero()
    Rfit.loadCovariance2D(hits_ge, hits_cov)

    var hits2D = Rfit.Matrix2xNd[N].Zero()
    for i in range(N):
        hits2D[0, i] = hits[0, i]
        hits2D[1, i] = hits[1, i]

    var rad = compute_rad[N](hits)
    var circle = Circle_fit(hits2D, hits_cov, fast_fit, rad, B, True)
    var line = Line_fit(hits, hits_ge, circle, fast_fit, B, True)
    Rfit.par_uvrtopak(circle, B, True)

    var res = FitResult()
    res.fast_fit = fast_fit
    res.circle = circle
    res.line = line
    return res


# Fill expected values from C++ testRiemannFit output and set valid = True.
def expected_for[N: Int]() -> ExpectedResult:
    var expected = ExpectedResult()
    expected.valid = False
    comptime if N == 4:
        return expected
    comptime if N == 3:
        return expected
    comptime if N == 5:
        return expected
    return expected


def near(a: Float64, b: Float64, rtol: Float64, atol: Float64) -> Bool:
    var diff = abs(a - b)
    var scale = max(abs(a), abs(b))
    return diff <= max(atol, rtol * scale)


def check_matrix[A: MatrixLike, B: MatrixLike](
    name: StringLiteral,
    a: A,
    b: B,
    rtol: Float64,
    atol: Float64,
) -> Bool:
    comptime a_cols = A.ColsAtCompileTime()
    comptime b_cols = B.ColsAtCompileTime()
    if a.num_rows() != b.num_rows() or a_cols != b_cols:
        print(name, " shape mismatch: ", a.num_rows(), "x", a_cols, " vs ", b.num_rows(), "x", b_cols)
        return False

    var ok = True
    for r in range(a.num_rows()):
        for c in range(a_cols):
            var av = Float64(a[r, c])
            var bv = Float64(b[r, c])
            if not near(av, bv, rtol, atol):
                print(name, "[", r, ",", c, "] ", av, " != ", bv)
                ok = False
    return ok


def compare_fit[N: Int](res: FitResult, expected: ExpectedResult) -> Bool:
    if not expected.valid:
        print("Expected values not set for N=", N)
        return False

    var rtol: Float64 = 1e-6
    var atol: Float64 = 1e-8
    var ok = True
    ok = check_matrix("fast_fit", res.fast_fit, expected.fast_fit, rtol, atol) and ok
    ok = check_matrix("circle_par", res.circle.par, expected.circle_par, rtol, atol) and ok
    ok = check_matrix("circle_cov", res.circle.cov, expected.circle_cov, rtol, atol) and ok
    if not near(Float64(res.circle.chi2), Float64(expected.circle_chi2), rtol, atol):
        print("circle_chi2 ", res.circle.chi2, " != ", expected.circle_chi2)
        ok = False
    ok = check_matrix("line_par", res.line.par, expected.line_par, rtol, atol) and ok
    ok = check_matrix("line_cov", res.line.cov, expected.line_cov, rtol, atol) and ok
    if not near(Float64(res.line.chi2), Float64(expected.line_chi2), rtol, atol):
        print("line_chi2 ", res.line.chi2, " != ", expected.line_chi2)
        ok = False
    return ok


def dump_fit[N: Int](res: FitResult):
    print("N=", N)
    print("fast_fit=", res.fast_fit)
    print("circle_par=", res.circle.par)
    print("circle_cov=", res.circle.cov)
    print("circle_chi2=", res.circle.chi2)
    print("line_par=", res.line.par)
    print("line_cov=", res.line.cov)
    print("line_chi2=", res.line.chi2)


def main() raises:
    var compare = False
    var args = argv()
    var i = 1
    while i < args.__len__():
        if args[i] == "--compare":
            compare = True
        elif args[i] == "--dump":
            compare = False
        else:
            print("Unknown arg: ", args[i])
            exit(1)
        i += 1

    if compare:
        var ok = True
        var res4 = run_fit[4]()
        ok = compare_fit[4](res4, expected_for[4]()) and ok
        var res3 = run_fit[3]()
        ok = compare_fit[3](res3, expected_for[3]()) and ok
        var res5 = run_fit[5]()
        ok = compare_fit[5](res5, expected_for[5]()) and ok
        if not ok:
            exit(1)
        print("All comparisons passed")
    else:
        dump_fit[4](run_fit[4]())
        dump_fit[3](run_fit[3]())
        dump_fit[5](run_fit[5]())
