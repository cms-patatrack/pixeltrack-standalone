# Translated from pixeltrack-standalone/src/serial/plugin-PixelTriplets/BrokenLine.h

import std.math as math
from MojoSerial.MojoBridge.Matrix import Matrix, MatrixLike
from MojoSerial.plugin_PixelTriplets.FitUtils import Rfit
import MojoSerial.plugin_PixelTriplets.choleskyInversion as choleskyInversion

comptime CPP_DUMP = False

#< Karimäki's parameters: (phi, d, k=1/R)
#< covariance matrix: \n
#  |cov(phi,phi)|cov( d ,phi)|cov( k ,phi)| \n
#  |cov(phi, d )|cov( d , d )|cov( k , d )| \n
#  |cov(phi, k )|cov( d , k )|cov( k , k )|
comptime karimaki_circle_fit = Rfit.circle_fit

#!
#\brief data needed for the Broken Line fit procedure.
struct PreparedBrokenLineData[N: Int]:
    var q: Int                     #!< particle charge
    var radii: Rfit.Matrix2xNd[Self.N]  #!< xy data in the system in which the pre-fitted center is the origin
    var s: Rfit.VectorNd[Self.N]        #!< total distance traveled in the transverse plane
                                   #   starting from the pre-fitted closest approach
    var S: Rfit.VectorNd[Self.N]        #!< total distance traveled (three-dimensional)
    var Z: Rfit.VectorNd[Self.N]        #!< orthogonal coordinate to the pre-fitted line in the sz plane
    var VarBeta: Rfit.VectorNd[Self.N]  #!< kink angles in the SZ plane

    def __init__(out self):
        self.q = 0
        self.radii = Rfit.Matrix2xNd[Self.N](uninitialized=True)
        self.s = Rfit.VectorNd[Self.N](uninitialized=True)
        self.S = Rfit.VectorNd[Self.N](uninitialized=True)
        self.Z = Rfit.VectorNd[Self.N](uninitialized=True)
        self.VarBeta = Rfit.VectorNd[Self.N](uninitialized=True)


#!
#\brief Computes the Coulomb multiple scattering variance of the planar angle.
#
#\param length length of the track in the material.
#\param B magnetic field in Gev/cm/c.
#\param R radius of curvature (needed to evaluate p).
#\param Layer denotes which of the four layers of the detector is the endpoint of the multiple scattered track. For example, if Layer=3, then the particle has just gone through the material between the second and the third layer.
#
#\todo add another Layer variable to identify also the start point of the track, so if there are missing hits or multiple hits, the part of the detector that the particle has traversed can be exactly identified.
#
#\warning the formula used here assumes beta=1, and so neglects the dependence of theta_0 on the mass of the particle at fixed momentum.
#
#\return the variance of the planar angle ((theta_0)^2 /3).
def MultScatt(length: Float64, B: Float64, R: Float64, Layer: Int, slope: Float64) -> Float64:
    # limit R to 20GeV...
    var _ = Layer
    var pt2 = min(20.0, B * R)
    pt2 *= pt2
    var XXI_0 = 0.06 / 16.0  #!< inverse of radiation length of the material in cm
    var geometry_factor = 0.7  #!< number between 1/3 (uniform material) and 1 (thin scatterer) to be manually tuned
    var fact = geometry_factor * Rfit.sqr(13.6 / 1000.0)
    return fact / (pt2 * (1.0 + Rfit.sqr(slope))) * (abs(length) * XXI_0) * Rfit.sqr(1.0 + 0.038 * math.log(abs(length) * XXI_0))


#!
#\brief Computes the 2D rotation matrix that transforms the line y=slope*x into the line y=0.
#
#\param slope tangent of the angle of rotation.
#
#\return 2D rotation matrix.
def RotationMatrix(slope: Float64) -> Rfit.Matrix2d:
    var Rot = Rfit.Matrix2d(uninitialized=True)
    Rot[0, 0] = 1.0 / math.sqrt(1.0 + Rfit.sqr(slope))
    Rot[0, 1] = slope * Rot[0, 0]
    Rot[1, 0] = -Rot[0, 1]
    Rot[1, 1] = Rot[0, 0]
    return Rot


#!
#\brief Changes the Karimäki parameters (and consequently their covariance matrix) under a translation of the coordinate system, such that the old origin has coordinates (x0,y0) in the new coordinate system. The formulas are taken from Karimäki V., 1990, Effective circle fitting for particle trajectories, Nucl. Instr. and Meth. A305 (1991) 187.
#
#\param circle circle fit in the old coordinate system.
#\param x0 x coordinate of the translation vector.
#\param y0 y coordinate of the translation vector.
#\param jacobian passed by reference in order to save stack.
def TranslateKarimaki(mut circle: karimaki_circle_fit, x0: Float64, y0: Float64, mut jacobian: Rfit.Matrix3d):
    var A: Float64 = 0.0
    var U: Float64 = 0.0
    var BB: Float64 = 0.0
    var C: Float64 = 0.0
    var DO: Float64 = 0.0
    var DP: Float64 = 0.0
    var uu: Float64 = 0.0
    var xi: Float64 = 0.0
    var v: Float64 = 0.0
    var mu: Float64 = 0.0
    var lambda_: Float64 = 0.0
    var zeta: Float64 = 0.0
    DP = x0 * math.cos(circle.par[0]) + y0 * math.sin(circle.par[0])
    DO = x0 * math.sin(circle.par[0]) - y0 * math.cos(circle.par[0]) + circle.par[1]
    uu = 1 + circle.par[2] * circle.par[1]
    C = -circle.par[2] * y0 + uu * math.cos(circle.par[0])
    BB = circle.par[2] * x0 + uu * math.sin(circle.par[0])
    A = 2.0 * DO + circle.par[2] * (Rfit.sqr(DO) + Rfit.sqr(DP))
    U = math.sqrt(1.0 + circle.par[2] * A)
    xi = 1.0 / (Rfit.sqr(BB) + Rfit.sqr(C))
    v = 1.0 + circle.par[2] * DO
    lambda_ = (0.5 * A) / (U * Rfit.sqr(1.0 + U))
    mu = 1.0 / (U * (1.0 + U)) + circle.par[2] * lambda_
    zeta = Rfit.sqr(DO) + Rfit.sqr(DP)

    jacobian[0, 0] = xi * uu * v
    jacobian[0, 1] = -xi * Rfit.sqr(circle.par[2]) * DP
    jacobian[0, 2] = xi * DP
    jacobian[1, 0] = 2.0 * mu * uu * DP
    jacobian[1, 1] = 2.0 * mu * v
    jacobian[1, 2] = mu * zeta - lambda_ * A
    jacobian[2, 0] = 0.0
    jacobian[2, 1] = 0.0
    jacobian[2, 2] = 1.0

    circle.par[0] = math.atan2(BB, C)
    circle.par[1] = A / (1 + U)

    circle.cov = jacobian * circle.cov * jacobian.transpose()


#!
#\brief Computes the data needed for the Broken Line fit procedure that are mainly common for the circle and the line fit.
#
#\param hits hits coordinates.
#\param hits_cov hits covariance matrix.
#\param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
#\param B magnetic field in Gev/cm/c.
#\param results PreparedBrokenLineData to be filled (see description of PreparedBrokenLineData).
def prepareBrokenLineData[M3xN: MatrixLike, V4: MatrixLike, N: Int](
    hits: M3xN, fast_fit: V4, B: Float64, mut results: PreparedBrokenLineData[N]
):
    var n = N
    var i: Int = 0
    var d = Rfit.Vector2d(uninitialized=True)
    var e = Rfit.Vector2d(uninitialized=True)

    d = (hits.block[2, 1](0, 1) - hits.block[2, 1](0, 0)).cast[DType.float64]()
    e = (hits.block[2, 1](0, n - 1) - hits.block[2, 1](0, n - 2)).cast[DType.float64]()
    if Rfit.cross2D(d, e) > 0:
        results.q = -1
    else:
        results.q = 1

    var slope: Float64 = -Float64(results.q) / fast_fit[3, 0].cast[DType.float64]()

    var R = RotationMatrix(slope)

    results.radii = hits.block[2, N](0, 0).cast[DType.float64]() - fast_fit.head[2]().cast[DType.float64]() * Matrix[DType.float64, 1, N].Constant(1.0)
    e = -fast_fit[2, 0].cast[DType.float64]() * fast_fit.head[2]().cast[DType.float64]() / fast_fit.head[2]().cast[DType.float64]().norm()
    i = 0
    while i < n:
        d = results.radii.block[2, 1](0, i)
        results.s[i] = Float64(results.q) * fast_fit[2, 0].cast[DType.float64]() * math.atan2(Rfit.cross2D(d, e), d.dot(e))
        i += 1
    var z : Rfit.VectorNd[N] = hits.block[1, N](2, 0).transpose().cast[DType.float64]()

    var pointsSZ: Rfit.Matrix2xNd[N] = Rfit.Matrix2xNd[N].Zero()
    i = 0
    while i < n:
        pointsSZ[0, i] = results.s[i]
        pointsSZ[1, i] = z[i]
        pointsSZ.set_block[2, 1](0, i, R * pointsSZ.block[2, 1](0, i))
        i += 1
    results.S = pointsSZ.block[1, N](0, 0).transpose()
    results.Z = pointsSZ.block[1, N](1, 0).transpose()

    results.VarBeta[0] = 0
    results.VarBeta[n - 1] = 0
    i = 1
    while i < n - 1:
        results.VarBeta[i] = MultScatt(results.S[i + 1] - results.S[i], B, fast_fit[2, 0].cast[DType.float64](), i + 2, slope) + MultScatt(results.S[i] - results.S[i - 1], B, fast_fit[2, 0].cast[DType.float64](), i + 1, slope)
        i += 1


#!
#\brief Computes the n-by-n band matrix obtained minimizing the Broken Line's cost function w.r.t u. This is the whole matrix in the case of the line fit and the main n-by-n block in the case of the circle fit.
#
#\param w weights of the first part of the cost function, the one with the measurements and not the angles (\sum_{i=1}^n w*(y_i-u_i)^2).
#\param S total distance traveled by the particle from the pre-fitted closest approach.
#\param VarBeta kink angles' variance.
#
#\return the n-by-n matrix of the linear system
def MatrixC_u[N: Int](w: Rfit.VectorNd[N], S: Rfit.VectorNd[N], VarBeta: Rfit.VectorNd[N]) -> Rfit.MatrixNd[N]:
    var n: Int = N
    var i: Int = 0

    var C_U = Rfit.MatrixNd[N].Zero()
    i = 0
    while i < n:
        C_U[i, i] = w[i]
        if i > 1:
            C_U[i, i] += 1.0 / (VarBeta[i - 1] * Rfit.sqr(S[i] - S[i - 1]))
        if i > 0 and i < n - 1:
            C_U[i, i] += (1.0 / VarBeta[i]) * Rfit.sqr((S[i + 1] - S[i - 1]) / ((S[i + 1] - S[i]) * (S[i] - S[i - 1])))
        if i < n - 2:
            C_U[i, i] += 1.0 / (VarBeta[i + 1] * Rfit.sqr(S[i + 1] - S[i]))

        if i > 0 and i < n - 1:
            C_U[i, i + 1] = 1.0 / (VarBeta[i] * (S[i + 1] - S[i])) * (-(S[i + 1] - S[i - 1]) / ((S[i + 1] - S[i]) * (S[i] - S[i - 1])))
        if i < n - 2:
            C_U[i, i + 1] += 1.0 / (VarBeta[i + 1] * (S[i + 1] - S[i])) * (-(S[i + 2] - S[i]) / ((S[i + 2] - S[i + 1]) * (S[i + 1] - S[i])))

        if i < n - 2:
            C_U[i, i + 2] = 1.0 / (VarBeta[i + 1] * (S[i + 2] - S[i + 1]) * (S[i + 1] - S[i]))

        C_U[i, i] *= 0.5
        i += 1
    return C_U + C_U.transpose()


#!
#\brief A very fast helix fit.
#
#\param hits the measured hits.
#
#\return (X0,Y0,R,tan(theta)).
#
#\warning sign of theta is (intentionally, for now) mistaken for negative charges.
def BL_Fast_fit[M3xN: MatrixLike, V4: MatrixLike](hits: M3xN, mut result: V4):
    var N = M3xN.ColsAtCompileTime()
    var n = N

    var a: Rfit.Vector2d = (hits.block[2, 1](0, n // 2) - hits.block[2, 1](0, 0)).cast[DType.float64]()
    var b: Rfit.Vector2d = (hits.block[2, 1](0, n - 1) - hits.block[2, 1](0, n // 2)).cast[DType.float64]()
    var c: Rfit.Vector2d = (hits.block[2, 1](0, 0) - hits.block[2, 1](0, n - 1)).cast[DType.float64]()

    var tmp = 0.5 / Rfit.cross2D(c, a)
    result[0, 0] = (hits[0, 0].cast[DType.float64]() - (a[1] * c.squaredNorm() + c[1] * a.squaredNorm()) * tmp).cast[V4.ElemType]()
    result[1, 0] = (hits[1, 0].cast[DType.float64]() + (a[0] * c.squaredNorm() + c[0] * a.squaredNorm()) * tmp).cast[V4.ElemType]()

    result[2, 0] = (math.sqrt(a.squaredNorm() * b.squaredNorm() * c.squaredNorm()) / (2.0 * abs(Rfit.cross2D(b, a)))).cast[V4.ElemType]()

    var d : Rfit.Vector2d= hits.block[2, 1](0, 0).cast[DType.float64]() - result.head[2]().cast[DType.float64]()
    var e : Rfit.Vector2d= hits.block[2, 1](0, n - 1).cast[DType.float64]() - result.head[2]().cast[DType.float64]()

    result[3, 0] = (result[2, 0].cast[DType.float64]() * math.atan2(Rfit.cross2D(d, e), d.dot(e)) / (hits[2, n - 1].cast[DType.float64]() - hits[2, 0].cast[DType.float64]())).cast[V4.ElemType]()


#!
#\brief Performs the Broken Line fit in the curved track case (that is, the fit parameters are the interceptions u and the curvature correction \Delta\kappa).
#
#\param hits hits coordinates.
#\param hits_cov hits covariance matrix.
#\param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
#\param B magnetic field in Gev/cm/c.
#\param data PreparedBrokenLineData.
#\param circle_results struct to be filled with the results in this form:
#-par parameter of the line in this form: (phi, d, k); \n
#-cov covariance matrix of the fitted parameter; \n
#-chi2 value of the cost function in the minimum.
#
#\details The function implements the steps 2 and 3 of the Broken Line fit with the curvature correction.\n
#The step 2 is the least square fit, done by imposing the minimum constraint on the cost function and solving the consequent linear system. It determines the fitted parameters u and \Delta\kappa and their covariance matrix.
#The step 3 is the correction of the fast pre-fitted parameters for the innermost part of the track. It is first done in a comfortable coordinate system (the one in which the first hit is the origin) and then the parameters and their covariance matrix are transformed to the original coordinate system.
def BL_Circle_fit[M3xN: MatrixLike, M6xN: MatrixLike, V4: MatrixLike, N: Int](
    hits: M3xN, hits_ge: M6xN, fast_fit: V4, B: Float64, mut data: PreparedBrokenLineData[N], mut circle_results: karimaki_circle_fit
):
    var n: Int = N
    var i: Int = 0

    circle_results.q = Int32(data.q)
    ref radii = data.radii
    ref s = data.s
    ref S = data.S
    ref Z = data.Z
    ref VarBeta = data.VarBeta
    var slope: Float64 = -Float64(circle_results.q) / fast_fit[3, 0].cast[DType.float64]()
    VarBeta *= 1.0 + Rfit.sqr(slope)

    i = 0
    while i < n:
        Z[i] = radii.block[2, 1](0, i).norm() - fast_fit[2, 0].cast[DType.float64]()
        i += 1

    var V = Rfit.Matrix2d(uninitialized=True)
    var w = Rfit.VectorNd[N](uninitialized=True)
    var RR = Rfit.Matrix2d(uninitialized=True)
    i = 0
    while i < n:
        V[0, 0] = hits_ge[0, i].cast[DType.float64]()
        V[0, 1] = hits_ge[1, i].cast[DType.float64]()
        V[1, 0] = hits_ge[1, i].cast[DType.float64]()
        V[1, 1] = hits_ge[2, i].cast[DType.float64]()
        RR = RotationMatrix(-radii[0, i] / radii[1, i])
        w[i] = 1.0 / ((RR * V * RR.transpose())[1, 1])
        i += 1
    var r_u : Rfit.VectorNplusONEd[N] = Rfit.VectorNplusONEd[N](uninitialized=True)
    r_u[n] = 0
    i = 0
    while i < n:
        r_u[i] = w[i] * Z[i]
        i += 1

    var C_U : Rfit.MatrixNplusONEd[N] = Rfit.MatrixNplusONEd[N](uninitialized=True)
    C_U.set_block[N, N](0, 0, MatrixC_u(w, s, VarBeta))
    C_U[n, n] = 0
    i = 0
    while i < n:
        C_U[i, n] = 0
        if i > 0 and i < n - 1:
            C_U[i, n] += -(s[i + 1] - s[i - 1]) * (s[i + 1] - s[i - 1]) / (2.0 * VarBeta[i] * (s[i + 1] - s[i]) * (s[i] - s[i - 1]))
        if i > 1:
            C_U[i, n] += (s[i] - s[i - 2]) / (2.0 * VarBeta[i - 1] * (s[i] - s[i - 1]))
        if i < n - 2:
            C_U[i, n] += (s[i + 2] - s[i]) / (2.0 * VarBeta[i + 1] * (s[i + 1] - s[i]))
        C_U[n, i] = C_U[i, n]
        if i > 0 and i < n - 1:
            C_U[n, n] += Rfit.sqr(s[i + 1] - s[i - 1]) / (4.0 * VarBeta[i])
        i += 1

    if CPP_DUMP:
        print("CU5")
        var rowsCU = C_U.num_rows()
        var colsCU = C_U.cols()
        var rCU = 0
        while rCU < rowsCU:
            var lineCU = ""
            var cCU = 0
            while cCU < colsCU:
                lineCU = lineCU + String(C_U[rCU, cCU]) + " "
                cCU += 1
            print(lineCU)
            rCU += 1

    var I := Rfit.MatrixNplusONEd[N](uninitialized=True)
    choleskyInversion.invert(C_U, I)

    if CPP_DUMP:
        print("I5")
        var rowsI = I.num_rows()
        var colsI = I.cols()
        var rI = 0
        while rI < rowsI:
            var lineI = ""
            var cI = 0
            while cI < colsI:
                lineI = lineI + String(I[rI, cI]) + " "
                cI += 1
            print(lineI)
            rI += 1

    var u = I * r_u

    radii.set_block[2, 1](0, 0, radii.block[2, 1](0, 0) / radii.block[2, 1](0, 0).norm())
    radii.set_block[2, 1](0, 1, radii.block[2, 1](0, 1) / radii.block[2, 1](0, 1).norm())

    var d : Rfit.Vector2d = hits.block[2, 1](0, 0).cast[DType.float64]() + (-Z[0] + u[0]) * radii.block[2, 1](0, 0)
    var e : Rfit.Vector2d = hits.block[2, 1](0, 1).cast[DType.float64]() + (-Z[1] + u[1]) * radii.block[2, 1](0, 1)

    circle_results.par[0] = math.atan2((e - d)[1], (e - d)[0])
    circle_results.par[1] = -Float64(circle_results.q) * (fast_fit[2, 0].cast[DType.float64]() - math.sqrt(Rfit.sqr(fast_fit[2, 0].cast[DType.float64]()) - 0.25 * (e - d).squaredNorm()))
    circle_results.par[2] = Float64(circle_results.q) * (1.0 / fast_fit[2, 0].cast[DType.float64]() + u[n])

    debug_assert(Float64(circle_results.q) * circle_results.par[1] <= 0)

    var eMinusd : Rfit.Vector2d = e - d
    var tmp1 = eMinusd.squaredNorm()

    var jacobian: Rfit.Matrix3d = Rfit.Matrix3d(uninitialized=True)
    jacobian[0, 0] = (radii[1, 0] * eMinusd[0] - eMinusd[1] * radii[0, 0]) / tmp1
    jacobian[0, 1] = (radii[1, 1] * eMinusd[0] - eMinusd[1] * radii[0, 1]) / tmp1
    jacobian[0, 2] = 0.0
    jacobian[1, 0] = (Float64(circle_results.q) / 2.0) * (eMinusd[0] * radii[0, 0] + eMinusd[1] * radii[1, 0]) / math.sqrt(Rfit.sqr(2.0 * fast_fit[2, 0].cast[DType.float64]()) - tmp1)
    jacobian[1, 1] = (Float64(circle_results.q) / 2.0) * (eMinusd[0] * radii[0, 1] + eMinusd[1] * radii[1, 1]) / math.sqrt(Rfit.sqr(2.0 * fast_fit[2, 0].cast[DType.float64]()) - tmp1)
    jacobian[1, 2] = 0.0
    jacobian[2, 0] = 0.0
    jacobian[2, 1] = 0.0
    jacobian[2, 2] = Float64(circle_results.q)

    circle_results.cov[0, 0] = I[0, 0]
    circle_results.cov[0, 1] = I[0, 1]
    circle_results.cov[0, 2] = I[0, n]
    circle_results.cov[1, 0] = I[1, 0]
    circle_results.cov[1, 1] = I[1, 1]
    circle_results.cov[1, 2] = I[1, n]
    circle_results.cov[2, 0] = I[n, 0]
    circle_results.cov[2, 1] = I[n, 1]
    circle_results.cov[2, 2] = I[n, n]

    circle_results.cov = jacobian * circle_results.cov * jacobian.transpose()

    TranslateKarimaki(circle_results, 0.5 * (e - d)[0], 0.5 * (e - d)[1], jacobian)
    circle_results.cov[0, 0] += (1.0 + Rfit.sqr(slope)) * MultScatt(S[1] - S[0], B, fast_fit[2, 0].cast[DType.float64](), 2, slope)

    TranslateKarimaki(circle_results, d[0], d[1], jacobian)

    circle_results.chi2 = 0.0
    i = 0
    while i < n:
        circle_results.chi2 += Float32(w[i] * Rfit.sqr(Z[i] - u[i]))
        if i > 0 and i < n - 1:
            circle_results.chi2 += Float32(Rfit.sqr(u[i - 1] / (s[i] - s[i - 1]) - u[i] * (s[i + 1] - s[i - 1]) / ((s[i + 1] - s[i]) * (s[i] - s[i - 1])) + u[i + 1] / (s[i + 1] - s[i]) + (s[i + 1] - s[i - 1]) * u[n] / 2.0) / VarBeta[i])
        i += 1


#!
#\brief Performs the Broken Line fit in the straight track case (that is, the fit parameters are only the interceptions u).
#
#\param hits hits coordinates.
#\param hits_cov hits covariance matrix.
#\param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
#\param B magnetic field in Gev/cm/c.
#\param data PreparedBrokenLineData.
#\param line_results struct to be filled with the results in this form:
#-par parameter of the line in this form: (cot(theta), Zip); \n
#-cov covariance matrix of the fitted parameter; \n
#-chi2 value of the cost function in the minimum.
#
#\details The function implements the steps 2 and 3 of the Broken Line fit without the curvature correction.\n
#The step 2 is the least square fit, done by imposing the minimum constraint on the cost function and solving the consequent linear system. It determines the fitted parameters u and their covariance matrix.
#The step 3 is the correction of the fast pre-fitted parameters for the innermost part of the track. It is first done in a comfortable coordinate system (the one in which the first hit is the origin) and then the parameters and their covariance matrix are transformed to the original coordinate system.
def BL_Line_fit[V4: MatrixLike, M6xN: MatrixLike, N: Int](
    hits_ge: M6xN, fast_fit: V4, B: Float64, data: PreparedBrokenLineData[N], mut line_results: Rfit.line_fit
):
    var n: Int = N
    var i: Int = 0

    ref radii = data.radii
    ref S = data.S
    ref Z = data.Z
    ref VarBeta = data.VarBeta

    var slope: Float64 = -Float64(data.q) / fast_fit[3, 0].cast[DType.float64]()
    var R = RotationMatrix(slope)

    var V : Rfit.Matrix3d = Rfit.Matrix3d.Zero()
    var JacobXYZtosZ : Rfit.Matrix2x3d = Rfit.Matrix2x3d.Zero()
    var w :Rfit.VectorNd[N]= Rfit.VectorNd[N].Zero()
    i = 0
    while i < n:
        V[0, 0] = hits_ge[0, i].cast[DType.float64]()
        V[0, 1] = hits_ge[1, i].cast[DType.float64]()
        V[1, 0] = hits_ge[1, i].cast[DType.float64]()
        V[0, 2] = hits_ge[3, i].cast[DType.float64]()
        V[2, 0] = hits_ge[3, i].cast[DType.float64]()
        V[1, 1] = hits_ge[2, i].cast[DType.float64]()
        V[2, 1] = hits_ge[4, i].cast[DType.float64]()
        V[1, 2] = hits_ge[4, i].cast[DType.float64]()
        V[2, 2] = hits_ge[5, i].cast[DType.float64]()
        var tmp = 1.0 / radii.block[2, 1](0, i).norm()
        JacobXYZtosZ[0, 0] = radii[1, i] * tmp
        JacobXYZtosZ[0, 1] = -radii[0, i] * tmp
        JacobXYZtosZ[1, 2] = 1.0
        w[i] = 1.0 / ((R * JacobXYZtosZ * V * JacobXYZtosZ.transpose() * R.transpose())[1, 1])
        i += 1
    var r_u : Rfit.VectorNd[N] = Rfit.VectorNd[N](uninitialized=True)
    i = 0
    while i < n:
        r_u[i] = w[i] * Z[i]
        i += 1

    if CPP_DUMP:
        print("CU4")
        var CU = MatrixC_u(w, S, VarBeta)
        var rowsCU = CU.num_rows()
        var colsCU = CU.cols()
        var rCU = 0
        while rCU < rowsCU:
            var lineCU = ""
            var cCU = 0
            while cCU < colsCU:
                lineCU = lineCU + String(CU[rCU, cCU]) + " "
                cCU += 1
            print(lineCU)
            rCU += 1

    var I : Rfit.MatrixNd[N] = Rfit.MatrixNd[N](uninitialized=True)
    var CUmat = MatrixC_u(w, S, VarBeta)
    choleskyInversion.invert(CUmat, I)

    if CPP_DUMP:
        print("I4")
        var rowsI = I.num_rows()
        var colsI = I.cols()
        var rI = 0
        while rI < rowsI:
            var lineI = ""
            var cI = 0
            while cI < colsI:
                lineI = lineI + String(I[rI, cI]) + " "
                cI += 1
            print(lineI)
            rI += 1

    var u : Rfit.VectorNd[N] = I * r_u

    line_results.par[0] = (u[1] - u[0]) / (S[1] - S[0])
    line_results.par[1] = u[0]
    var idiff = 1.0 / (S[1] - S[0])
    line_results.cov[0, 0] = (I[0, 0] - 2 * I[0, 1] + I[1, 1]) * Rfit.sqr(idiff) + MultScatt(S[1] - S[0], B, fast_fit[2, 0].cast[DType.float64](), 2, slope)
    line_results.cov[0, 1] = (I[0, 1] - I[0, 0]) * idiff
    line_results.cov[1, 0] = (I[0, 1] - I[0, 0]) * idiff
    line_results.cov[1, 1] = I[0, 0]

    var jacobian = Rfit.Matrix2d(uninitialized=True)
    jacobian[0, 0] = 1.0
    jacobian[0, 1] = 0.0
    jacobian[1, 0] = -S[0]
    jacobian[1, 1] = 1.0
    line_results.par[1] += -line_results.par[0] * S[0]
    line_results.cov = jacobian * line_results.cov * jacobian.transpose()

    var tmp2 = R[0, 0] - line_results.par[0] * R[0, 1]
    jacobian[1, 1] = 1.0 / tmp2
    jacobian[0, 0] = jacobian[1, 1] * jacobian[1, 1]
    jacobian[0, 1] = 0.0
    jacobian[1, 0] = line_results.par[1] * R[0, 1] * jacobian[0, 0]
    line_results.par[1] = line_results.par[1] * jacobian[1, 1]
    line_results.par[0] = (R[0, 1] + line_results.par[0] * R[0, 0]) * jacobian[1, 1]
    line_results.cov = jacobian * line_results.cov * jacobian.transpose()

    line_results.chi2 = 0.0
    i = 0
    while i < n:
        line_results.chi2 += w[i] * Rfit.sqr(Z[i] - u[i])
        if i > 0 and i < n - 1:
            line_results.chi2 += Rfit.sqr(u[i - 1] / (S[i] - S[i - 1]) - u[i] * (S[i + 1] - S[i - 1]) / ((S[i + 1] - S[i]) * (S[i] - S[i - 1])) + u[i + 1] / (S[i + 1] - S[i])) / VarBeta[i]
        i += 1


#!
#\brief Helix fit by three step:
#-fast pre-fit (see Fast_fit() for further info); \n
#-circle fit of the hits projected in the transverse plane by Broken Line algorithm (see BL_Circle_fit() for further info); \n
#-line fit of the hits projected on the (pre-fitted) cilinder surface by Broken Line algorithm (see BL_Line_fit() for further info); \n
#Points must be passed ordered (from inner to outer layer).
#
#\param hits Matrix3xNd hits coordinates in this form: \n
#|x1|x2|x3|...|xn| \n
#|y1|y2|y3|...|yn| \n
#|z1|z2|z3|...|zn|
#\param hits_cov Matrix3Nd covariance matrix in this form (()->cov()): \n
#|(x1,x1)|(x2,x1)|(x3,x1)|(x4,x1)|.|(y1,x1)|(y2,x1)|(y3,x1)|(y4,x1)|.|(z1,x1)|(z2,x1)|(z3,x1)|(z4,x1)| \n
#|(x1,x2)|(x2,x2)|(x3,x2)|(x4,x2)|.|(y1,x2)|(y2,x2)|(y3,x2)|(y4,x2)|.|(z1,x2)|(z2,x2)|(z3,x2)|(z4,x2)| \n
#|(x1,x3)|(x2,x3)|(x3,x3)|(x4,x3)|.|(y1,x3)|(y2,x3)|(y3,x3)|(y4,x3)|.|(z1,x3)|(z2,x3)|(z3,x3)|(z4,x3)| \n
#|(x1,x4)|(x2,x4)|(x3,x4)|(x4,x4)|.|(y1,x4)|(y2,x4)|(y3,x4)|(y4,x4)|.|(z1,x4)|(z2,x4)|(z3,x4)|(z4,x4)| \n
#.       .       .       .       . .       .       .       .       . .       .       .       .       . \n
#|(x1,y1)|(x2,y1)|(x3,y1)|(x4,y1)|.|(y1,y1)|(y2,y1)|(y3,x1)|(y4,y1)|.|(z1,y1)|(z2,y1)|(z3,y1)|(z4,y1)| \n
#|(x1,y2)|(x2,y2)|(x3,y2)|(x4,y2)|.|(y1,y2)|(y2,y2)|(y3,x2)|(y4,y2)|.|(z1,y2)|(z2,y2)|(z3,y2)|(z4,y2)| \n
#|(x1,y3)|(x2,y3)|(x3,y3)|(x4,y3)|.|(y1,y3)|(y2,y3)|(y3,x3)|(y4,y3)|.|(z1,y3)|(z2,y3)|(z3,y3)|(z4,y3)| \n
#|(x1,y4)|(x2,y4)|(x3,y4)|(x4,y4)|.|(y1,y4)|(y2,y4)|(y3,x4)|(y4,y4)|.|(z1,y4)|(z2,y4)|(z3,y4)|(z4,y4)| \n
#.       .       .    .          . .       .       .       .       . .       .       .       .       . \n
#|(x1,z1)|(x2,z1)|(x3,z1)|(x4,z1)|.|(y1,z1)|(y2,z1)|(y3,z1)|(y4,z1)|.|(z1,z1)|(z2,z1)|(z3,z1)|(z4,z1)| \n
#|(x1,z2)|(x2,z2)|(x3,z2)|(x4,z2)|.|(y1,z2)|(y2,z2)|(y3,z2)|(y4,z2)|.|(z1,z2)|(z2,z2)|(z3,z2)|(z4,z2)| \n
#|(x1,z3)|(x2,z3)|(x3,z3)|(x4,z3)|.|(y1,z3)|(y2,z3)|(y3,z3)|(y4,z3)|.|(z1,z3)|(z2,z3)|(z3,z3)|(z4,z3)| \n
#|(x1,z4)|(x2,z4)|(x3,z4)|(x4,z4)|.|(y1,z4)|(y2,z4)|(y3,z4)|(y4,z4)|.|(z1,z4)|(z2,z4)|(z3,z4)|(z4,z4)|
#\param B magnetic field in the center of the detector in Gev/cm/c, in order to perform the p_t calculation.
#
#\warning see BL_Circle_fit(), BL_Line_fit() and Fast_fit() warnings.
#
#\bug see BL_Circle_fit(), BL_Line_fit() and Fast_fit() bugs.
#
#\return (phi,Tip,p_t,cot(theta)),Zip), their covariance matrix and the chi2's of the circle and line fits.
def BL_Helix_fit[N: Int](
    hits: Rfit.Matrix3xNd[N],
    hits_ge: Matrix[DType.float32, 6, N],
    B: Float64,
) -> Rfit.helix_fit:
    var helix = Rfit.helix_fit()
    var fast_fit = Rfit.Vector4d(uninitialized=True)
    BL_Fast_fit(hits, fast_fit)

    var data = PreparedBrokenLineData[N]()
    var circle = karimaki_circle_fit()
    var line = Rfit.line_fit()
    var jacobian = Rfit.Matrix3d(uninitialized=True)

    prepareBrokenLineData(hits, fast_fit, B, data)
    BL_Line_fit(hits_ge, fast_fit, B, data, line)
    BL_Circle_fit(hits, hits_ge, fast_fit, B, data, circle)

    jacobian[0, 0] = 1.0
    jacobian[0, 1] = 0.0
    jacobian[0, 2] = 0.0
    jacobian[1, 0] = 0.0
    jacobian[1, 1] = 1.0
    jacobian[1, 2] = 0.0
    jacobian[2, 0] = 0.0
    jacobian[2, 1] = 0.0
    jacobian[2, 2] = -abs(circle.par[2]) * B / (Rfit.sqr(circle.par[2]) * circle.par[2])
    circle.par[2] = B / abs(circle.par[2])
    circle.cov = jacobian * circle.cov * jacobian.transpose()

    helix.par[0] = circle.par[0]
    helix.par[1] = circle.par[1]
    helix.par[2] = circle.par[2]
    helix.par[3] = line.par[0]
    helix.par[4] = line.par[1]
    helix.cov = Matrix[DType.float64, 5, 5]()
    helix.cov.set_block[3, 3](0, 0, circle.cov)
    helix.cov.set_block[2, 2](3, 3, line.cov)
    helix.q = circle.q
    helix.chi2_circle = circle.chi2
    helix.chi2_line = Float32(line.chi2)

    return helix
