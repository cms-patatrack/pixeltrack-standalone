import std.math as math
from std.sys import is_defined

import MojoSerial.plugin_PixelTriplets.choleskyInversion as choleskyInversion
from MojoSerial.plugin_PixelTriplets.FitUtils import Rfit
from MojoSerial.MojoBridge.Matrix import Matrix, MatrixLike
from MojoSerial.MojoBridge.SymmetricEigen import (
    min_eigen_2x2,
    min_eigen_3x3,
    min_eigen_3x3_fast,
)


#/*!  Compute the Radiation length in the uniform hypothesis
# *
# * The Pixel detector, barrel and forward, is considered as an omogeneous
# * cilinder of material, whose radiation lengths has been derived from the TDR
# * plot that shows that 16cm correspond to 0.06 radiation lengths. Therefore
# * one radiation length corresponds to 16cm/0.06 =~ 267 cm. All radiation
# * lengths are computed using this unique number, in both regions, barrel and
# * endcap.
# *
# * NB: no angle corrections nor projections are computed inside this routine.
# * It is therefore the responsibility of the caller to supply the proper
# * lengths in input. These lenghts are the path travelled by the particle along
# * its trajectory, namely the so called S of the helix in 3D space.
# *
# * \param length_values vector of incremental distances that will be translated
# * into radiation length equivalent. Each radiation length i is computed
# * incrementally with respect to the previous length i-1. The first lenght has
# * no reference point (i.e. it has the dca).
# *
# * \return incremental radiation lengths that correspond to each segment.
# */
def computeRadLenUniformMaterial[
    VNd1: MatrixLike,
    VNd2: MatrixLike,
](length_values: VNd1, mut rad_lengths:  VNd2):
    # Radiation length of the pixel detector in the uniform assumption, with
    # 0.06 rad_len at 16 cm
    comptime XX_0_inv: Float64 = 0.06 / 16.0
    var n = length_values.num_rows()
    rad_lengths[0] = (length_values[0].cast[DType.float64]() * XX_0_inv).cast[VNd2.ElemType]()
    var j: Int = 1
    while j < n:
        rad_lengths[j] = (abs(length_values[j].cast[DType.float64]() - length_values[j - 1].cast[DType.float64]()) * XX_0_inv).cast[VNd2.ElemType]()
        j += 1


#/*!
#   \brief Compute the covariance matrix along cartesian S-Z of points due to
#   multiple Coulomb scattering to be used in the line_fit, for the barrel
#   and forward cases.
#   The input covariance matrix is in the variables s-z, original and
#   unrotated.
#   The multiple scattering component is computed in the usual linear
#   approximation, using the 3D path which is computed as the squared root of
#   the squared sum of the s and z components passed in.
#   Internally a rotation by theta is performed and the covariance matrix
#   returned is the one in the direction orthogonal to the rotated S3D axis,
#   i.e. along the rotated Z axis.
#   The choice of the rotation is not arbitrary, but derived from the fact that
#   putting the horizontal axis along the S3D direction allows the usage of the
#   ordinary least squared fitting techiques with the trivial parametrization y
#   = mx + q, avoiding the patological case with m = +/- inf, that would
#   correspond to the case at eta = 0.
# */
def Scatter_cov_line[
    V4: MatrixLike,
    VNd1: MatrixLike,
    VNd2: MatrixLike,
    N: Int,
](
    cov_sz: Span[Rfit.Matrix2d, _],
    fast_fit: V4,
    s_arcs: VNd1,
    z_values: VNd2,
    theta: Float64,
    B: Float64,
    mut ret: Rfit.MatrixNd[N],
):

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](s_arcs, "Scatter_cov_line - s_arcs: ")

    comptime n: Int = N
    # limit pt to avoid too small error!!!
    var p_t = min(20.0, fast_fit[2].cast[DType.float64]() * B)
    var fast_fit_3 = fast_fit[3].cast[DType.float64]()
    var p_2 = p_t * p_t * (1.0 + 1.0 / (fast_fit_3 * fast_fit_3))

    var rad_lengths_S = Rfit.VectorNd[N]()
    # See documentation at http://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html
    # Basically, to perform cwise operations on Matrices and Vectors, you need
    # to transform them into Array-like objects.
    var S_values = Rfit.VectorNd[N]()
    var i = 0
    while i < n:
        var s_val = s_arcs[i].cast[DType.float64]()
        var z_val = z_values[i].cast[DType.float64]()
        S_values[i] = s_val * s_val + z_val * z_val
        i += 1
     
    i = 0
    while i < n:
        S_values[i] = math.sqrt(S_values[i])
        i += 1
    computeRadLenUniformMaterial(S_values, rad_lengths_S)
    var sig2_S = Rfit.VectorNd[N]()
    
    i = 0
    while i < n:
        var tmp = 1.0 + 0.038 * math.log(rad_lengths_S[i])
        sig2_S[i] = 0.000225 / p_2 * (tmp * tmp) * rad_lengths_S[i]
        i += 1

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](cov_sz[0], "Scatter_cov_line - cov_sz: ")



    var tmp = Rfit.Matrix2Nd[N].Zero()

    for k in range(n):
        tmp[k, k] = cov_sz[k][0, 0]
        tmp[k + n, k + n] = cov_sz[k][1, 1]
        tmp[k, k + n] = cov_sz[k][0, 1]
        tmp[k + n, k] = cov_sz[k][0, 1]



    for k in range(n):
        for l in range(k, n):
            for i in range(min(k, l)):
                tmp[k + n, l + n] += abs(S_values[k] - S_values[i]) * abs(S_values[l] - S_values[i]) * sig2_S[i]
            tmp[l + n, k + n] = tmp[k + n, l + n]


    

    






    # We are interested only in the errors orthogonal to the rotated s-axis
    # which, in our formalism, are in the lower square matrix.
    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](tmp, "Scatter_cov_line - tmp: ")

    for i in range(n):
        for j in range(n):
            ret[i, j] = tmp[i + n, j + n]


#/*!
#   \brief Compute the covariance matrix (in radial coordinates) of points in
#   the transverse plane due to multiple Coulomb scattering.
#   \param p2D 2D points in the transverse plane.
#   \param fast_fit fast_fit Vector4d result of the previous pre-fit
#   structured in this form:(X0, Y0, R, Tan(Theta))).
#   \param B magnetic field use to compute p
#   \return scatter_cov_rad errors due to multiple scattering.
#   \warning input points must be ordered radially from the detector center
#   (from inner layer to outer ones; points on the same layer must ordered too).
#   \details Only the tangential component is computed (the radial one is
#   negligible).
# */
def Scatter_cov_rad[
    M2xN: MatrixLike,
    V4: MatrixLike,
    N: Int,
](
    p2D: M2xN,
    fast_fit: V4,
    rad: Rfit.VectorNd[N],
    B: Float64,
) -> Rfit.MatrixNd[N]:
    comptime n: Int = N
    var ff2 = fast_fit[2].cast[DType.float64]()
    var ff3 = fast_fit[3].cast[DType.float64]()
    # limit pt to avoid too small error!!!
    var p_t = min(20.0, ff2 * B)
    var p_2 = p_t * p_t * (1.0 + 1.0 / (ff3 * ff3))
    var theta = math.atan(ff3)
    if theta < 0.0:
        theta += math.pi

    var s_values = Rfit.VectorNd[N]()
    var rad_lengths = Rfit.VectorNd[N]()
    var o = Rfit.Vector2d()
    o[0] = fast_fit[0].cast[DType.float64]()
    o[1] = fast_fit[1].cast[DType.float64]()

    # associated Jacobian, used in weights and errors computation
    for i in range(n):  # x
        var px = p2D[0, i].cast[DType.float64]() - o[0]
        var py = p2D[1, i].cast[DType.float64]() - o[1]
        var p = Rfit.Vector2d()
        p[0] = px
        p[1] = py
        var o_neg = Rfit.Vector2d()
        o_neg[0] = -o[0]
        o_neg[1] = -o[1]
        var cross: Float64 = Rfit.cross2D(o_neg, p)
        var dot: Float64 = o_neg[0] * p[0] + o_neg[1] * p[1]
        var atan2_: Float64 = math.atan2(cross, dot)
        s_values[i] = abs(atan2_ * ff2)

    var scale = math.sqrt(1.0 + 1.0 / (ff3 * ff3))
    var scaled_s_values = Rfit.VectorNd[N]()
    for i in range(n):
        scaled_s_values[i] = s_values[i] * scale

    computeRadLenUniformMaterial(scaled_s_values, rad_lengths)

    var scatter_cov_rad = Rfit.MatrixNd[N].Zero()
    var sig2 = Rfit.VectorNd[N]()
    var sin_theta = math.sin(theta)
    var inv_factor = 0.000225 / (p_2 * Rfit.sqr(sin_theta))
    for i in range(n):
        var tmp = 1.0 + 0.038 * math.log(rad_lengths[i])
        sig2[i] = (tmp * tmp) * rad_lengths[i] * inv_factor

    for k in range(n):
        for l in range(k, n):
            for i in range(min(k, l)):
                scatter_cov_rad[k, l] += (rad[k] - rad[i]) * (rad[l] - rad[i]) * sig2[i]
            scatter_cov_rad[l, k] = scatter_cov_rad[k, l]

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](scatter_cov_rad, "Scatter_cov_rad - scatter_cov_rad: ")

    return scatter_cov_rad


#/*!
#   \brief Transform covariance matrix from radial (only tangential component)
#   to Cartesian coordinates (only transverse plane component).
#   \param p2D 2D points in the transverse plane.
#   \param cov_rad covariance matrix in radial coordinate.
#   \return cov_cart covariance matrix in Cartesian coordinates.
# */
def cov_radtocart[
    M2xN: MatrixLike,
    N: Int,
](
    p2D: M2xN,
    cov_rad: Rfit.MatrixNd[N],
    rad: Rfit.VectorNd[N],
) -> Rfit.Matrix2Nd[N]:
    comptime if is_defined["RFIT_DEBUG"]():
        print("Address of p2D: ", Pointer(to=p2D))

    comptime n: Int = N
    var cov_cart = Rfit.Matrix2Nd[N].Zero()
    var rad_inv = Rfit.VectorNd[N]()
    for i in range(n):
        rad_inv[i] = 1.0 / rad[i]

    #####
    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](p2D, "cov_radtocart - p2D:")
        Rfit.printIt[RFIT_DEBUG=True](rad_inv, "cov_radtocart - rad_inv:")



    for i in range(n):
        for j in range(i, n):
            var p2D_0i = p2D[0, i].cast[DType.float64]()
            var p2D_1i = p2D[1, i].cast[DType.float64]()
            var p2D_0j = p2D[0, j].cast[DType.float64]()
            var p2D_1j = p2D[1, j].cast[DType.float64]()
            cov_cart[i, j] = cov_rad[i, j] * p2D_1i * rad_inv[i] * p2D_1j * rad_inv[j]
            cov_cart[i + n, j + n] = cov_rad[i, j] * p2D_0i * rad_inv[i] * p2D_0j * rad_inv[j]
            cov_cart[i, j + n] = -cov_rad[i, j] * p2D_1i * rad_inv[i] * p2D_0j * rad_inv[j]
            cov_cart[i + n, j] = -cov_rad[i, j] * p2D_0i * rad_inv[i] * p2D_1j * rad_inv[j]
            cov_cart[j, i] = cov_cart[i, j]
            cov_cart[j + n, i + n] = cov_cart[i + n, j + n]
            cov_cart[j + n, i] = cov_cart[i, j + n]
            cov_cart[j, i + n] = cov_cart[i + n, j]

    return cov_cart


#/*!
#   \brief Transform covariance matrix from Cartesian coordinates (only
#   transverse plane component) to radial coordinates (both radial and
#   tangential component but only diagonal terms, correlation between different
#   point are not managed).
#   \param p2D 2D points in transverse plane.
#   \param cov_cart covariance matrix in Cartesian coordinates.
#   \return cov_rad covariance matrix in raidal coordinate.
#   \warning correlation between different point are not computed.
# */
def cov_carttorad[
    M2xN: MatrixLike,
    N: Int,
](
    p2D: M2xN,
    cov_cart: Rfit.Matrix2Nd[N],
    rad: Rfit.VectorNd[N],
) -> Rfit.VectorNd[N]:
    comptime n: Int = N
    var cov_rad = Rfit.VectorNd[N]()
    var rad_inv2 = Rfit.VectorNd[N]()
    for i in range(n):
        var inv = 1.0 / rad[i]
        rad_inv2[i] = inv * inv

    for i in range(n):
        # in case you have (0,0) to avoid dividing by 0 radius
        if rad[i] < 1.0e-4:
            cov_rad[i] = cov_cart[i, i]
        else:
            var p2D_0i = p2D[0, i].cast[DType.float64]()
            var p2D_1i = p2D[1, i].cast[DType.float64]()
            cov_rad[i] = rad_inv2[i] * (
                cov_cart[i, i] * Rfit.sqr(p2D_1i)
                + cov_cart[i + n, i + n] * Rfit.sqr(p2D_0i)
                - 2.0 * cov_cart[i, i + n] * p2D_0i * p2D_1i
            )

    return cov_rad


#/*!
#   \brief Transform covariance matrix from Cartesian coordinates (only
#   transverse plane component) to coordinates system orthogonal to the
#   pre-fitted circle in each point.
#   Further information in attached documentation.
#   \param p2D 2D points in transverse plane.
#   \param cov_cart covariance matrix in Cartesian coordinates.
#   \param fast_fit fast_fit Vector4d result of the previous pre-fit
#   structured in this form:(X0, Y0, R, tan(theta))).
#   \return cov_rad covariance matrix in the pre-fitted circle's
#   orthogonal system.
# */
def cov_carttorad_prefit[
    M2xN: MatrixLike,
    V4: MatrixLike,
    N: Int,
](
    p2D: M2xN,
    cov_cart: Rfit.Matrix2Nd[N],
    fast_fit: V4,
    rad: Rfit.VectorNd[N],
) -> Rfit.VectorNd[N]:
    comptime n: Int = N
    var cov_rad = Rfit.VectorNd[N]()
    for i in range(n):
        # in case you have (0,0) to avoid dividing by 0 radius
        if rad[i] < 1.0e-4:
            cov_rad[i] = cov_cart[i, i]  # TO FIX
        else:
            var ax = p2D[0, i].cast[DType.float64]()
            var ay = p2D[1, i].cast[DType.float64]()
            var bx = ax - fast_fit[0].cast[DType.float64]()
            var by = ay - fast_fit[1].cast[DType.float64]()
            var x2 = ax * bx + ay * by
            var a = Rfit.Vector2d()
            a[0] = ax
            a[1] = ay
            var b = Rfit.Vector2d()
            b[0] = bx
            b[1] = by
            var y2 = Rfit.cross2D(a, b)
            var tan_c = -y2 / x2
            var tan_c2 = Rfit.sqr(tan_c)
            cov_rad[i] = 1.0 / (1.0 + tan_c2) * (
                cov_cart[i, i]
                + cov_cart[i + n, i + n] * tan_c2
                + 2.0 * cov_cart[i, i + n] * tan_c
            )

    return cov_rad


#/*!
#   \brief Compute the points' weights' vector for the circle fit when multiple
#   scattering is managed.
#   Further information in attached documentation.
#   \param cov_rad_inv covariance matrix inverse in radial coordinated
#   (or, beter, pre-fitted circle's orthogonal system).
#   \return weight VectorNd points' weights' vector.
#   \bug I'm not sure this is the right way to compute the weights for non
#   diagonal cov matrix. Further investigation needed.
# */
def Weight_circle[
    N: Int,
](cov_rad_inv: Rfit.MatrixNd[N]) -> Rfit.VectorNd[N]:
    var n: Int = N
    var weight = Rfit.VectorNd[N]()
    for j in range(n):
        var sum: Float64 = 0.0
        for i in range(n):
            sum += cov_rad_inv[i, j]
        weight[j] = sum
    return weight


#/*!
#   \brief Find particle q considering the  sign of cross product between
#   particles velocity (estimated by the first 2 hits) and the vector radius
#   between the first hit and the center of the fitted circle.
#   \param p2D 2D points in transverse plane.
#   \param par_uvr result of the circle fit in this form: (X0,Y0,R).
#   \return q int 1 or -1.
# */
def Charge[
    M2xN: MatrixLike,
](p2D: M2xN, par_uvr: Rfit.Vector3d) -> Int32:
    var p2D_00 = p2D[0, 0].cast[DType.float64]()
    var p2D_01 = p2D[0, 1].cast[DType.float64]()
    var p2D_10 = p2D[1, 0].cast[DType.float64]()
    var p2D_11 = p2D[1, 1].cast[DType.float64]()
    var val = (
        (p2D_01 - p2D_00) * (par_uvr[1] - p2D_10)
        - (p2D_11 - p2D_10) * (par_uvr[0] - p2D_00)
    )
    return -1 if val > 0.0 else 1


#/*!
#   \brief Compute the eigenvector associated to the minimum eigenvalue.
#   \param A the Matrix you want to know eigenvector and eigenvalue.
#   \param chi2 the double were the chi2-related quantity will be stored.
#   \return the eigenvector associated to the minimum eigenvalue.
#   \warning double precision is needed for a correct assessment of chi2.
#   \details The minimus eigenvalue is related to chi2.
#   We exploit the fact that the matrix is symmetrical and small (2x2 for line
#   fit and 3x3 for circle fit), so the SelfAdjointEigenSolver from Eigen
#   library is used, with the computedDirect  method (available only for 2x2
#   and 3x3 Matrix) wich computes eigendecomposition of given matrix using a
#   fast closed-form algorithm.
#   For this optimization the matrix type must be known at compiling time.
# */
def min_eigen3D(
    A: Rfit.Matrix3d,
    mut chi2: Float64,
) -> Rfit.Vector3d:
    comptime if is_defined["RFIT_DEBUG"]():
        print("min_eigen3D - enter")

    var v = min_eigen_3x3(A, chi2)

    comptime if is_defined["RFIT_DEBUG"]():
        print("min_eigen3D - exit")

    return v


#/*!
#   \brief A faster version of min_eigen3D() where double precision is not
#   needed.
#   \param A the Matrix you want to know eigenvector and eigenvalue.
#   \param chi2 the double were the chi2-related quantity will be stored
#   \return the eigenvector associated to the minimum eigenvalue.
#   \detail The computedDirect() method of SelfAdjointEigenSolver for 3x3 Matrix
#   indeed, use trigonometry function (it solves a third degree equation) which
#   speed up in  single precision.
# */
def min_eigen3D_fast(
    A: Rfit.Matrix3d,
) -> Rfit.Vector3d:
    return min_eigen_3x3_fast(A)


#/*!
#   \brief 2D version of min_eigen3D().
#   \param A the Matrix you want to know eigenvector and eigenvalue.
#   \param chi2 the double were the chi2-related quantity will be stored
#   \return the eigenvector associated to the minimum eigenvalue.
#   \detail The computedDirect() method of SelfAdjointEigenSolver for 2x2 Matrix
#   do not use special math function (just sqrt) therefore it doesn't speed up
#   significantly in single precision.
# */
def min_eigen2D(
    A: Rfit.Matrix2d,
    mut chi2: Float64,
) -> Rfit.Vector2d:
    return min_eigen_2x2(A, chi2)


#/*!
#   \brief A very fast helix fit: it fits a circle by three points (first, middle
#   and last point) and a line by two points (first and last).
#   \param hits points to be fitted
#   \return result in this form: (X0,Y0,R,tan(theta)).
#   \warning points must be passed ordered (from internal layer to external) in
#   order to maximize accuracy and do not mistake tan(theta) sign.
#   \details This fast fit is used as pre-fit which is needed for:
#   - weights estimation and chi2 computation in line fit (fundamental);
#   - weights estimation and chi2 computation in circle fit (useful);
#   - computation of error due to multiple scattering.
# */
def Fast_fit[
    M3xN: MatrixLike,
    V4: MatrixLike,
](hits: M3xN, mut result: V4):
    # get the number of hits
    var n = M3xN.ColsAtCompileTime()

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](hits, "Fast_fit - hits: ")

    # CIRCLE FIT
    # Make segments between middle-to-first(b) and last-to-first(c) hits
    var mid = n // 2
    var hits_00 = hits[0, 0].cast[DType.float64]()
    var hits_10 = hits[1, 0].cast[DType.float64]()
    var b0 = hits[0, mid].cast[DType.float64]() - hits_00
    var b1 = hits[1, mid].cast[DType.float64]() - hits_10
    var c0 = hits[0, n - 1].cast[DType.float64]() - hits_00
    var c1 = hits[1, n - 1].cast[DType.float64]() - hits_10

    # Compute their lengths
    var b2 = b0 * b0 + b1 * b1
    var c2 = c0 * c0 + c1 * c1

    var b = Rfit.Vector2d()
    b[0] = b0
    b[1] = b1
    var c = Rfit.Vector2d()
    c[0] = c0
    c[1] = c1

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](b, "Fast_fit - b: ")
        Rfit.printIt[RFIT_DEBUG=True](c, "Fast_fit - c: ")

    # The algebra has been verified (MR). The usual approach has been followed:
    # * use an orthogonal reference frame passing from the first point.
    # * build the segments (chords)
    # * build orthogonal lines through mid points
    # * make a system and solve for X0 and Y0.
    # * add the initial point
    var flip = abs(b0) < abs(b1)
    var bx = b1 if flip else b0
    var by = b0 if flip else b1
    var cx = c1 if flip else c0
    var cy = c0 if flip else c1

    # in case b.x is 0 (2 hits with same x)
    var div = 2.0 * (cx * by - bx * cy)
    # if aligned TO FIX
    var Y0 = (cx * b2 - bx * c2) / div
    var X0 = (0.5 * b2 - Y0 * by) / bx

    result[0] = (hits_00 + (Y0 if flip else X0)).cast[V4.ElemType]()
    result[1] = (hits_10 + (X0 if flip else Y0)).cast[V4.ElemType]()
    result[2] = math.sqrt(Rfit.sqr(X0) + Rfit.sqr(Y0)).cast[V4.ElemType]()

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](result, "Fast_fit - result: ")

    # LINE FIT
    var result_0 = result[0].cast[DType.float64]()
    var result_1 = result[1].cast[DType.float64]()
    var d0 = hits_00 - result_0
    var d1 = hits_10 - result_1
    var e0 = hits[0, n - 1].cast[DType.float64]() - result_0
    var e1 = hits[1, n - 1].cast[DType.float64]() - result_1

    var d = Rfit.Vector2d()
    d[0] = d0
    d[1] = d1
    var e = Rfit.Vector2d()
    e[0] = e0
    e[1] = e1

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](e, "Fast_fit - e: ")
        Rfit.printIt[RFIT_DEBUG=True](d, "Fast_fit - d: ")

    var cross = Rfit.cross2D(d, e)
    var dot = d0 * e0 + d1 * e1
    # Compute the arc-length between first and last point: L = R * theta = R * atan (tan (Theta) )
    var dr = result[2].cast[DType.float64]() * math.atan2(cross, dot)
    # Simple difference in Z between last and first hit
    var dz = hits[2, n - 1].cast[DType.float64]() - hits[2, 0].cast[DType.float64]()

    result[3] = (dr / dz).cast[V4.ElemType]()

    comptime if is_defined["RFIT_DEBUG"]():
        print("Fast_fit: [", result[0], ", ", result[1], ", ", result[2], ", ", result[3], "]")


#/*!
#   \brief Fit a generic number of 2D points with a circle using Riemann-Chernov
#   algorithm. Covariance matrix of fitted parameter is optionally computed.
#   Multiple scattering (currently only in barrel layer) is optionally handled.
#   \param hits2D 2D points to be fitted.
#   \param hits_cov2D covariance matrix of 2D points.
#   \param fast_fit pre-fit result in this form: (X0,Y0,R,tan(theta)).
#   (tan(theta) is not used).
#   \param B magnetic field
#   \param error flag for error computation.
#   \param scattering flag for multiple scattering
#   \return circle circle_fit:
#   -par parameter of the fitted circle in this form (X0,Y0,R); \n
#   -cov covariance matrix of the fitted parameter (not initialized if
#   error = false); \n
#   -q charge of the particle; \n
#   -chi2.
#   \warning hits must be passed ordered from inner to outer layer (double hits
#   on the same layer must be ordered too) so that multiple scattering is
#   treated properly.
#   \warning Multiple scattering for barrel is still not tested.
#   \warning Multiple scattering for endcap hits is not handled (yet). Do not
#   fit endcap hits with scattering = true !
#   \bug for small pt (<0.3 Gev/c) chi2 could be slightly underestimated.
#   \bug further investigation needed for error propagation with multiple
#   scattering.
# */
def Circle_fit[
    M2xN: MatrixLike,
    V4: MatrixLike,
    N: Int,
](
    hits2D: M2xN,
    hits_cov2D: Rfit.Matrix2Nd[N],
    fast_fit: V4,
    rad: Rfit.VectorNd[N],
    B: Float64,
    error: Bool,
) -> Rfit.circle_fit:
    comptime if is_defined["RFIT_DEBUG"]():
        print("circle_fit - enter")

    # INITIALIZATION
    var V: Rfit.Matrix2Nd[N]= hits_cov2D
    comptime n: Int = N

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](hits2D, "circle_fit - hits2D:")
        Rfit.printIt[RFIT_DEBUG=True](hits_cov2D, "circle_fit - hits_cov2D:")

    comptime if is_defined["RFIT_DEBUG"]():
        print("circle_fit - WEIGHT COMPUTATION")

    # WEIGHT COMPUTATION
    var weight = Rfit.VectorNd[N]()
    var G = Rfit.MatrixNd[N]()
    var renorm: Float64 = 0.0


    var cov_rad_vec = cov_carttorad_prefit[M2xN, V4, N](hits2D, V, fast_fit, rad)
    var cov_rad = Rfit.MatrixNd[N].Zero()
    for i in range(n):
        cov_rad[i, i] = cov_rad_vec[i]

    var scatter_cov_rad = Scatter_cov_rad[M2xN, V4, N](hits2D, fast_fit, rad, B)

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](scatter_cov_rad, "circle_fit - scatter_cov_rad:")
        Rfit.printIt[RFIT_DEBUG=True](hits2D, "circle_fit - hits2D bis:")
        print("Address of hits2D: a) ", Pointer(to=hits2D))

    V += cov_radtocart[M2xN, N](hits2D, scatter_cov_rad, rad)

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](V, "circle_fit - V:")

    cov_rad += scatter_cov_rad

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](cov_rad, "circle_fit - cov_rad:")

    choleskyInversion.invert(cov_rad, G)
    # G = cov_rad.inverse();
    renorm = Float64(G.reduce_add())
    var scale = 1.0 / renorm
    for i in range(n):
        for j in range(n):
            G[i, j] *= scale
    weight = Weight_circle[N](G)

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](weight, "circle_fit - weight:")
        # SPACE TRANSFORMATION
        print("circle_fit - SPACE TRANSFORMATION")
        # center
        print("Address of hits2D: b) ", Pointer(to=hits2D))



    # centroid
    var h_ = Rfit.Vector2d()
    var sum0: Float64 = 0.0
    var sum1: Float64 = 0.0
    for i in range(n):
        sum0 += hits2D[0, i].cast[DType.float64]()
        sum1 += hits2D[1, i].cast[DType.float64]()
    h_[0] = sum0 / Float64(n)
    h_[1] = sum1 / Float64(n)

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](h_, "circle_fit - h_:")

    var p3D = Rfit.Matrix3xNd[N]()
    for i in range(n):
        p3D[0, i] = hits2D[0, i].cast[DType.float64]() - h_[0]
        p3D[1, i] = hits2D[1, i].cast[DType.float64]() - h_[1]

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](p3D, "circle_fit - p3D: a)")

    # centered hits, used in error computation
    var mc = Rfit.Vector2Nd[N]()
    for i in range(n):
        mc[i] = p3D[0, i]
        mc[i + n] = p3D[1, i]

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](mc, "circle_fit - mc(centered hits):")

    # scale
    var q: Float64 = 0.0
    for i in range(2 * n):
        q += mc[i] * mc[i]

    # scaling factor
    var s = math.sqrt(Float64(n) / q)
    for i in range(3):
        for j in range(n):
            p3D[i, j] *= s

    # project on paraboloid
    for i in range(n):
        p3D[2, i] = p3D[0, i] * p3D[0, i] + p3D[1, i] * p3D[1, i]

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](p3D, "circle_fit - p3D: b)")
        print("circle_fit - COST FUNCTION")

    # COST FUNCTION
    # compute
    # center of gravity
    var r0 = Rfit.Vector3d()
    for i in range(3):
        var sum: Float64 = 0.0
        for j in range(n):
            sum += p3D[i, j] * weight[j]
        r0[i] = sum

    var X = Rfit.Matrix3xNd[N]()
    for j in range(n):
        for i in range(3):
            X[i, j] = p3D[i, j] - r0[i]

    var A = (X @ G) @ X.transpose()

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](A, "circle_fit - A:")
        print("circle_fit - MINIMIZE")

    # minimize
    var chi2: Float64 = 0.0
    var v = min_eigen3D(A, chi2)

    comptime if is_defined["RFIT_DEBUG"]():
        print("circle_fit - AFTER MIN_EIGEN")
        Rfit.printIt[RFIT_DEBUG=True](v, "v BEFORE INVERSION")

    # TO FIX dovrebbe essere N(3)>0
    if v[2] <= 0.0:
        v[0] = -v[0]
        v[1] = -v[1]
        v[2] = -v[2]

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](v, "v AFTER INVERSION")

    # This hack to be able to run on GPU where the automatic assignment to a
    # double from the vector multiplication is not working.
    comptime if is_defined["RFIT_DEBUG"]():
        print("circle_fit - AFTER MIN_EIGEN 1")

    var cm = Matrix[DType.float64, 1, 1]()

    comptime if is_defined["RFIT_DEBUG"]():
        print("circle_fit - AFTER MIN_EIGEN 2")

    cm = -(v.transpose() @ r0)

    comptime if is_defined["RFIT_DEBUG"]():
        print("circle_fit - AFTER MIN_EIGEN 3")

    var c = cm[0, 0]
    # const double c = -v.transpose() * r0;

    comptime if is_defined["RFIT_DEBUG"]():
        print("circle_fit - COMPUTE CIRCLE PARAMETER")

    # COMPUTE CIRCLE PARAMETER
    # auxiliary quantities
    var h = math.sqrt(1.0 - Rfit.sqr(v[2]) - 4.0 * c * v[2])
    var v2x2_inv = 1.0 / (2.0 * v[2])
    var s_inv = 1.0 / s

    # used in error propagation
    var par_uvr_ = Rfit.Vector3d()
    par_uvr_[0] = -v[0] * v2x2_inv
    par_uvr_[1] = -v[1] * v2x2_inv
    par_uvr_[2] = h * v2x2_inv

    var circle = Rfit.circle_fit()
    circle.par[0] = par_uvr_[0] * s_inv + h_[0]
    circle.par[1] = par_uvr_[1] * s_inv + h_[1]
    circle.par[2] = par_uvr_[2] * s_inv
    circle.q = Charge(hits2D, circle.par)
    circle.chi2 = (abs(chi2) * renorm * 1.0 / Rfit.sqr(2.0 * v[2] * par_uvr_[2] * s)).cast[DType.float32]()

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](circle.par, "circle_fit - CIRCLE PARAMETERS:")
        Rfit.printIt[RFIT_DEBUG=True](circle.cov, "circle_fit - CIRCLE COVARIANCE:")
        print("circle_fit - CIRCLE CHARGE: ", circle.q)
        print("circle_fit - ERROR PROPAGATION")

    # ERROR PROPAGATION
    if error:
        comptime if is_defined["RFIT_DEBUG"]():
            print("circle_fit - ERROR PRPAGATION ACTIVATED")

        # cov matrix of center & scaled points
        var Vcs_ = InlineArray[InlineArray[Rfit.ArrayNd[N], 2], 2](
            fill=InlineArray[Rfit.ArrayNd[N], 2](fill=Rfit.ArrayNd[N]())
        )
        # cov matrix of 3D transformed points
        var C = InlineArray[InlineArray[Rfit.MatrixNd[N], 3], 3](
            fill=InlineArray[Rfit.MatrixNd[N], 3](fill=Rfit.MatrixNd[N]())
        )

        comptime if is_defined["RFIT_DEBUG"]():
            print("circle_fit - ERROR PRPAGATION ACTIVATED 2")

        var cm = Matrix[DType.float64, 1, 1]()
        var cm2 = Matrix[DType.float64, 1, 1]()
        cm = (mc.transpose() @ V) @ mc
        var c = cm[0, 0]

        var Vcs = Rfit.Matrix2Nd[N].Zero()
        var V_sq_norm: Float64 = 0.0
        for i in range(2 * n):
            for j in range(2 * n):
                V_sq_norm += V[i, j] * V[i, j]

        # mc.transpose() * V * mc) *
        var scale_vcs = (Rfit.sqr(s) * Rfit.sqr(s)) * (2.0 * V_sq_norm + 4.0 * c) / (4.0 * q * Float64(n))
        for i in range(2 * n):
            for j in range(2 * n):
                Vcs[i, j] = Rfit.sqr(s) * V[i, j] + scale_vcs * mc[i] * mc[j]

        comptime if is_defined["RFIT_DEBUG"]():
            Rfit.printIt[RFIT_DEBUG=True](Vcs, "circle_fit - Vcs:")

        for i in range(n):
            for j in range(n):
                C[0][0][i, j] = Vcs[i, j]
                Vcs_[0][1][i, j] = Vcs[i, j + n]
                C[1][1][i, j] = Vcs[i + n, j + n]
        for i in range(n):
            for j in range(n):
                Vcs_[1][0][i, j] = Vcs_[0][1][j, i]

        comptime if is_defined["RFIT_DEBUG"]():
            Rfit.printIt[RFIT_DEBUG=True](Vcs, "circle_fit - Vcs:")

        var t0 = Rfit.ArrayNd[N]()
        var t1 = Rfit.ArrayNd[N]()
        var t00 = Rfit.ArrayNd[N]()
        var t01 = Rfit.ArrayNd[N]()
        var t11 = Rfit.ArrayNd[N]()
        var t10 = Rfit.ArrayNd[N]()

        for i in range(n):
            for j in range(n):
                t0[i, j] = p3D[0, j]
                t1[i, j] = p3D[1, j]
                t00[i, j] = p3D[0, i] * p3D[0, j]
                t01[i, j] = p3D[0, i] * p3D[1, j]
                t11[i, j] = p3D[1, i] * p3D[1, j]

        # t10 = t01.transpose() -- kept as its own pass since it depends on
        # t01 being fully populated first (unlike t0/t1/t00/t01/t11, each of
        # which only reads p3D directly).
        for i in range(n):
            for j in range(n):
                t10[i, j] = t01[j, i]

        for i in range(n):
            for j in range(n):
                Vcs_[0][0][i, j] = C[0][0][i, j]
                C[0][1][i, j] = Vcs_[0][1][i, j]
                C[0][2][i, j] = 2.0 * (Vcs_[0][0][i, j] * t0[i, j] + Vcs_[0][1][i, j] * t1[i, j])
                Vcs_[1][1][i, j] = C[1][1][i, j]
                C[1][2][i, j] = 2.0 * (Vcs_[1][0][i, j] * t0[i, j] + Vcs_[1][1][i, j] * t1[i, j])

        var tmp = Rfit.MatrixNd[N]()
        for i in range(n):
            for j in range(n):
                var term1 = (
                    Vcs_[0][0][i, j] * Vcs_[0][0][i, j]
                    + Vcs_[0][0][i, j] * Vcs_[0][1][i, j]
                    + Vcs_[1][1][i, j] * Vcs_[1][0][i, j]
                    + Vcs_[1][1][i, j] * Vcs_[1][1][i, j]
                )
                var term2 = (
                    Vcs_[0][0][i, j] * t00[i, j]
                    + Vcs_[0][1][i, j] * t01[i, j]
                    + Vcs_[1][0][i, j] * t10[i, j]
                    + Vcs_[1][1][i, j] * t11[i, j]
                )
                tmp[i, j] = 2.0 * term1 + 4.0 * term2
        C[2][2] = tmp

        comptime if is_defined["RFIT_DEBUG"]():
            Rfit.printIt[RFIT_DEBUG=True](C[0][0], "circle_fit - C[0][0]:")

        # cov matrix of center of gravity (r0.x,r0.y,r0.z)
        var C0 = Rfit.Matrix3d()
        for i in range(3):
            for j in range(i, 3):
                var sum: Float64 = 0.0
                for a in range(n):
                    for b in range(n):
                        sum += weight[a] * C[i][j][a, b] * weight[b]
                # weight.transpose() * C[i][j] * weight;
                C0[i, j] = sum
                C0[j, i] = sum

        comptime if is_defined["RFIT_DEBUG"]():
            Rfit.printIt[RFIT_DEBUG=True](C0, "circle_fit - C0:")

        var W = Rfit.MatrixNd[N]()
        for i in range(n):
            for j in range(n):
                W[i, j] = weight[i] * weight[j]

        var H = Rfit.MatrixNd[N].identity()
        for i in range(n):
            for j in range(n):
                H[i, j] -= weight[j]

        var s_v = H @ p3D.transpose()

        comptime if is_defined["RFIT_DEBUG"]():
            Rfit.printIt[RFIT_DEBUG=True](W, "circle_fit - W:")
            Rfit.printIt[RFIT_DEBUG=True](H, "circle_fit - H:")
            Rfit.printIt[RFIT_DEBUG=True](s_v, "circle_fit - s_v:")

        # cov(s_v)
        var D_ = InlineArray[InlineArray[Rfit.MatrixNd[N], 3], 3](
            fill=InlineArray[Rfit.MatrixNd[N], 3](fill=Rfit.MatrixNd[N]())
        )
        var tmp00 = (H @ C[0][0]) @ H.transpose()
        var tmp01 = (H @ C[0][1]) @ H.transpose()
        var tmp02 = (H @ C[0][2]) @ H.transpose()
        var tmp11 = (H @ C[1][1]) @ H.transpose()
        var tmp12 = (H @ C[1][2]) @ H.transpose()
        var tmp22 = (H @ C[2][2]) @ H.transpose()

        for i in range(n):
            for j in range(n):
                D_[0][0][i, j] = tmp00[i, j] * W[i, j]
                D_[0][1][i, j] = tmp01[i, j] * W[i, j]
                D_[0][2][i, j] = tmp02[i, j] * W[i, j]
                D_[1][1][i, j] = tmp11[i, j] * W[i, j]
                D_[1][2][i, j] = tmp12[i, j] * W[i, j]
                D_[2][2][i, j] = tmp22[i, j] * W[i, j]

        D_[1][0] = D_[0][1].transpose()
        D_[2][0] = D_[0][2].transpose()
        D_[2][1] = D_[1][2].transpose()

        comptime if is_defined["RFIT_DEBUG"]():
            Rfit.printIt[RFIT_DEBUG=True](D_[0][0], "circle_fit - D_[0][0]:")

        comptime nu: InlineArray[InlineArray[UInt32, 2], 6] = [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 1],
            [1, 2],
            [2, 2],
        ]
        # runtime-indexed below; materialized once rather than per access
        var nu_rt = materialize[nu]()

        # cov matrix of the 6 independent elements of A
        var E = Rfit.Matrix6d()
        for a in range(6):
            var i = Int(nu_rt[a][0])
            var j = Int(nu_rt[a][1])
            for b in range(a, 6):
                var k = Int(nu_rt[b][0])
                var l = Int(nu_rt[b][1])

                var t0 = Rfit.VectorNd[N]()
                var t1 = Rfit.VectorNd[N]()

                if l == k:
                    for idx in range(n):
                        var sum0: Float64 = 0.0
                        for m in range(n):
                            sum0 += D_[j][l][idx, m] * s_v[m, l]
                        t0[idx] = 2.0 * sum0
                    if i == j:
                        t1 = t0
                    else:
                        for idx in range(n):
                            var sum1: Float64 = 0.0
                            for m in range(n):
                                sum1 += D_[i][l][idx, m] * s_v[m, l]
                            t1[idx] = 2.0 * sum1
                else:
                    for idx in range(n):
                        var sum0: Float64 = 0.0
                        for m in range(n):
                            sum0 += (
                                D_[j][l][idx, m] * s_v[m, k]
                                + D_[j][k][idx, m] * s_v[m, l]
                            )
                        t0[idx] = sum0
                    if i == j:
                        t1 = t0
                    else:
                        for idx in range(n):
                            var sum1: Float64 = 0.0
                            for m in range(n):
                                sum1 += (
                                    D_[i][l][idx, m] * s_v[m, k]
                                    + D_[i][k][idx, m] * s_v[m, l]
                                )
                            t1[idx] = sum1

                var cm: Float64 = 0.0
                if i == j:
                    for idx in range(n):
                        cm += s_v[idx, i] * (t0[idx] + t1[idx])
                else:
                    for idx in range(n):
                        cm += s_v[idx, i] * t0[idx] + s_v[idx, j] * t1[idx]

                # (s_v.col(i).transpose() * t0) + (s_v.col(j).transpose() * t1);
                E[a, b] = cm
                if b != a:
                    E[b, a] = E[a, b]

        comptime if is_defined["RFIT_DEBUG"]():
            Rfit.printIt[RFIT_DEBUG=True](E, "circle_fit - E:")

        # Jacobian of min_eigen() (numerically computed)
        var J2 = Matrix[DType.float64, 3, 6]()
        for a in range(6):
            var i : Int = Int(nu_rt[a][0])
            var j : Int = Int(nu_rt[a][1])
            var Delta = Rfit.Matrix3d.Zero()
            var delta_val = abs(A[i, j] * Rfit.d)
            Delta[i, j] = delta_val
            Delta[j, i] = delta_val

            var J2_col = min_eigen3D_fast(A + Delta)
            var sign = 1.0 if J2_col[2] > 0.0 else -1.0
            for r in range(3):
                J2_col[r] = (J2_col[r] * sign - v[r]) / delta_val
            for r in range(3):
                J2[r, a] = J2_col[r]

        comptime if is_defined["RFIT_DEBUG"]():
            Rfit.printIt[RFIT_DEBUG=True](J2, "circle_fit - J2:")

        # joint cov matrix of (v0,v1,v2,c)
        var Cvc = Rfit.Matrix4d()
        var t0_cvc = (J2 @ E) @ J2.transpose()
        var t1_cvc = Rfit.Vector3d()
        for r in range(3):
            t1_cvc[r] = -(t0_cvc[r, 0] * r0[0] + t0_cvc[r, 1] * r0[1] + t0_cvc[r, 2] * r0[2])

        for r in range(3):
            for c in range(3):
                Cvc[r, c] = t0_cvc[r, c]
            Cvc[r, 3] = t1_cvc[r]
            Cvc[3, r] = t1_cvc[r]

        var cm1: Float64 = 0.0
        var cm3: Float64 = 0.0
        var cm_sum: Float64 = 0.0
        for i in range(3):
            var row_sum: Float64 = 0.0
            var row_sum_r0: Float64 = 0.0
            for j in range(3):
                row_sum += C0[i, j] * v[j]
                cm_sum += C0[i, j] * t0_cvc[i, j]
                row_sum_r0 += t0_cvc[i, j] * r0[j]
            cm1 += v[i] * row_sum
            cm3 += r0[i] * row_sum_r0

        #      cm2 = (C0.cwiseProduct(t0)).sum();
        Cvc[3, 3] = cm1 + cm_sum + cm3
        # (v.transpose() * C0 * v) + (C0.cwiseProduct(t0)).sum() + (r0.transpose() * t0 * r0);

        comptime if is_defined["RFIT_DEBUG"]():
            Rfit.printIt[RFIT_DEBUG=True](Cvc, "circle_fit - Cvc:")

        # Jacobian (v0,v1,v2,c)->(X0,Y0,R)
        var J3 = Matrix[DType.float64, 3, 4]()
        var t: Float64= 1.0 / h
        J3[0, 0] = -v2x2_inv
        J3[0, 1] = 0.0
        J3[0, 2] = v[0] * Rfit.sqr(v2x2_inv) * 2.0
        J3[0, 3] = 0.0
        J3[1, 0] = 0.0
        J3[1, 1] = -v2x2_inv
        J3[1, 2] = v[1] * Rfit.sqr(v2x2_inv) * 2.0
        J3[1, 3] = 0.0
        J3[2, 0] = v[0] * v2x2_inv * t
        J3[2, 1] = v[1] * v2x2_inv * t
        J3[2, 2] = -h * Rfit.sqr(v2x2_inv) * 2.0 - (2.0 * c + v[2]) * v2x2_inv * t
        J3[2, 3] = -t

        comptime if is_defined["RFIT_DEBUG"]():
            Rfit.printIt[RFIT_DEBUG=True](J3, "circle_fit - J3:")

        # var(q)
        var Jq = Rfit.RowVector2Nd[N]()
        for i in range(2 * n):
            Jq[0, i] = mc[i] * s / Float64(n)

        comptime if is_defined["RFIT_DEBUG"]():
            Rfit.printIt[RFIT_DEBUG=True](Jq, "circle_fit - Jq:")

        # cov(X0,Y0,R)
        var cov_uvr = (J3 @ Cvc) @ J3.transpose()
        var scale = Rfit.sqr(s_inv)
        for i in range(3):
            for j in range(3):
                cov_uvr[i, j] *= scale


        # Maybe optimise if V is a covariance matrix 
        var scalar: Float64 = 0.0
        for i in range(2 * n):
            for j in range(2 * n):
                scalar += Jq[0, i] * V[i, j] * Jq[0, j]


        for i in range(3):
            for j in range(3):
                cov_uvr[i, j] += par_uvr_[i] * par_uvr_[j] * scalar

        circle.cov = cov_uvr

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](circle.cov, "Circle cov:")
        print("circle_fit - exit")

    return circle


#/*!  \brief Perform an ordinary least square fit in the s-z plane to compute
# * the parameters cotTheta and Zip.
# *
# * The fit is performed in the rotated S3D-Z' plane, following the formalism of
# * Frodesen, Chapter 10, p. 259.
# *
# * The system has been rotated to both try to use the combined errors in s-z
# * along Z', as errors in the Y direction and to avoid the patological case of
# * degenerate lines with angular coefficient m = +/- inf.
# *
# * The rotation is using the information on the theta angle computed in the
# * fast fit. The rotation is such that the S3D axis will be the X-direction,
# * while the rotated Z-axis will be the Y-direction. This pretty much follows
# * what is done in the same fit in the Broken Line approach.
# */
def Line_fit[
    M3xN: MatrixLike,
    M6xN: MatrixLike,
    V4: MatrixLike,
](
    hits: M3xN,
    hits_ge: M6xN,
    circle: Rfit.circle_fit,
    fast_fit: V4,
    B: Float64,
    error: Bool,
) -> Rfit.line_fit:
    comptime N: Int = M3xN.ColsAtCompileTime()
    comptime n: Int = N
    var theta = -Float64(circle.q) * math.atan(fast_fit[3].cast[DType.float64]())
    if theta < 0.0:
        theta += math.pi

    # Prepare the Rotation Matrix to rotate the points
    var rot = Rfit.Matrix2d()
    rot[0, 0] = math.sin(theta)
    rot[0, 1] = math.cos(theta)
    rot[1, 0] = -math.cos(theta)
    rot[1, 1] = math.sin(theta)

    # PROJECTION ON THE CILINDER
    #
    # p2D will be:
    # [s1, s2, s3, ..., sn]
    # [z1, z2, z3, ..., zn]
    # s values will be ordinary x-values
    # z values will be ordinary y-values
    var p2D = Rfit.Matrix2xNd[N].Zero()
    var Jx = Matrix[DType.float64, 2, 6]()

    comptime if is_defined["RFIT_DEBUG"]():
        print("Line_fit - B: ", B)
        Rfit.printIt[RFIT_DEBUG=True](hits, "Line_fit points: ")
        Rfit.printIt[RFIT_DEBUG=True](hits_ge, "Line_fit covs: ")
        Rfit.printIt[RFIT_DEBUG=True](rot, "Line_fit rot: ")

    # x & associated Jacobian
    # cfr https://indico.cern.ch/event/663159/contributions/2707659/attachments/1517175/2368189/Riemann_fit.pdf
    # Slide 11
    # a ==> -o i.e. the origin of the circle in XY plane, negative
    # b ==> p i.e. distances of the points wrt the origin of the circle.
    var ox = circle.par[0]
    var oy = circle.par[1]

    # associated Jacobian, used in weights and errors computation
    var Cov = Rfit.Matrix6d.Zero()
    var cov_sz = InlineArray[Rfit.Matrix2d, N](fill=Rfit.Matrix2d())

    for i in range(n):
        var px = hits[0, i].cast[DType.float64]() - ox
        var py = hits[1, i].cast[DType.float64]() - oy
        var p = Rfit.Vector2d()
        p[0] = px
        p[1] = py
        var o_neg = Rfit.Vector2d()
        o_neg[0] = -ox
        o_neg[1] = -oy
        var cross :Float64 = Rfit.cross2D(o_neg, p)
        var dot : Float64 = o_neg[0] * p[0] + o_neg[1] * p[1]
        # atan2(cross, dot) give back the angle in the transverse plane so tha the
        # final equation reads: x_i = -q*R*theta (theta = angle returned by atan2)
        var atan2_ : Float64 = -Float64(circle.q) * math.atan2(cross, dot)
        #    p2D.coeffRef(1, i) = atan2_ * circle.par(2);
        p2D[0, i] = atan2_ * circle.par[2]

        # associated Jacobian, used in weights and errors- computation
        var temp0 :Float64 = -Float64(circle.q) * circle.par[2] * 1.0 / (Rfit.sqr(dot) + Rfit.sqr(cross))
        # good approximation for big pt and eta
        var d_X0 : Float64= 0.0
        var d_Y0 : Float64= 0.0
        var d_R : Float64 = 0.0
        if error:
            d_X0 = -temp0 * ((py + oy) * dot - (px - ox) * cross)
            d_Y0 = temp0 * ((px + ox) * dot - (oy - py) * cross)
            d_R = atan2_
        var d_x : Float64 = temp0 * (oy * dot + ox * cross)
        var d_y : Float64 = temp0 * (-ox * dot + oy * cross)

        Jx[0, 0] = d_X0
        Jx[0, 1] = d_Y0
        Jx[0, 2] = d_R
        Jx[0, 3] = d_x
        Jx[0, 4] = d_y
        Jx[0, 5] = 0.0
        Jx[1, 0] = 0.0
        Jx[1, 1] = 0.0
        Jx[1, 2] = 0.0
        Jx[1, 3] = 0.0
        Jx[1, 4] = 0.0
        Jx[1, 5] = 1.0

        for r in range(3):
            for c in range(3):
                Cov[r, c] = circle.cov[r, c]

        # x errors
        Cov[3, 3] = hits_ge[0, i].cast[DType.float64]()
        # y errors
        Cov[4, 4] = hits_ge[2, i].cast[DType.float64]()
        # z errors
        Cov[5, 5] = hits_ge[5, i].cast[DType.float64]()
        # cov_xy
        Cov[3, 4] = hits_ge[1, i].cast[DType.float64]()
        Cov[4, 3] = hits_ge[1, i].cast[DType.float64]()
        # cov_xz
        Cov[3, 5] = hits_ge[3, i].cast[DType.float64]()
        Cov[5, 3] = hits_ge[3, i].cast[DType.float64]()
        # cov_yz
        Cov[4, 5] = hits_ge[4, i].cast[DType.float64]()
        Cov[5, 4] = hits_ge[4, i].cast[DType.float64]()

        var tmp : Rfit.Matrix2d = (Jx @ Cov) @ Jx.transpose()
        cov_sz[i] = (rot @ tmp) @ rot.transpose()

    # Math of d_{X0,Y0,R,x,y} all verified by hand
    for i in range(n):
        p2D[1, i] = hits[2, i].cast[DType.float64]()

    var s_arcs = Rfit.VectorNd[N]()
    var z_values = Rfit.VectorNd[N]()
    for i in range(n):
        s_arcs[i] = p2D[0, i]
        z_values[i] = p2D[1, i]

    # The following matrix will contain errors orthogonal to the rotated S
    # component only, with the Multiple Scattering properly treated!!
    var cov_with_ms = Rfit.MatrixNd[N]()
    Scatter_cov_line(
        Span(cov_sz),
        fast_fit,
        s_arcs,
        z_values,
        theta,
        B,
        cov_with_ms,
    )

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](cov_sz[0], "line_fit - cov_sz:")
        Rfit.printIt[RFIT_DEBUG=True](cov_with_ms, "line_fit - cov_with_ms: ")

    # Rotate Points with the shape [2, n]
    var p2D_rot : Rfit.Matrix2xNd[N]= rot @ p2D

    comptime if is_defined["RFIT_DEBUG"]():
        print("Fast fit Tan(theta): ", fast_fit[3])
        print("Rotation angle: ", theta)
        Rfit.printIt[RFIT_DEBUG=True](rot, "Rotation Matrix:")
        Rfit.printIt[RFIT_DEBUG=True](p2D, "Original Hits(s,z):")
        Rfit.printIt[RFIT_DEBUG=True](p2D_rot, "Rotated hits(S3D, Z'):")

    var p2D_rot_row1 = Rfit.VectorNd[N]()
    for i in range(n):
        p2D_rot_row1[i] = p2D_rot[1, i]

    # Build the A Matrix
    var A = Rfit.Matrix2xNd[N]()
    for i in range(n):
        A[0, i] = 1.0
        # rotated s values
        A[1, i] = p2D_rot[0, i]

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](A, "A Matrix:")

    # Build A^T V-1 A, where V-1 is the covariance of only the Y components.
    var Vy_inv = Rfit.MatrixNd[N]()
    choleskyInversion.invert(cov_with_ms, Vy_inv)
    # MatrixNd<N> Vy_inv = cov_with_ms.inverse();

    # Compute the Covariance Matrix of the fit parameters
    var Cov_params: Rfit.Matrix2d = (A @ Vy_inv) @ A.transpose()
    var Cov_params_src = Cov_params
    choleskyInversion.invert(Cov_params_src, Cov_params)

    # Now Compute the Parameters in the form [2,1]
    # The first component is q.
    # The second component is m.
    var sol = (Cov_params @ A) @ (Vy_inv @ p2D_rot_row1)

    comptime if is_defined["RFIT_DEBUG"]():
        Rfit.printIt[RFIT_DEBUG=True](sol, "Rotated solutions:")

    # We need now to transfer back the results in the original s-z plane
    var common_factor = 1.0 / (math.sin(theta) - sol[1, 0] * math.cos(theta))
    var J = Rfit.Matrix2d()
    J[0, 0] = 0.0
    J[0, 1] = common_factor * common_factor
    J[1, 0] = common_factor
    J[1, 1] = sol[0, 0] * math.cos(theta) * common_factor * common_factor

    var m = common_factor * (sol[1, 0] * math.sin(theta) + math.cos(theta))
    var q = common_factor * sol[0, 0]
    var cov_mq = (J @ Cov_params) @ J.transpose()

    var res = p2D_rot_row1 - (A.transpose() @ sol)
    var chi2 = (res.transpose() @ (Vy_inv @ res))[0, 0]

    var line = Rfit.line_fit()
    line.par[0] = m
    line.par[1] = q
    line.cov = cov_mq
    line.chi2 = chi2

    comptime if is_defined["RFIT_DEBUG"]():
        print("Common_factor: ", common_factor)
        Rfit.printIt[RFIT_DEBUG=True](J, "Jacobian:")
        Rfit.printIt[RFIT_DEBUG=True](sol, "Rotated solutions:")
        Rfit.printIt[RFIT_DEBUG=True](Cov_params, "Cov_params:")
        Rfit.printIt[RFIT_DEBUG=True](cov_mq, "Rotated Covariance Matrix:")
        Rfit.printIt[RFIT_DEBUG=True](line.par, "Real Parameters:")
        Rfit.printIt[RFIT_DEBUG=True](line.cov, "Real Covariance Matrix:")
        print("Chi2: ", chi2)

    return line


#/*!
#   \brief Helix fit by three step:
#   -fast pre-fit (see Fast_fit() for further info); \n
#   -circle fit of hits projected in the transverse plane by Riemann-Chernov
#       algorithm (see Circle_fit() for further info); \n
#   -line fit of hits projected on cylinder surface by orthogonal distance
#       regression (see Line_fit for further info). \n
#   Points must be passed ordered (from inner to outer layer).
#   \param hits Matrix3xNd hits coordinates in this form: \n
#       |x0|x1|x2|...|xn| \n
#       |y0|y1|y2|...|yn| \n
#       |z0|z1|z2|...|zn|
#   \param hits_cov Matrix3Nd covariance matrix in this form (()->cov()): \n
#  |(x0,x0)|(x1,x0)|(x2,x0)|.|(y0,x0)|(y1,x0)|(y2,x0)|.|(z0,x0)|(z1,x0)|(z2,x0)| \n
#  |(x0,x1)|(x1,x1)|(x2,x1)|.|(y0,x1)|(y1,x1)|(y2,x1)|.|(z0,x1)|(z1,x1)|(z2,x1)| \n
#  |(x0,x2)|(x1,x2)|(x2,x2)|.|(y0,x2)|(y1,x2)|(y2,x2)|.|(z0,x2)|(z1,x2)|(z2,x2)| \n
#      .       .       .    .    .       .       .    .    .       .       .     \n
#  |(x0,y0)|(x1,y0)|(x2,y0)|.|(y0,y0)|(y1,y0)|(y2,x0)|.|(z0,y0)|(z1,y0)|(z2,y0)| \n
#  |(x0,y1)|(x1,y1)|(x2,y1)|.|(y0,y1)|(y1,y1)|(y2,x1)|.|(z0,y1)|(z1,y1)|(z2,y1)| \n
#  |(x0,y2)|(x1,y2)|(x2,y2)|.|(y0,y2)|(y1,y2)|(y2,x2)|.|(z0,y2)|(z1,y2)|(z2,y2)| \n
#      .       .       .    .    .       .       .    .    .       .       .     \n
#  |(x0,z0)|(x1,z0)|(x2,z0)|.|(y0,z0)|(y1,z0)|(y2,z0)|.|(z0,z0)|(z1,z0)|(z2,z0)| \n
#  |(x0,z1)|(x1,z1)|(x2,z1)|.|(y0,z1)|(y1,z1)|(y2,z1)|.|(z0,z1)|(z1,z1)|(z2,z1)| \n
#  |(x0,z2)|(x1,z2)|(x2,z2)|.|(y0,z2)|(y1,z2)|(y2,z2)|.|(z0,z2)|(z1,z2)|(z2,z2)|
#  \param B magnetic field in the center of the detector in Gev/cm/c
#  unit, in order to perform pt calculation.
#  \param error flag for error computation.
#  \param scattering flag for multiple scattering treatment.
#  (see Circle_fit() documentation for further info).
#  \warning see Circle_fit(), Line_fit() and Fast_fit() warnings.
#  \bug see Circle_fit(), Line_fit() and Fast_fit() bugs.
# */
def Helix_fit[
    N: Int,
](
    hits: Rfit.Matrix3xNd[N],
    hits_ge: Matrix[DType.float32, 6, N],
    B: Float64,
    error: Bool,
) -> Rfit.helix_fit:
    var n: Int = N
    var rad = Rfit.VectorNd[N]()
    for i in range(n):
        rad[i] = math.sqrt(hits[0, i] * hits[0, i] + hits[1, i] * hits[1, i])

    # Fast_fit gives back (X0, Y0, R, theta) w/o errors, using only 3 points.
    var fast_fit = Rfit.Vector4d()
    Fast_fit(hits, fast_fit)

    var hits_cov: Rfit.Matrix2Nd[N] = Rfit.Matrix2Nd[N].Zero()
    Rfit.loadCovariance2D(hits_ge, hits_cov)

    var hits2D = Rfit.Matrix2xNd[N]()
    for i in range(n):
        hits2D[0, i] = hits[0, i]
        hits2D[1, i] = hits[1, i]

    var circle = Circle_fit(
        hits2D,
        hits_cov,
        fast_fit,
        rad,
        B,
        error,
    )
    var line = Line_fit(
        hits,
        hits_ge,
        circle,
        fast_fit,
        B,
        error,
    )

    Rfit.par_uvrtopak(circle, B, error)

    var helix = Rfit.helix_fit()
    helix.par[0] = circle.par[0]
    helix.par[1] = circle.par[1]
    helix.par[2] = circle.par[2]
    helix.par[3] = line.par[0]
    helix.par[4] = line.par[1]

    if error:
        helix.cov = Rfit.Matrix5d.Zero()
        for i in range(3):
            for j in range(3):
                helix.cov[i, j] = circle.cov[i, j]
        for i in range(2):
            for j in range(2):
                helix.cov[i + 3, j + 3] = line.cov[i, j]

    helix.q = circle.q
    helix.chi2_circle = circle.chi2
    helix.chi2_line = line.chi2.cast[DType.float32]()

    return helix
