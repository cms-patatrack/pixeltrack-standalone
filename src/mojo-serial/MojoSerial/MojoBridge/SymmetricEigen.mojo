import std.math as math

from MojoSerial.MojoBridge.Matrix import Matrix


from std.math import sqrt, sin, cos, asin, pi

def min_eigen_2x2(
    A: Matrix[DType.float64, 2, 2], mut chi2: Float64
) -> Matrix[DType.float64, 2, 1]:
    var a = A[0, 0]
    var b = A[0, 1]
    var c = A[1, 1]
    var tr = a + c
    var diff = a - c
    var delta = math.sqrt(diff * diff + 4.0 * b * b)
    var lambda_min = 0.5 * (tr - delta)
    chi2 = lambda_min

    var v0 = b
    var v1 = lambda_min - a
    if abs(v0) < 1.0e-12 and abs(v1) < 1.0e-12:
        if a <= c:
            v0 = 1.0
            v1 = 0.0
        else:
            v0 = 0.0
            v1 = 1.0

    var norm = math.sqrt(v0 * v0 + v1 * v1)
    if norm != 0.0:
        v0 /= norm
        v1 /= norm

    var vec = Matrix[DType.float64, 2, 1]()
    vec[0, 0] = v0
    vec[1, 0] = v1
    return vec









# this code is based on the following work https://csm.mech.utah.edu/content/wp-content/uploads/2011/08/ExactEigenSystemForSymmetricTensors.pdf
# the code has comments explaining the math in each section and referencing relevant parts in the pdf mentioned above 


# ---------------------------
# Helpers
# ---------------------------

@always_inline
def clamp(x: Float64, lo: Float64, hi: Float64) -> Float64:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

@always_inline
def norm3(x: Float64, y: Float64, z: Float64) -> Float64:
    return sqrt(x * x + y * y + z * z)

@always_inline
def normalize3(
    x: Float64, y: Float64, z: Float64
) -> Tuple[Float64, Float64, Float64]:
    var n = norm3(x, y, z)
    if n > 0.0:
        return (x / n, y / n, z / n)
    return (1.0, 0.0, 0.0)

@always_inline
def cross3(ax: Float64, ay: Float64, az: Float64,
          bx: Float64, by: Float64, bz: Float64) -> Tuple[Float64, Float64, Float64]:
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx
    )

@always_inline
def best_of_three_vectors(
    v0x: Float64, v0y: Float64, v0z: Float64,
    v1x: Float64, v1y: Float64, v1z: Float64,
    v2x: Float64, v2y: Float64, v2z: Float64
) -> Tuple[Float64, Float64, Float64, Float64]:
    # returns (x,y,z, n2)
    var n0 = v0x*v0x + v0y*v0y + v0z*v0z
    var n1 = v1x*v1x + v1y*v1y + v1z*v1z
    var n2 = v2x*v2x + v2y*v2y + v2z*v2z

    var x = v0x
    var y = v0y
    var z = v0z
    var nmax = n0

    if n1 > nmax:
        nmax = n1
        x = v1x; y = v1y; z = v1z
    if n2 > nmax:
        nmax = n2
        x = v2x; y = v2y; z = v2z

    return (x, y, z, nmax)

@always_inline
def nullvec_from_rows_rank12(
    # rows r0,r1,r2 of M (each is 3-vector)
    r0x: Float64, r0y: Float64, r0z: Float64,
    r1x: Float64, r1y: Float64, r1z: Float64,
    r2x: Float64, r2y: Float64, r2z: Float64
) -> Tuple[Float64, Float64, Float64]:
    # ============================================================
    # Math: For Mv=0 in 3D:
    # - If rank(M)=2 (simple eigenvalue), nullspace is 1D.
    #   Any cross product of two independent rows lies in null(M).
    # - If rank(M)=1 (double eigenvalue for λ), nullspace is 2D:
    #   rows are all multiples of one row r, so nullspace is all v ⟂ r.
    #   We can get one by v = r × e_i for some basis vector e_i not parallel to r.
    # ============================================================

    # Try null vector via cross products of row pairs (rank-2 case).
    var (c01x, c01y, c01z) = cross3(r0x, r0y, r0z, r1x, r1y, r1z)
    var (c02x, c02y, c02z) = cross3(r0x, r0y, r0z, r2x, r2y, r2z)
    var (c12x, c12y, c12z) = cross3(r1x, r1y, r1z, r2x, r2y, r2z)

    var (vx, vy, vz, nmax) = best_of_three_vectors(
        c01x, c01y, c01z,
        c02x, c02y, c02z,
        c12x, c12y, c12z
    )

    if nmax > 1.0e-24:
        return normalize3(vx, vy, vz)

    # If all cross products are tiny, rows are nearly parallel (rank ~ 1),
    # or everything is tiny (near MMM). Pick a nonzero row and return a vector ⟂ to it.
    var n0 = r0x*r0x + r0y*r0y + r0z*r0z
    var n1 = r1x*r1x + r1y*r1y + r1z*r1z
    var n2 = r2x*r2x + r2y*r2y + r2z*r2z

    var rx = r0x
    var ry = r0y
    var rz = r0z
    var rn = n0
    if n1 > rn:
        rn = n1
        rx = r1x; ry = r1y; rz = r1z
    if n2 > rn:
        rn = n2
        rx = r2x; ry = r2y; rz = r2z

    # If even the biggest row is ~0, M~0 => MMM-ish => any vector
    if rn <= 1.0e-24:
        return (1.0, 0.0, 0.0)

    # Choose basis vector least aligned with r to avoid tiny cross product.
    # If |rx| is smallest, use e_x, etc.
    if abs(rx) <= abs(ry) and abs(rx) <= abs(rz):
        # v = r × e_x = (0, rz, -ry)
        return normalize3(0.0, rz, -ry)
    elif abs(ry) <= abs(rz):
        # v = r × e_y = (-rz, 0, rx)
        return normalize3(-rz, 0.0, rx)
    else:
        # v = r × e_z = (ry, -rx, 0)
        return normalize3(ry, -rx, 0.0)

# ---------------------------
# Main: smallest eigenpair of symmetric 3x3 using Brannon invariants / Lode angle
# ---------------------------

def min_eigen_3x3(
    A: Matrix[DType.float64, 3, 3],
    mut chi2: Float64
) -> Matrix[DType.float64, 3, 1]:

    # ============================================================
    # Section 0: Read symmetric components
    # A = [[a00,a01,a02],[a01,a11,a12],[a02,a12,a22]]
    # ============================================================
    var a00 = A[0, 0]
    var a01 = A[0, 1]
    var a02 = A[0, 2]
    var a11 = A[1, 1]
    var a12 = A[1, 2]
    var a22 = A[2, 2]

    # ============================================================
    # Section 1: Deviatoric part S = A - (tr(A)/3) I   (Eq. 21.99)
    # Let I1 = tr(A). Then q = I1/3 and S = A - q I.
    #
    # A and S share eigenvectors; eigenvalues satisfy:
    #   λ_k = s_k + I1/3   (Eq. 21.100)
    # ============================================================
    var I1 = a00 + a11 + a22
    var q = I1 / 3.0

    var s00 = a00 - q
    var s11 = a11 - q
    var s22 = a22 - q
    var s01 = a01
    var s02 = a02
    var s12 = a12

    # ============================================================
    # Section 2: Compute J2 and r = ||S|| = sqrt(2 J2)
    #
    # For deviatoric S:
    #   J2 = (1/2) tr(S^2)
    #   r  = sqrt(2 J2) = ||S||  (Eq. 21.108a, 21.121)
    #
    # For symmetric matrices, tr(S^2) expands to:
    #   s00^2+s11^2+s22^2 + 2(s01^2+s02^2+s12^2)
    # ============================================================
    var trS2 = (
        s00*s00 + s11*s11 + s22*s22
        + 2.0*(s01*s01 + s02*s02 + s12*s12)
    )
    var J2 = 0.5 * trS2
    var r = sqrt(2.0 * J2)

    # If r ~ 0 => S ~ 0 => A ~ q I => triple root (MMM), any vector works.
    # (Page 398: if r=0, one distinct eigenvalue)
    if r <= 1.0e-18:
        chi2 = q
        var v = Matrix[DType.float64, 3, 1]()
        v[0, 0] = 1.0
        v[1, 0] = 0.0
        v[2, 0] = 0.0
        return v

    # ============================================================
    # Section 3: Compute sin(3θ) from det(S_hat)
    #
    # Let S_hat = S / r.
    # Then:
    #   sin(3θ) = 3 * sqrt(6) * det(S_hat)   (Eq. 21.108b)
    #
    # We use the principal arcsin to enforce θ ∈ [-π/6, π/6],
    # which guarantees ordered eigenvalues λL ≤ λM ≤ λH.
    # ============================================================
    var sh00 = s00 / r
    var sh11 = s11 / r
    var sh22 = s22 / r
    var sh01 = s01 / r
    var sh02 = s02 / r
    var sh12 = s12 / r

    # det of symmetric 3x3:
    # det = s00*s11*s22 + 2*s01*s02*s12 - s00*s12^2 - s11*s02^2 - s22*s01^2
    var det_sh = (
        sh00 * sh11 * sh22
        + 2.0 * sh01 * sh02 * sh12
        - sh00 * sh12 * sh12
        - sh11 * sh02 * sh02
        - sh22 * sh01 * sh01
    )

    var sin3t = 3.0 * sqrt(6.0) * det_sh
    sin3t = clamp(sin3t, -1.0, 1.0)

    # θ in principal range ensures ordering (discussion around Eq. 21.111)
    var theta = asin(sin3t) / 3.0

    # ============================================================
    # Section 4: Eigenvalues from (r, θ, z)
    #
    # In the PDF, z = A : I_hat with I_hat = I/√3, so z = tr(A)/√3.
    # Then z/√3 = tr(A)/3 = q.
    #
    # Closed-form ordered eigenvalues (Eq. 21.111):
    #   λH = q + sqrt(2/3) r cos(θ - π/6)
    #   λM = q - sqrt(2/3) r sin(θ)
    #   λL = q - sqrt(2/3) r cos(θ + π/6)
    # ============================================================
    var k = sqrt(2.0 / 3.0) * r
    var lamH = q + k * cos(theta - pi / 6.0)
    var lamM = q - k * sin(theta)
    var lamL = q - k * cos(theta + pi / 6.0)

    # By construction lamL <= lamM <= lamH, so the minimum is lamL
    chi2 = lamL

    # ============================================================
    # Section 5: Direct eigenvector from nullspace of M = A - λL I
    # Solve M v = 0 by row-cross-product method with rank-1 fallback.
    # ============================================================
    var m00 = a00 - lamL
    var m01 = a01
    var m02 = a02
    var m10 = a01
    var m11 = a11 - lamL
    var m12 = a12
    var m20 = a02
    var m21 = a12
    var m22 = a22 - lamL

    # rows of M
    var r0x = m00; var r0y = m01; var r0z = m02
    var r1x = m10; var r1y = m11; var r1z = m12
    var r2x = m20; var r2y = m21; var r2z = m22

    var (vx, vy, vz) = nullvec_from_rows_rank12(
        r0x, r0y, r0z,
        r1x, r1y, r1z,
        r2x, r2y, r2z
    )

    var vec = Matrix[DType.float64, 3, 1]()
    vec[0, 0] = vx
    vec[1, 0] = vy
    vec[2, 0] = vz
    return vec


def min_eigen_3x3_fast(
    A: Matrix[DType.float64, 3, 3]
) -> Matrix[DType.float64, 3, 1]:
    var chi2: Float64 = 0.0
    return min_eigen_3x3(A, chi2)
