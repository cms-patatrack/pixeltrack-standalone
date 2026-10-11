from std.math import sqrt

# | 1) circle is parameterized as:                                              |
# |    C*[(X-Xp)**2+(Y-Yp)**2] - 2*alpha*(X-Xp) - 2*beta*(Y-Yp) = 0             |
# |    Xp,Yp is a point on the track;                                           |
# |    C = 1/r0 is the curvature  ( sign of C is charge of particle );          |
# |    alpha & beta are the direction cosines of the radial vector at Xp,Yp     |
# |    i.e.  alpha = C*(X0-Xp),                                                 |
# |          beta  = C*(Y0-Yp),                                                 |
# |    where center of circle is at X0,Y0.                                      |
# |                                                                             |
# |    Slope dy/dx of tangent at Xp,Yp is -alpha/beta.                          |
# | 2) the z dimension of the helix is parameterized by gamma = dZ/dSperp       |
# |    this is also the tangent of the pitch angle of the helix.                |
# |    with this parameterization, (alpha,beta,gamma) rotate like a vector.     |
# | 3) For tracks going inward at (Xp,Yp), C, alpha, beta, and gamma change sign|


struct CircleEq[T: DType](Copyable, Movable):
    var m_xp: Scalar[Self.T]
    var m_yp: Scalar[Self.T]
    var m_c: Scalar[Self.T]
    var m_alpha: Scalar[Self.T]
    var m_beta: Scalar[Self.T]

    @always_inline
    def __init__(out self):
        self.m_xp = 0
        self.m_yp = 0
        self.m_c = 0
        self.m_alpha = 0
        self.m_beta = 0

    @always_inline
    def __init__(
        out self,
        x1: Scalar[Self.T],
        y1: Scalar[Self.T],
        x2: Scalar[Self.T],
        y2: Scalar[Self.T],
        x3: Scalar[Self.T],
        y3: Scalar[Self.T],
    ):
        self.m_xp = 0
        self.m_yp = 0
        self.m_c = 0
        self.m_alpha = 0
        self.m_beta = 0
        self.compute(x1, y1, x2, y2, x3, y3)

    def compute(
        mut self,
        x1: Scalar[Self.T],
        y1: Scalar[Self.T],
        x2: Scalar[Self.T],
        y2: Scalar[Self.T],
        x3: Scalar[Self.T],
        y3: Scalar[Self.T],
    ):
        var noflip = abs(x3 - x1) < abs(y3 - y1)

        var x1p = x1 - x2 if noflip else y1 - y2
        var y1p = y1 - y2 if noflip else x1 - x2
        var d12 = x1p * x1p + y1p * y1p
        var x3p = x3 - x2 if noflip else y3 - y2
        var y3p = y3 - y2 if noflip else x3 - x2
        var d32 = x3p * x3p + y3p * y3p

        var num = x1p * y3p - y1p * x3p  # num also gives correct sign for CT
        var det = d12 * y3p - d32 * y1p

        var st2 = d12 * x3p - d32 * x1p
        var seq = det * det + st2 * st2
        var al2 = Scalar[Self.T](1.0) / sqrt(seq)
        var be2 = -st2 * al2
        var ct = Scalar[Self.T](2.0) * num * al2
        al2 *= det

        self.m_xp = x2
        self.m_yp = y2
        self.m_c = ct if noflip else -ct
        self.m_alpha = al2 if noflip else -be2
        self.m_beta = be2 if noflip else -al2

    # dca to origin divided by curvature
    def dca0(self) -> Scalar[Self.T]:
        var x = self.m_c * self.m_xp + self.m_alpha
        var y = self.m_c * self.m_yp + self.m_beta
        return sqrt(x * x + y * y) - Scalar[Self.T](1)

    # dca to given point (divided by curvature)
    def dca(self, x: Scalar[Self.T], y: Scalar[Self.T]) -> Scalar[Self.T]:
        var xx = self.m_c * (self.m_xp - x) + self.m_alpha
        var yy = self.m_c * (self.m_yp - y) + self.m_beta
        return sqrt(xx * xx + yy * yy) - Scalar[Self.T](1)

    # curvature
    def curvature(self) -> Scalar[Self.T]:
        return self.m_c

    # alpha and beta
    def cosdir(self) -> Tuple[Scalar[Self.T], Scalar[Self.T]]:
        return (self.m_alpha, self.m_beta)

    # alpha and beta at given point
    def cosdir(self, x: Scalar[Self.T], y: Scalar[Self.T]) -> Tuple[Scalar[Self.T], Scalar[Self.T]]:
        return (
            self.m_alpha - self.m_c * (x - self.m_xp),
            self.m_beta - self.m_c * (y - self.m_yp),
        )

    # center
    def center(self) -> Tuple[Scalar[Self.T], Scalar[Self.T]]:
        return (
            self.m_xp + self.m_alpha / self.m_c,
            self.m_yp + self.m_beta / self.m_c,
        )

    def radius(self) -> Scalar[Self.T]:
        return Scalar[Self.T](1) / self.m_c
