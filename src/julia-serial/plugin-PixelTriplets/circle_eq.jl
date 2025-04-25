"""
| 1) circle is parameterized as:                                              |
|    C*[(X-Xp)**2+(Y-Yp)**2] - 2*alpha*(X-Xp) - 2*beta*(Y-Yp) = 0             |
|    Xp,Yp is a point on the track;                                           |
|    C = 1/r0 is the curvature  ( sign of C is charge of particle );          |
|    alpha & beta are the direction cosines of the radial vector at Xp,Yp     |
|    i.e.  alpha = C*(X0-Xp),                                                 |
|          beta  = C*(Y0-Yp),                                                 |
|    where center of circle is at X0,Y0.                                      |
|                                                                             |
|    Slope dy/dx of tangent at Xp,Yp is -alpha/beta.                          |
| 2) the z dimension of the helix is parameterized by gamma = dZ/dSperp       |
|    this is also the tangent of the pitch angle of the helix.                |
|    with this parameterization, (alpha,beta,gamma) rotate like a vector.     |
| 3) For tracks going inward at (Xp,Yp), C, alpha, beta, and gamma change sign|
|
"""
struct CircleEq{T}
    m_xp::T
    m_yp::T
    m_c::T
    m_alpha::T
    m_beta::T
    function CircleEq{T}(x1::T,y1::T,x2::T,y2::T,x3::T,y3::T) where T
        no_flip::Bool = abs(x3-x1) < abs(y3-y1)
        x1p = no_flip ? x1 - x2 : y1 - y2
        y1p = no_flip ? y1 - y2 : x1 - x2
        d12 = x1p * x1p + y1p * y1p
        x3p = no_flip ? x3 - x2 : y3 - y2
        y3p = no_flip ? y3 - y2 : x3 - x2
        d32 = x3p * x3p + y3p * y3p
        num = x1p * y3p - y1p * x3p  # num also gives correct sign for CT
        det = d12 * y3p - d32 * y1p
        st2 = (d12 * x3p - d32 * x1p)
        seq = det * det + st2 * st2
        al2 = T(1.) / √(seq)
        be2 = -st2 * al2
        ct = T(2.) * num * al2
        al2 *= det
        m_xp = x2
        m_yp = y2
        m_c =  no_flip ? ct : -ct
        m_alpha = no_flip ? al2 : -be2
        m_beta = no_flip ? be2 : -al2
        new(m_xp,m_yp,m_c,m_alpha,m_beta)
    end
end



function compute(self::CircleEq{T},x1::T,y1::T,x2::T,y2::T,x3::T,y3::T) where T <: AbstractFloat
    no_flip::Bool = abs(x3-x1) < abs(y3-y1)
    x1p = no_flip ? x1 - x2 : y1 - y2
    y1p = no_flip ? y1 - y2 : x1 - x2
    d12 = x1p * x1p + y1p * y1p
    x3p = no_flip ? x3 - x2 : y3 - y2
    y3p = no_flip ? y3 - y2 : x3 - x2
    d32 = x3p * x3p + y3p * y3p
    num = x1p * y3p - y1p * x3p  # num also gives correct sign for CT
    det = d12 * y3p - d32 * y1p
    st2 = (d12 * x3p - d32 * x1p)
    seq = det * det + st2 * st2
    al2 = T(1.) / √(seq)
    be2 = -st2 * al2
    ct = T(2.) * num * al2
    al2 *= det
    self.m_xp = x2
    self.m_yp = y2
    self.m_c =  no_flip ? ct : -ct
    self.m_alpha = no_flip ? al2 : -be2
    self.m_beta = no_flip ? be2 : -al2
    return nothing
end

curvature(self::CircleEq{T}) where T = self.m_c
cos_dir(self::CircleEq{T}) where T = (self.m_alpha,self.m_beta)

cos_dir(self::CircleEq{T},x::T,y::T) where T = (m_alpha - m_c*(x-m_xp),m_beta - m_c * (y - m_yp))

center(self::CircleEq{T}) where T = (self.m_xp + (self.m_alpha/self.m_c),self.m_yp + (self.m_beta/m_c))
radius(self::CircleEq{T}) where T = T(1)/m_c
# distance of closest approach
function dca(self::CircleEq{T},x::T,y::T) where T
    x = self.m_c * (self.m_xp - x) + self.m_alpha
    y = self.m_c * (self.m_yp - y) + self.m_beta
    return  √(x^2 + y^2) - T(1)
end

function dca0(self::CircleEq{T}) where T
    x = self.m_c * self.m_xp + self.m_alpha
    y = self.m_c * self.m_yp + self.m_beta
    return  √(x^2 + y^2) - T(1)
end
