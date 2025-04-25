module RecoPixelVertexing_PixelTrackFitting_interface_BrokenLine_h

using ..RecoPixelVertexing_PixelTrackFitting_interface_FitUtils_h: circle_fit, line_fit, cross2D
using LinearAlgebra
using Statistics
using Test
using StaticArrays
using ..DataFormat_Math_choleskyInversion_h

#  Karimäki's parameters: (phi, d, k=1/R)
# /*!< covariance matrix: \n
#   |cov(phi,phi)|cov( d ,phi)|cov(  k ,phi)| \n
#   |cov(phi, d )|cov( d , d )|cov( k , d )| \n
#   |cov(phi, k )|cov( d , k )|cov( k , k )|

# Brief data needed for the Broken Line fit procedure.
mutable struct PreparedBrokenLineData
    q::Int                           # Particle charge
    radii::Matrix{Float64}           # Matrix: xy data in the system where the pre-fitted center is the origin
    s::Vector{Float64}               # Vector: total distance traveled in the transverse plane from the pre-fitted closest approach
    S::Vector{Float64}               # Vector: total distance traveled (three-dimensional)
    Z::Vector{Float64}               # Vector: orthogonal coordinate to the pre-fitted line in the sz plane
    VarBeta::Vector{Float64}         # Vector: kink angles in the SZ plane

    function PreparedBrokenLineData(q::Int64, radii::Matrix{Float64}, s::Vector{Float64}, S::Vector{Float64}, Z::Vector{Float64}, VarBeta::Vector{Float64})
        new(q, radii, s, S, Z, VarBeta)
    end
end


export PreparedBrokenLineData


# \brief Computes the Coulomb multiple scattering variance of the planar angle.

#     \param length length of the track in the material.
#     \param B magnetic field in Gev/cm/c.
#     \param R radius of curvature (needed to evaluate p).
#     \param Layer denotes which of the four layers of the detector is the endpoint of the multiple scattered track. For example, if Layer=3, then the particle has just gone through the material between the second and the third layer.

#     \todo add another Layer variable to identify also the start point of the track, so if there are missing hits or multiple hits, the part of the detector that the particle has traversed can be exactly identified.

#     \warning the formula used here assumes beta=1, and so neglects the dependence of theta_0 on the mass of the particle at fixed momentum.

#     \return the variance of the planar angle ((theta_0)^2 /3).
@inline function mult_scatt(length::Float64, B::Float64, R::Float64, Layer::Integer, slope::Float64)
    # open("debug_output.txt", "a") do f
        # println(f, "B: ", B)
        # println(f, "R: ", R)
        pt2 = min(20.0, B * R)
        # println(f, "pt2: ", pt2)
        pt2 = pt2 * pt2
        # println(f, "pt2 squared: ", pt2)

        XXI_0 = 0.06 / 16.0 # inverse of radiation length of the material in cm
        # println(f, "XXI_0: ", XXI_0)

        geometry_factor = 0.7
        fact = geometry_factor * (13.6 / 1000)^2
        # println(f, "fact: ", fact)

        slope_term = (1 + slope^2)
        # println(f, "slope_term: ", slope_term)

        abs_length_XXI_0 = abs(length) * XXI_0
        # println(f, "abs_length_XXI_0: ", abs_length_XXI_0)

        log_term = log(abs_length_XXI_0)
        # println(f, "log_term: ", log_term)

        sqrt_term = (1 + 0.038 * log_term)^2
        # println(f, "sqrt_term: ", sqrt_term)

        result = fact / (pt2 * slope_term) * abs_length_XXI_0 * sqrt_term
        # println(f, "output of mult_scatt: ", result)

        return result
    # end
end


sqr(x) = x * x

# \brief Computes the 2D rotation matrix that transforms the line y=slope*x into the line y=0.

#     \param slope tangent of the angle of rotation.

#     \return 2D rotation matrix.
@inline function rotation_matrix(slope::Float64)
    a = 1.0 / sqrt(1.0 + slope^2)
    b = slope * a
    # construct a 2×2 SMatrix which is stack‑allocated and unrolled by the compiler
    return @SMatrix [ a   b
                     -b   a ]
end

# \brief Changes the Karimäki parameters (and consequently their covariance matrix) under a translation of the coordinate system, such that the old origin has coordinates (x0,y0) in the new coordinate system. The formulas are taken from Karimäki V., 1990, Effective circle fitting for particle trajectories, Nucl. Instr. and Meth. A305 (1991) 187.

#     \param circle circle fit in the old coordinate system.
#     \param x0 x coordinate of the translation vector.
#     \param y0 y coordinate of the translation vector.
#     \param jacobian passed by reference in order to save stack.
@inline function translate_karimaki(circle::circle_fit, x0::Float64, y0::Float64, jacobian::AbstractArray{Float64})
    DP = x0 * cos(circle.par[1]) + y0 * sin(circle.par[1])
    DO = x0 * sin(circle.par[1]) - y0 * cos(circle.par[1]) + circle.par[2]
    uu = 1 + circle.par[3] * circle.par[2]
    C = -circle.par[3] * y0 + uu * cos(circle.par[1])
    BB = circle.par[3] * x0 + uu * sin(circle.par[1])
    A = 2.0 * DO + circle.par[3] * (DO^2 + DP^2)
    U = sqrt(1.0 + circle.par[3] * A)
    xi = 1.0 / (BB^2 + C^2)
    v = 1.0 + circle.par[3] * DO
    lambda = (0.5 * A) / (U * (1.0 + U)^2)
    mu = 1.0 / (U * (1.0 + U)) + circle.par[3] * lambda
    zeta = DO^2 + DP^2

    jacobian[1, 1] = xi * uu * v
    jacobian[1, 2] = -xi * circle.par[3]^2 * DP
    jacobian[1, 3] = xi * DP

    jacobian[2, 1] = 2.0 * mu * uu * DP
    jacobian[2, 2] = 2.0 * mu * v
    jacobian[2, 3] = mu * zeta - lambda * A

    jacobian[3, 1] = 0.0
    jacobian[3, 2] = 0.0
    jacobian[3, 3] = 1.0

    circle.par[1] = atan2(BB, C)
    circle.par[2] = A / (1.0 + U)
    circle.cov = jacobian * circle.cov * jacobian'
end

# \brief Computes the data needed for the Broken Line fit procedure that are mainly common for the circle and the line fit.

#     \param hits hits coordinates.
#     \param hits_cov hits covariance matrix.
#     \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
#     \param B magnetic field in Gev/cm/c.
#     \param results PreparedBrokenLineData to be filled (see description of PreparedBrokenLineData).
#   */

@inline function compute_e(f::AbstractVector{Float64})
    x, y = f[1], f[2]
    r    = sqrt(x*x + y*y)         # norm
    s    = -f[3] / r               # scalar factor
    return SVector{2,Float64}(s*x, s*y)          # one SVector stack allocation, no heap
end

@inline function prepare_broken_line_data(
    hits::AbstractMatrix{Float64},
    fast_fit::AbstractVector{Float64},
    B::Float64,
    results::PreparedBrokenLineData
)
    n = size(hits, 2)
    # 1) hoist params
    f1, f2, f3, f4 = fast_fit[1], fast_fit[2], fast_fit[3], fast_fit[4]

    # 2) d,e,q
    x1 = hits[1,2] - hits[1,1]
    y1 = hits[2,2] - hits[2,1]
    xn = hits[1,n] - hits[1,n-1]
    yn = hits[2,n] - hits[2,n-1]
    results.q = (x1*yn - y1*xn) > 0 ? -1 : 1

    # 3) rotation
    slope = -results.q / f4
    R     = rotation_matrix(slope)   # SMatrix{2,2,Float64}

    # 4) radii
    @inbounds for j in 1:n
        results.radii[1,j] = hits[1,j] - f1
        results.radii[2,j] = hits[2,j] - f2
    end

    # 5) s
    ev = compute_e(fast_fit)
    e1, e2 = ev[1], ev[2]
    fac     = results.q * f3

    @inbounds for i in 1:n
        xi = results.radii[1,i]
        yi = results.radii[2,i]
        c  = xi*e2 - yi*e1
        d  = xi*e1 + yi*e2
        results.s[i] = fac * atan2(c, d)
    end

    # 6) S, Z
    @inbounds for i in 1:n
        xi = results.s[i]
        yi = hits[3,i]
        results.S[i] = R[1,1]*xi + R[1,2]*yi
        results.Z[i] = R[2,1]*xi + R[2,2]*yi
    end

    # 7) VarBeta
    vb       = results.VarBeta
    vb[1], vb[n] = 0, 0
    for i in 2:n-1
        Δ1 = results.S[i+1] - results.S[i]
        Δ2 = results.S[i]   - results.S[i-1]
        vb[i] = mult_scatt(Δ1, B, f3, i+2, slope) +
                mult_scatt(Δ2, B, f3, i+1, slope)
    end
end


# \brief Computes the n-by-n band matrix obtained minimizing the Broken Line's cost function w.r.t u. This is the whole matrix in the case of the line fit and the main n-by-n block in the case of the circle fit.

#     \param w weights of the first part of the cost function, the one with the measurements and not the angles (\sum_{i=1}^n w*(y_i-u_i)^2).
#     \param S total distance traveled by the particle from the pre-fitted closest approach.
#     \param VarBeta kink angles' variance.

#     \return the n-by-n matrix of the linear system
@inline function matrixc_u(w::AbstractArray{Float64}, S::Vector{Float64}, VarBeta::Vector{Float64},::Val{N}) where N
    n = N
    C_U = @MArray zeros(Float64, N, N)
    for i in 1:n
        C_U[i, i] = w[i]

        if i > 2
            C_U[i, i] += 1.0 / (VarBeta[i-1] * (S[i] - S[i-1])^2)
        end

        if i > 1 && i < n
            C_U[i, i] += (1.0 / VarBeta[i]) * ((S[i+1] - S[i-1]) / ((S[i+1] - S[i]) * (S[i] - S[i-1])))^2
        end

        if i < n - 1
            C_U[i, i] += 1.0 / (VarBeta[i+1] * (S[i+1] - S[i])^2)
        end

        if i > 1 && i < n
            C_U[i, i+1] = 1.0 / (VarBeta[i] * (S[i+1] - S[i])) * (-(S[i+1] - S[i-1]) / ((S[i+1] - S[i]) * (S[i] - S[i-1])))
        end

        if i < n - 1
            C_U[i, i+1] += 1.0 / (VarBeta[i+1] * (S[i+1] - S[i])) * (-(S[i+2] - S[i]) / ((S[i+2] - S[i+1]) * (S[i+1] - S[i])))
        end

        if i < n - 1
            C_U[i, i+2] = 1.0 / (VarBeta[i+1] * (S[i+2] - S[i+1]) * (S[i+1] - S[i]))
        end

        C_U[i, i] *= 0.5
    end
    return C_U + C_U'
end

function squaredNorm(v::AbstractVector{T}) where {T}
    return sum(v .^ 2)
end

function atan2(y::Float64, x::Float64)
    if x > 0
        return atan(y / x)
    elseif x < 0
        return atan(y / x) + π
    elseif y > 0 && x == 0
        return π / 2  # 90 degrees
    elseif y < 0 && x == 0
        return -π / 2 # -90 degrees
    else
        return NaN    # undefined for (0, 0)
    end
end


# \brief A very fast helix fit.  
#     \param hits the measured hits.
#     \return (X0,Y0,R,tan(theta)).
#     \warning sign of theta is (intentionally, for now) mistaken for negative charges.
@inline function BL_Fast_fit(hits, results)
    n = size(hits, 2)

    mid = (n >> 1) + 1

    h11 = hits[1, 1];  h21 = hits[2, 1]
    h1m = hits[1, mid]; h2m = hits[2, mid]
    h1n = hits[1, n];  h2n = hits[2, n]
   

    a1 = h1m - h11;  a2 = h2m - h21
    b1 = h1n - h1m;  b2 = h2n - h2m
    c1 = h11 - h1n;  c2 = h21 - h2n

    ca =   c1*a2 - c2*a1           # cross2D(c, a)
    ba =   b1*a2 - b2*a1           # cross2D(b, a)
    nc =   c1*c1 + c2*c2           # squaredNorm(c)
    na =   a1*a1 + a2*a2           # squaredNorm(a)
    nb =   b1*b1 + b2*b2           # squaredNorm(b)

    tmp = 0.5 / ca
    results[1] = h11 - (a2*nc + c2*na) * tmp
    results[2] = h21 + (a1*nc + c1*na) * tmp

    results[3] = sqrt(na * nb * nc) / (2.0 * abs(ba))

    cx = results[1];  cy = results[2]
    d1 = h11 - cx;  d2 = h21 - cy
    e1 = h1n - cx;  e2 = h2n - cy

    de    = d1*e2 - d2*e1          # cross2D(d, e)
    dotde = d1*e1 + d2*e2          # dot(d, e)
    dz    = hits[3, n] - hits[3, 1]

    results[4] = results[3] * atan2(de, dotde) / dz

    return nothing

end

# \brief Performs the Broken Line fit in the curved track case (that is, the fit parameters are the interceptions u and the curvature correction \Delta\kappa).

#     \param hits hits coordinates.
#     \param hits_cov hits covariance matrix.
#     \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
#     \param B magnetic field in Gev/cm/c.
#     \param data PreparedBrokenLineData.
#     \param circle_results struct to be filled with the results in this form:
#     -par parameter of the line in this form: (phi, d, k); \n
#     -cov covariance matrix of the fitted parameter; \n
#     -chi2 value of the cost function in the minimum.

#     \details The function implements the steps 2 and 3 of the Broken Line fit with the curvature correction.\n
#     The step 2 is the least square fit, done by imposing the minimum constraint on the cost function and solving the consequent linear system. It determines the fitted parameters u and \Delta\kappa and their covariance matrix.
#     The step 3 is the correction of the fast pre-fitted parameters for the innermost part of the track. It is first done in a comfortable coordinate system (the one in which the first hit is the origin) and then the parameters and their covariance matrix are transformed to the original coordinate system.
@inline function BL_Circle_fit(hits, hits_ge, fast_fit, B::Float64, data::PreparedBrokenLineData, circle_results::circle_fit,::Val{N}) where N
    # open("debug_output.txt", "a") do f
        n = size(hits, 2)
        circle_results.q = data.q
        radii = data.radii
        s = data.s
        S = data.S
        Z = data.Z
        VarBeta = data.VarBeta
        slope = -circle_results.q / fast_fit[4]

        # println(f, "VarBeta before mult with slope: ", VarBeta)
        VarBeta = VarBeta * (1.0 + (slope)^2)
        # println(f, "VarBeta after mult with slope: ", VarBeta)

        for i in 1:n
            Z[i] = norm((radii[1, i],radii[2, i])) - fast_fit[3]
        end

        V = @MArray zeros(Float64, 2, 2)
        w = @MArray zeros(Float64, N, 1)
        RR = @MArray zeros(Float64, 2, 2)
        tol = 1e-12
        
        # abs(hits_ge[2, i]) < tol ? 0.0 : hits_ge[2, i]
        for i in 1:n
            V[1, 1] = abs(hits_ge[1, i]) < tol ? 0.0 : hits_ge[1, i]
            V[1, 2] = V[2, 1] = abs(hits_ge[2, i]) < tol ? 0.0 : hits_ge[2, i]
            V[2, 2] = abs(hits_ge[3, i]) < tol ? 0.0 : hits_ge[3, i]
            RR = rotation_matrix(-radii[1, i] / radii[2, i])
            w[i] = 1.0 / ((RR*V*RR')[2, 2])
            # println(f, "Iteration $i:")
            # println(f, "  V = ", V)
            # println(f, "  RR = ", RR)
            # println(f, "  w[$i] = ", w[i])
        end
        
        r_u = @MArray zeros(Float64, N + 1)
        r_u[n+1] = 0
        for i in 1:n
            r_u[i] = w[i] * Z[i]
        end

        C_U = @MArray zeros(Float64, N + 1, N + 1)
        C_U[1:n, 1:n] = matrixc_u(w, s, VarBeta,Val(N))
        C_U[n+1, n+1] = 0.0
        
        for i in 1:n
            C_U[i, n+1] = 0.0

            if i > 1 && i < n
                C_U[i, n+1] += -(s[i+1] - s[i-1]) * (s[i+1] - s[i-1]) / (2.0 * VarBeta[i] * (s[i+1] - s[i]) * (s[i] - s[i-1]))
            end

            if i > 2
                C_U[i, n+1] += (s[i] - s[i-2]) / (2.0 * VarBeta[i-1] * (s[i] - s[i-1]))
            end

            if i < n - 1
                C_U[i, n+1] += (s[i+2] - s[i]) / (2.0 * VarBeta[i+1] * (s[i+1] - s[i]))
            end

            C_U[n+1, i] = C_U[i, n+1]

            if i > 1 && i < n
                C_U[n+1, n+1] += (s[i+1] - s[i-1])^2 / (4.0 * VarBeta[i])
            end
        end
        
        I = @MArray zeros(Float64, N + 1, N + 1)
        DataFormat_Math_choleskyInversion_h.invert(C_U, I)
        u = I * r_u
        # println("u: ", u)
        # println("I: ", I)
        # println("r_u: ", r_u)
        
        # for column 1
        t = norm((radii[1,1], radii[2,1]))
        radii[1,1] /= t
        radii[2,1] /= t

        # for column 2
        t = norm((radii[1,2], radii[2,2]))
        radii[1,2] /= t
        radii[2,2] /= t

        
        tmp1 = -Z[1] + u[1]
        d1   = hits[1,1] + tmp1 * radii[1,1]
        d2   = hits[2,1] + tmp1 * radii[2,1]

        tmp2 = -Z[2] + u[2]
        e1   = hits[1,2] + tmp2 * radii[1,2]
        e2   = hits[2,2] + tmp2 * radii[2,2]
        d = @SVector [d1,d2]
        e = @SVector [e1,e2]
        
        # println("d: ", d)
        # println("e: ", e)

        # println("e-d: ", (e - d))
        # println("atan2((e-d)[2], (e-d)[1]): ", atan2((e-d)[2], (e-d)[1]))
        circle_results.par[1] = atan2((e-d)[2], (e-d)[1])
        circle_results.par[2] = -circle_results.q * (fast_fit[3] - sqrt(sqr(fast_fit[3]) - 0.25 * norm(e - d)^2))
        circle_results.par[3] = circle_results.q * (1.0 / fast_fit[3] + u[n+1])
        # println("fast_fit[3]: ", fast_fit[3])
        # println("circle_results.q:", circle_results.q)
        # println("hits:  ", hits[1:2, 2])
        # println("norm(e - d)^2: ", norm(e - d)^2) #issue here

        # println("circle_results.par[2]: ", circle_results.par[2])
        # println("circle_results.q * circle_results.par[2]:", circle_results.q * circle_results.par[2])
        # println("Initial circle_results.par:", circle_results.par)
        
        @assert circle_results.q * circle_results.par[2] <= 0

        eMinusd = e - d
        tmp1 = sqr(norm(eMinusd))
        # println("tmp1: ", tmp1)
        # println("radii: ", radii)
        # println("fast_fit: ", fast_fit)
        jacobian = @MArray zeros(Float64, 3, 3)

        
        jacobian[1, 1] = (radii[2, 1] * eMinusd[1] - eMinusd[2] * radii[1, 1]) / tmp1
        jacobian[1, 2] = (radii[2, 2] * eMinusd[1] - eMinusd[2] * radii[1, 2]) / tmp1
        jacobian[1, 3] = 0
        jacobian[2, 1] = floor(circle_results.q / 2) * (eMinusd[1] * radii[1, 1] + eMinusd[2] * radii[2, 1]) /
                         sqrt(sqr(2 * fast_fit[3]) - tmp1)
        jacobian[2, 2] = floor(circle_results.q / 2) * (eMinusd[1] * radii[1, 2] + eMinusd[2] * radii[2, 2]) /
                         sqrt(sqr(2 * fast_fit[3]) - tmp1)
        jacobian[2, 3] = 0
        jacobian[3, 1] = 0
        jacobian[3, 2] = 0
        jacobian[3, 3] = circle_results.q
        
        circle_results.cov = @MArray [
            I[1, 1] I[1, 2] I[1, n+1];
            I[2, 1] I[2, 2] I[2, n+1];
            I[n+1, 1] I[n+1, 2] I[n+1, n+1]
        ]
        # println("circle_cov 1: ", circle_results.cov)
        # println("jacobian: ", jacobian)
        
        circle_results.cov = jacobian * circle_results.cov * jacobian'
        # println("circle_cov 2: ", circle_results.cov)

         translate_karimaki(circle_results, 0.5 * (e-d)[1], 0.5 * (e-d)[2], jacobian)
        # println("circle_cov 3: ", circle_results.cov)
        
        circle_results.cov[1, 1] += (1 + sqr(slope)) * mult_scatt(S[2] - S[1], B, fast_fit[3], 2, slope)
        # println("mult_scatt(S[2] - S[1], B, fast_fit[3], 2, slope): ", mult_scatt(S[2] - S[1], B, fast_fit[3], 2, slope))

        translate_karimaki(circle_results, d[1], d[2], jacobian)
        
        # println(f, " VarBeta: ", VarBeta)
        # println(f, "  Z = ", Z)
        # println(f, "  s = ", s)
        # println(f, "  u = ", u)
        # println("circle_cov 4: ", circle_results.cov)
        # open("chi_debug.txt", "a") do d
            # println(d, "New chi2 calculation")
            circle_results.chi2 = 0.0
            for i in 1:n
                diff = Float64(Z[i] - u[i])
                term1 = Float64(w[i] * (diff^2))
                circle_results.chi2 += term1
                # print(d, "i = $i | term1 = $term1 | chi2 = $(circle_results.chi2)\n")

                if i > 1 && i < n
                    d1 = Float64(s[i] - s[i-1])     # denominator part 1
                    d2 = Float64(s[i+1] - s[i])     # denominator part 2
                    d3 = Float64(s[i+1] - s[i-1])   # combined span

                    term2_part1 = u[i-1] / d1
                    term2_part2 = (u[i] * d3) / (d1 * d2)
                    term2_part3 = u[i+1] / d2
                    term2_part4 = (d3 * u[n+1]) / 2.0
                    # println(d, "u = ", u[n+1])
                    # print(d, "i = $i | term2_part1 = $term2_part1\n")
                    # print(d, "i = $i | term2_part2 = $term2_part2\n")
                    # print(d, "i = $i | term2_part3 = $term2_part3\n")
                    # print(d, "i = $i | term2_part4 = $term2_part4\n")

                    term2_inner = term2_part1 - term2_part2 + term2_part3 + term2_part4
                    term2 = (term2_inner^2) / VarBeta[i]
                    circle_results.chi2 += term2

                    # print(d, "i = $i | d1 = $d1 | d2 = $d2 | d3 = $d3\n")
                    # print(d, "i = $i | term2_inner = $term2_inner | term2 = $term2 | chi2 = $(circle_results.chi2)\n")
                end
            end
            # println(d, "circle.chi2 = ", circle_results.chi2)
        # end
        # println(f, "circle.chi2 = ", circle_results.chi2)
    # end
            
    return circle_results.cov
end
# /*!
# \brief Performs the Broken Line fit in the straight track case (that is, the fit parameters are only the interceptions u).

# \param hits hits coordinates.
# \param hits_cov hits covariance matrix.
# \param fast_fit pre-fit result in the form (X0,Y0,R,tan(theta)).
# \param B magnetic field in Gev/cm/c.
# \param data PreparedBrokenLineData.
# \param line_results struct to be filled with the results in this form:
# -par parameter of the line in this form: (cot(theta), Zip); \n
# -cov covariance matrix of the fitted parameter; \n
# -chi2 value of the cost function in the minimum.

# \details The function implements the steps 2 and 3 of the Broken Line fit without the curvature correction.\n
# The step 2 is the least square fit, done by imposing the minimum constraint on the cost function and solving the consequent linear system. It determines the fitted parameters u and their covariance matrix.
# The step 3 is the correction of the fast pre-fitted parameters for the innermost part of the track. It is first done in a comfortable coordinate system (the one in which the first hit is the origin) and then the parameters and their covariance matrix are transformed to the original coordinate system.
# */
@inline function BL_Line_fit(hits_ge, fast_fit, B::Float64, data::PreparedBrokenLineData, line_results::line_fit,::Val{N}) where N 
    # open("debug_output.txt", "a") do f
        n = size(hits_ge, 2)
        radii = data.radii
        S = data.S
        Z = data.Z
        VarBeta = data.VarBeta

        slope = -data.q / fast_fit[4]
        
        R = rotation_matrix(slope)
        # println("R BL_Line_Fit: ", R)

            # ——— Pre‑allocate once, before the loop ———
        V       = @MArray zeros(Float64, 3, 3)   # your 3×3 covariance
        Jacob   = @MArray zeros(Float64, 2, 3)   # your 2×3 JacobXYZtosZ
        w       = @MArray zeros(Float64, N)      # collects the weights

        # buffers for the product R*J*V*J'*R'
        M1      = @MArray zeros(Float64, 2, 3)   # R * J       → 2×3
        M2      = @MArray zeros(Float64, 2, 3)   # M1 * V      → 2×3
        M3      = @MArray zeros(Float64, 2, 2)   # M2 * J'     → 2×2
        M4      = @MArray zeros(Float64, 2, 2)   # M3 * R'     → 2×2

        tol     = 1e-12

        for i in 1:n
            # — fill V and JacobXYZtosZ exactly as before —
            V[1, 1] = hits_ge[1, i]
            V[1, 2] = V[2, 1] = abs(hits_ge[2, i]) < tol ? 0.0 : hits_ge[2, i]
            V[1, 3] = V[3, 1] = abs(hits_ge[4, i]) < tol ? 0.0 : hits_ge[4, i]
            V[2, 2] = hits_ge[3, i]
            V[2, 3] = V[3, 2] = abs(hits_ge[5, i]) < tol ? 0.0 : hits_ge[5, i]
            V[3, 3] = hits_ge[6, i]

            tmp = 1 / norm((radii[1,i], radii[2,i]))
            Jacob[1,1] =  radii[2,i] * tmp
            Jacob[1,2] = -radii[1,i] * tmp
            Jacob[2,3] = 1.0

            # — now the chained product, in‑place ——
            mul!(M1, R,        Jacob      )   # 2×2 * 2×3 → 2×3
            mul!(M2, M1,       V          )   # 2×3 * 3×3 → 2×3
            mul!(M3, M2,       transpose(Jacob)) # 2×3 * 3×2 → 2×2
            mul!(M4, M3,       transpose(R)   ) # 2×2 * 2×2 → 2×2

            w[i] = 1.0 / M4[2,2]
        end
        
        # println("JacobXYZtosZ: ", JacobXYZtosZ)
        # println("V:", V)
        # println("w:", w)

        r_u = @MArray zeros(N)
        for i in 1:n
            r_u[i] = w[i] * Z[i]
        end
        
        # println("matrixc_u(w, S, VarBeta): ", matrixc_u(w, S, VarBeta))

        I = @MArray zeros(Float64, N, N)
        DataFormat_Math_choleskyInversion_h.invert(matrixc_u(w, S, VarBeta,Val(N)), I)
        
        # println("I: ", I)
        u = I * r_u
        
        # println("u: ", u)
        line_results.par =  @MArray [(u[2] - u[1]) / (S[2] - S[1]), u[1]]
        
        # println("line_results.par: ", line_results.par)
        idiff = 1.0 / (S[2] - S[1])
        

        line_results.cov = @MArray [
            (I[1, 1]-2*I[1, 2]+I[2, 2])*idiff^2+mult_scatt(S[2] - S[1], B, fast_fit[3], 2, slope) (I[1, 2]-I[1, 1])*idiff;
            (I[1, 2]-I[1, 1])*idiff I[1, 1]
        ]
        
        
        jacobian = @MArray zeros(2, 2)
        jacobian[1, 1] = 1
        jacobian[1, 2] = 0
        jacobian[2, 1] = -S[1]
        jacobian[2, 2] = 1
        
        line_results.par[2] = line_results.par[2] - line_results.par[1] * S[1]
        # scratch = @MMatrix zeros(Float64, 2, 2)
        line_results.cov = jacobian * line_results.cov * jacobian'
        # mul!(scratch, jacobian, line_results.cov)       
        # mul!(line_results.cov, scratch, transpose(jacobian))
        tmp = R[1, 1] - line_results.par[1] * R[1, 2]
        jacobian[2, 2] = 1 / tmp
        jacobian[1, 1] = jacobian[2, 2] * jacobian[2, 2]
        jacobian[1, 2] = 0
        jacobian[2, 1] = line_results.par[2] * R[1, 2] * jacobian[1, 1]
        line_results.par[2] = line_results.par[2] * jacobian[2, 2]
        line_results.par[1] = (R[1, 2] + line_results.par[1] * R[1, 1]) * jacobian[2, 2]
        line_results.cov = jacobian * line_results.cov * jacobian'
        # mul!(scratch, jacobian, line_results.cov)       
        # mul!(line_results.cov, scratch, transpose(jacobian))
        

        line_results.chi2 = 0
        for i in 1:n
            # Compute the primary term: w[i] * (Z[i] - u[i])^2
            diff = Z[i] - u[i]
            term1 = w[i] * diff^2
            # println(f, "Iteration $i:")
            # println(f, "  w[$i] = ", w[i])
            # println(f, "  Z[$i] = ", Z[i])
            # println(f, "  u[$i] = ", u[i])
            # println(f, "  diff (Z-u) = ", diff)
            # println(f, "  term1 = w[$i]*(Z[$i]-u[$i])^2 = ", term1)
            # flush(f)

            line_results.chi2 += term1

            if i > 1 && i < n
                # Define denominators used in the extra term calculation
                denom1 = S[i] - S[i-1]
                denom2 = S[i+1] - S[i]
                denom3 = S[i] - S[i-1]  # same as denom1 for clarity
                denom4 = S[i+1] - S[i-1]

                # Compute the additional term's inner value:
                term2_inner = (u[i-1] / denom1) -
                              (u[i] * (denom4 / (denom2 * denom3))) +
                              (u[i+1] / (S[i+1] - S[i]))
                term2 = (term2_inner^2) / VarBeta[i]

                # println(f, "  Additional term calculation:")
                # println(f, "    u[", i - 1, "] = ", u[i-1], " , S[$i] - S[", i - 1, "] = ", denom1)
                # println(f, "    u[$i] = ", u[i], " , (S[", i + 1, "] - S[", i - 1, "]) = ", denom4)
                # println(f, "    S[", i + 1, "] - S[$i] = ", denom2)
                # println(f, "    u[", i + 1, "] = ", u[i+1])
                # println(f, "    term2_inner = ", term2_inner)
                # println(f, "    VarBeta[$i] = ", VarBeta[i])
                # println(f, "    term2 = (term2_inner^2)/VarBeta[$i] = ", term2)
                # flush(f)

                line_results.chi2 += term2
            end

            # println(f, "  Chi2 after iteration $i: ", line_results.chi2)
            # println(f, "------------------------------------------------")
            # flush(f)
        end
    # end
    
    return line_results
end

# \brief Helix fit by three step:
# -fast pre-fit (see Fast_fit() for further info); \n
# -circle fit of the hits projected in the transverse plane by Broken Line algorithm (see BL_Circle_fit() for further info); \n
# -line fit of the hits projected on the (pre-fitted) cilinder surface by Broken Line algorithm (see BL_Line_fit() for further info); \n
# Points must be passed ordered (from inner to outer layer).

# \param hits Matrix3xNd hits coordinates in this form: \n
# |x1|x2|x3|...|xn| \n
# |y1|y2|y3|...|yn| \n
# |z1|z2|z3|...|zn|
# \param hits_cov Matrix3Nd covariance matrix in this form (()->cov()): \n
# |(x1,x1)|(x2,x1)|(x3,x1)|(x4,x1)|.|(y1,x1)|(y2,x1)|(y3,x1)|(y4,x1)|.|(z1,x1)|(z2,x1)|(z3,x1)|(z4,x1)| \n
# |(x1,x2)|(x2,x2)|(x3,x2)|(x4,x2)|.|(y1,x2)|(y2,x2)|(y3,x2)|(y4,x2)|.|(z1,x2)|(z2,x2)|(z3,x2)|(z4,x2)| \n
# |(x1,x3)|(x2,x3)|(x3,x3)|(x4,x3)|.|(y1,x3)|(y2,x3)|(y3,x3)|(y4,x3)|.|(z1,x3)|(z2,x3)|(z3,x3)|(z4,x3)| \n
# |(x1,x4)|(x2,x4)|(x3,x4)|(x4,x4)|.|(y1,x4)|(y2,x4)|(y3,x4)|(y4,x4)|.|(z1,x4)|(z2,x4)|(z3,x4)|(z4,x4)| \n
# .       .       .       .       . .       .       .       .       . .       .       .       .       . \n
# |(x1,y1)|(x2,y1)|(x3,y1)|(x4,y1)|.|(y1,y1)|(y2,y1)|(y3,x1)|(y4,y1)|.|(z1,y1)|(z2,y1)|(z3,y1)|(z4,y1)| \n
# |(x1,y2)|(x2,y2)|(x3,y2)|(x4,y2)|.|(y1,y2)|(y2,y2)|(y3,x2)|(y4,y2)|.|(z1,y2)|(z2,y2)|(z3,y2)|(z4,y2)| \n
# |(x1,y3)|(x2,y3)|(x3,y3)|(x4,y3)|.|(y1,y3)|(y2,y3)|(y3,x3)|(y4,y3)|.|(z1,y3)|(z2,y3)|(z3,y3)|(z4,y3)| \n
# |(x1,y4)|(x2,y4)|(x3,y4)|(x4,y4)|.|(y1,y4)|(y2,y4)|(y3,x4)|(y4,y4)|.|(z1,y4)|(z2,y4)|(z3,y4)|(z4,y4)| \n
# .       .       .    .          . .       .       .       .       . .       .       .       .       . \n
# |(x1,z1)|(x2,z1)|(x3,z1)|(x4,z1)|.|(y1,z1)|(y2,z1)|(y3,z1)|(y4,z1)|.|(z1,z1)|(z2,z1)|(z3,z1)|(z4,z1)| \n
# |(x1,z2)|(x2,z2)|(x3,z2)|(x4,z2)|.|(y1,z2)|(y2,z2)|(y3,z2)|(y4,z2)|.|(z1,z2)|(z2,z2)|(z3,z2)|(z4,z2)| \n
# |(x1,z3)|(x2,z3)|(x3,z3)|(x4,z3)|.|(y1,z3)|(y2,z3)|(y3,z3)|(y4,z3)|.|(z1,z3)|(z2,z3)|(z3,z3)|(z4,z3)| \n
# |(x1,z4)|(x2,z4)|(x3,z4)|(x4,z4)|.|(y1,z4)|(y2,z4)|(y3,z4)|(y4,z4)|.|(z1,z4)|(z2,z4)|(z3,z4)|(z4,z4)|
# \param B magnetic field in the center of the detector in Gev/cm/c, in order to perform the p_t calculation.

# \warning see BL_Circle_fit(), BL_Line_fit() and Fast_fit() warnings.

# \bug see BL_Circle_fit(), BL_Line_fit() and Fast_fit() bugs.

# \return (phi,Tip,p_t,cot(theta)),Zip), their covariance matrix and the chi2's of the circle and line fits.
@inline function BL_Helix_fit(hits::Matrix{Float64}, hits_ge::Matrix{Float64}, B::Float64)

    helix = helix_fit(zeros(5), zeros(5, 5), 0.0, 0.0, 1)
    fast_fit = Vector{Float64}(undef, 4)
    BL_Fast_fit(hits, fast_fit)

    data = PreparedBrokenLineData(
        0,
        zeros(3, 3),
        zeros(3),
        zeros(3),
        zeros(3),
        zeros(3)
    )
    circle = circle_fit(
        zeros(3),
        zeros(3, 3),
        0,
        0.0
    )
    line = line_fit(
        zeros(2),
        zeros(2, 2),
        0.0
    )
    jacobian = zeros(3, 3)
    prepare_broken_line_data(hits, fast_fit, B, data)
    BL_Line_fit(hits_ge, fast_fit, B, data, line)
    BL_Circle_fit(hits, hits_ge, fast_fit, B, data, circle)

    jacobian .= [1.0 0 0;
        0 1.0 0;
        0 0 -abs(circle.par[3])*B/(circle.par[3]^2*circle.par[3])]

    circle.par[3] = B / abs(circle.par[3])
    circle.cov = jacobian * circle.cov * jacobian'

    helix.par = vcat(circle.par, line.par)
    helix.cov = zeros(5, 5)
    helix.cov[1:3, 1:3] .= circle.cov
    helix.cov[4:5, 4:5] .= line.cov
    helix.q = circle.q
    helix.chi2_circle = circle.chi2
    helix.chi2_line = line.chi2


    return helix
end


end