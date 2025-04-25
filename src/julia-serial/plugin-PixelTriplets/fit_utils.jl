module RecoPixelVertexing_PixelTrackFitting_interface_FitUtils_h

using StaticArrays
using ..RecoPixelVertexing_PixelTrackFitting_interface_FitResult_h
export cross2D, printIt, sqr
function printIt(m::C, prefix::String="") where {C}
    for r in 1:size(m, 1)
        for c in 1:size(m, 2)
            println("$prefix Matrix($(r-1),$(c-1)) = $(m[r, c])")
        end
    end
end

function sqr(a::T) where {T}
    return a .* a
end


function cross2D(a::AbstractVector{Float64}, b::AbstractVector{Float64})
    return a[1] * b[2] - a[2] * b[1]
end

function loadCovariance2D(ge::Matrix{Float32}, hits_cov::Matrix{Float64})
    hits_in_fit = size(ge, 2)


    for i in 1:hits_in_fit
        ge_idx = 1
        j = 1
        l = 1
        hits_cov[i, i] = ge[ge_idx, i]

        ge_idx = 3
        j = 2
        l = 2
        hits_cov[i+hits_in_fit, i+hits_in_fit] = ge[ge_idx, i]

        ge_idx = 2
        j = 2
        l = 1
        hits_cov[i, i+hits_in_fit] = hits_cov[i+hits_in_fit, i] = ge[ge_idx, i]
    end
end



function loadCovariance(ge::M6xNf, hits_cov::M2Nd) where {M6xNf,M2Nd}
    hits_in_fit = size(M6xNf, 2)
    for i in 1:hits_in_fit
        ge_idx = 1
        j = 1
        l = 1
        column_i = ge[:, i]
        element = column_i[ge_idx]
        hits_cov[i+j*hits_in_fit, i+l*hits_in_fit] = element

        ge_idx = 3
        j = 2
        l = 2
        element = column_i[ge_idx]
        hits_cov[i+j*hits_in_fit, i+l*hits_in_fit] = element

        ge_idx = 6
        j = 3
        l = 3
        element = column_i[ge_idx]
        hits_cov[i+j*hits_in_fit, i+l*hits_in_fit] = element

        ge_idx = 2
        j = 2
        l = 3
        element = column_i[ge_idx]
        hits_cov[i+l*hits_in_fit, i+j*hits_in_fit] = hits_cov[i+j*hits_in_fit, i+l*hits_in_fit] = element

        ge_idx = 4
        j = 3
        l = 1
        element = column_i[ge_idx]
        hits_cov[i+l*hits_in_fit, i+j*hits_in_fit] = hits_cov[i+j*hits_in_fit, i+l*hits_in_fit] = element

        ge_idx = 5
        j = 3
        l = 2
        element = column_i[ge_idx]
        hits_cov[i+l*hits_in_fit, i+j*hits_in_fit] = hits_cov[i+j*hits_in_fit, i+l*hits_in_fit] = element
    end
end

@inline function par_uvrtopak(circle::circle_fit, B::Float64, error::Bool)
    par_pak = Vector{Float64}(undef, 3)
    temp0 = circle.par.head(2).squaredNorm()
    temp1 = sqrt(temp0)
    par_pak[1] = atan2(circle.q * circle.par[1], -circle.q * circle.par[2])
    par_pak[2] = circle.q * (temp1 - circle.par[3])
    par_pak[3] = circle.par[3] * B

    if (error)
        temp2 = sqr(circle.par[1]) * 1.0 / temp0
        temp3 = 1.0 / temp1 * circle.q
        J4 = Matrix{Float64}(undef, 3, 3)
        J4[1, 1] = -circle.par[1] * temp2 / circle.par[1]^2
        J4[1, 2] = temp2 / circle.par[1]
        J4[1, 3] = 0.0
        J4[2, 1] = circle.par[1] * temp3
        J4[2, 2] = circle.par[2] * temp3
        J4[2, 3] = -circle.q
        J4[3, 1] = 0.0
        J4[3, 2] = 0.0
        J4[3, 3] = B
        circle.cov = J4 * circle.cov * transpose(J4)
    end
    circle.par = par_pak
end

@inline function fromCircleToPerigee(circle::circle_fit)
    par_pak = Vector{Float64}(undef, 3)
    temp0 = circle.pat.head(2).squaredNorm()
    temp1 = sqrt(temp0)

    par_pak[1] = atan2(circle.q * circle.par[1], -circle.q * circle.par[2])
    par_pak[2] = circle.q * (temp1 - circle.par[3])
    par_pak[3] = circle.par[3] * B

    temp2 = sqr(circle.par[1]) * 1.0 / temp0
    temp3 = 1.0 / temp1 * circle.q
    J4 = Matrix{Float64}(undef, 3, 3)

    J4[1, 1] = -circle.par[1] * temp2 / circle.par[1]^2
    J4[1, 2] = temp2 / circle.par[1]
    J4[1, 3] = 0.0
    J4[2, 1] = circle.par[1] * temp3
    J4[2, 2] = circle.par[2] * temp3
    J4[2, 3] = -circle.q
    J4[3, 1] = 0.0
    J4[3, 2] = 0.0
    J4[3, 3] = -circle.q / (circle.par[3]^2)

    circle.cov = J4 * circle.cov * transpose(J4)

    circle.par = par_pak
end

@inline function transformToPerigeePlane(ip::VI5, icov::MI5, op::VO5, ocov::MO5) where {VI5,MI5,VO5,MO5}
    sinTheta2 = 1 / 1 + ip[4] * ip[4]
    sinTheta = sqrt(sinTheta2)
    cosTheta = ip[4] * sinTheta

    op[1] = sinTheta * ip[3]
    op[2] = 0
    op[3] = -ip[4]
    op[4] = ip[2]
    op[5] = -ip[5]

    J = Matrix{Float64}(undef, 5, 5)
    J[1, 3] = sinTheta
    J[1, 4] = -sinTheta2 * cosTheta * ip[3]
    J[2, 1] = 1
    J[3, 4] = -1
    J[4, 2] = 1
    J[5, 5] = -1

    ocov = J * icov * transpose(J)
end


end