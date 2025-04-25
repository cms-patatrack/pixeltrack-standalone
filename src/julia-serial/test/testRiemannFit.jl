using ..RecoPixelVertexing_PixelTrackFitting_interface_RiemannFit_h
# using ..RecoPixelVertexing_PixelTrackFitting_interface_BrokenLine_h
using LinearAlgebra
using StaticArrays

const MAX_NUMBER_OF_TRACKS = 5 * 1024

function stride()
    return MAX_NUMBER_OF_TRACKS
end

function fill_hits_and_hitscov(hits::Matrix{Float64}, hits_ge::Matrix{Float32})
    n = size(hits, 2)
    if n == 5
        hits[:, 1] .= [2.934787, 0.773211, -10.980247]
        hits[:, 2] .= [6.314229, 1.816356, -23.162731]
        hits[:, 3] .= [8.936963, 2.765734, -32.759060]
        hits[:, 4] .= [10.360559, 3.330824, -38.061260]
        hits[:, 5] .= [12.856387, 4.422212, -47.518867]

        hits_ge[:, 1] .= [1.424715e-07, -4.996975e-07, 1.752614e-06, 3.660689e-11, 1.644638e-09, 7.346080e-05]
        hits_ge[:, 2] .= [6.899177e-08, -1.873414e-07, 5.087101e-07, -2.078806e-10, -2.210498e-11, 4.346079e-06]
        hits_ge[:, 3] .= [1.406273e-06, 4.042467e-07, 6.391180e-07, -3.141497e-07, 6.513821e-08, 1.163863e-07]
        hits_ge[:, 4] .= [1.176358e-06, 2.154100e-07, 5.072816e-07, -8.161219e-08, 1.437878e-07, 5.951832e-08]
        hits_ge[:, 5] .= [2.852843e-05, 7.956492e-06, 3.117701e-06, -1.060541e-06, 8.777413e-09, 1.426417e-07]

        return
    end

    if n > 3
        hits[:, 1] .= [1.98645, 2.18002, 2.46338]
        hits[:, 2] .= [4.72598, 4.88864, 6.99838]
        hits[:, 3] .= [7.65632, 7.75845, 11.808]
        hits[:, 4] .= [11.3151, 11.3134, 17.793]
    else
        hits[:, 1] .= [1.98645, 2.18002, 2.46338]
        hits[:, 2] .= [4.72598, 4.88864, 6.99838]
        hits[:, 3] .= [7.65632, 7.75845, 11.808]
    end

    hits_ge[1, 1] = 7.14652014722e-06
    hits_ge[1, 2] = 2.15789009417e-06
    hits_ge[1, 3] = 1.63328002145e-06

    if n > 3
        hits_ge[1, 4] = 6.27919007457e-06
    end

    hits_ge[3, 1] = 6.10347979091e-06
    hits_ge[3, 2] = 2.08211008612e-06
    hits_ge[3, 3] = 1.61672005561e-06

    if n > 3
        hits_ge[3, 4] = 6.28080988463e-06
    end

    hits_ge[6, 1] = 5.184e-05
    hits_ge[6, 2] = 1.444e-05
    hits_ge[6, 3] = 6.25e-06

    if n > 3
        hits_ge[6, 4] = 3.136e-05
    end

    hits_ge[2, 1] = -5.60076978218e-06
    hits_ge[2, 2] = -1.11936003577e-06
    hits_ge[2, 3] = -6.24945016625e-07

    if n > 3
        hits_ge[2, 4] = -5.28000009581e-06
    end
end
function block_colwise_norm(hits::Matrix{Float64}, N::Int)

    block = hits[1:2, 1:N]
    return vec([norm(block[:, i]) for i in 1:N])

end

function testFit(N::Int)
    B = 0.0113921
    hits = zeros(Float64, 3, N)  # Initialize a 3xN matrix of Float64 zeros
    hits_ge = zeros(Float32, 6, N)  # Initialize a 6xN matrix of Float32 zeros

    fill_hits_and_hitscov(hits, hits_ge)  # Pass N to the function

    # println("sizes: ", N, " ", sizeof(hits), " ", sizeof(hits_ge), " ", sizeof(Vector{Float64}))

    println("Generated hits:")
    println(hits)

    println("Generated cov:")
    println(hits_ge)

    # FAST_FIT_CPU
    fast_fit_results = zeros(Float64, 4)
    Main.RecoPixelVertexing_PixelTrackFitting_interface_RiemannFit_h.fast_fit(hits, fast_fit_results)

    println("Fitted values (FastFit, [X0, Y0, R, tan(theta)]):\n")
    println(fast_fit_results)

    # CIRCLE_FIT CPU
    rad = zeros(Float64, N)
    rad = block_colwise_norm(hits, N)
    hits_cov = zeros(Float64, 2 * N, 2 * N)
    Main.RecoPixelVertexing_PixelTrackFitting_interface_FitUtils_h.loadCovariance2D(hits_ge, hits_cov)

    println("hits_cov: ", hits_cov)

    circle_fit_results = Main.RecoPixelVertexing_PixelTrackFitting_interface_RiemannFit_h.circle_fit(hits[1:2, 1:N], hits_cov, fast_fit_results, rad, B, true)

    println(circle_fit_results)

    # line_fit_results = Main.RecoPixelVertexing_PixelTrackFitting_interface_RiemannFit_h.line_fit(hits, hits_ge, circle_fit_results, rad, B, true)
    # println(line_fit_results)
end

testFit(4)
# testFit(3)
# testFit(5)