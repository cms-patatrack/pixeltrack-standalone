module test
using Random

function printIt(m::AbstractMatrix{T}) where T
    println("\nMatrix $(size(m, 1))x$(size(m, 2))")
    for r in 1:size(m, 1)
        for c in 1:size(m, 2)
            println("Matrix($r,$c) = $(m[r, c])")
        end
    end
end

function isEqualFuzzy(a::AbstractMatrix, b::AbstractMatrix, epsilon::Float64 = 1e-6)
    for i in 1:size(a, 1)
        for j in 1:size(a, 2)
            @assert abs(a[i, j] - b[i, j]) < min(abs(a[i, j]), abs(b[i, j])) * epsilon
        end
    end
    return true
end

function isEqualFuzzy(a::Float64, b::Float64, epsilon::Float64 = 1e-6)
    return abs(a - b) < min(abs(a), abs(b)) * epsilon
end

function fillMatrix!(t::AbstractMatrix{T}) where T
    rng = MersenneTwister(RandomDevice())
    for row in 1:size(t, 1)
        for col in 1:size(t, 2)
            t[row, col] = rand(rng) * 2.0 
        end
    end
end

end
