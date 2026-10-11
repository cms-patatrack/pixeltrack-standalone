module CUDADataFormatsTrackTrajectoryStateSOA_H

using StaticArrays
const Vector5f = SVector{5,Float32}
const Vector15f = SVector{15,Float32}
const Vector5d = SVector{5,Float64}
const Matrix5d = SMatrix{5,5,Float64}
struct TrajectoryStateSoA
   state::Matrix{Float32}
   covariance::Matrix{Float32}

   function TrajectoryStateSoA()
      new(zeros(Float32, 0, 5), zeros(Float32, 0, 15))
   end

   # Not initialised: the state of a track is written by its fit, and only read for the tracks that are fitted,
   # as in the C++.
   function TrajectoryStateSoA(n::Int)
      new(Matrix{Float32}(undef, n, 5), Matrix{Float32}(undef, n, 15))
   end
end
function copyFromCircle!(trajectory::TrajectoryStateSoA,
   cp::AbstractVector{Float64}, ccov::AbstractArray{Float64},
   lp::AbstractVector{Float64}, lcov::AbstractArray{Float64},
   b::Float64, i::UInt16)

   # static vectors rather than arrays, so that nothing is allocated for each track
   trajectory.state[i, :] .= Float32.(SVector(cp[1], cp[2], cp[3] * b, lp[1], lp[2]))

   trajectory.covariance[i, :] .= Float32.(SVector(
      ccov[1, 1],
      ccov[1, 2],
      b * ccov[1, 3],
      0.0,
      0.0,
      ccov[2, 2],
      b * ccov[2, 3],
      0.0,
      0.0,
      b^2 * ccov[3, 3],
      0.0,
      0.0,
      lcov[1, 1],
      lcov[1, 2],
      lcov[2, 2]
   ))
end


export copyFromCircle!
end
