module RecoPixelVertexing_PixelTrackFitting_plugins_HelixFitOnGPU_h

using ..CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h: TrackingRecHit2DSOAView
using ..Tracks: HitContainer, TrackSOA
using ..histogram: OneToManyAssoc
using ..RecoPixelVertexing_PixelTrackFitting_interface_FitResult_h
using ..caConstants: TupleMultiplicity
using LinearAlgebra

# Namespace equivalent for Rfit
module Rfit

# Max number of concurrent fits
@inline function max_number_of_concurrent_fits()
    return 24 * 1024
end

# Stride for concurrent fits
@inline function stride()
    return max_number_of_concurrent_fits()
end

# Equivalent of C++ templates for matrices
function Matrix3xNd(N::Int)
    return Matrix{Float64}(undef, 3, N)
end

function Matrix6xNf(N::Int)
    return Matrix{Float32}(undef, 6, N)
end

# Mapping functions for specific types
function Map3xNd(data::AbstractVector{Float64}, N::Int)
    stride_len = stride()
    return reshape(data[1:stride_len:stride_len*3*N], 3, N)
end

function Map6xNf(data::AbstractVector{Float32}, N::Int)
    stride_len = stride()
    return reshape(data[1:stride_len:stride_len*6*N], 6, N)
end

function Map4d(data::AbstractVector{Float64})
    stride_len = stride()
    return data[1:stride_len:end]
end

end  # module Rfit

# Type aliases
const HitsView = TrackingRecHit2DSOAView

const Tuples = HitContainer
const OutputSoA = TrackSOA
const Tuple_Multiplicity = TupleMultiplicity

# HelixFitOnGPU structure
mutable struct HelixFitOnGPU
    max_number_of_concurrent_fits::Int
    tuples_d::Union{Nothing,Tuples}  # Nullable equivalent
    tuple_multiplicity_d::Union{Nothing,Tuple_Multiplicity}
    output_soa_d::Union{Nothing,OutputSoA}
    b_field::Float32
    fit5as4::Bool

    function HelixFitOnGPU(b_field::Float32, fit5as4::Bool)
        return new(
            Rfit.max_number_of_concurrent_fits(),
            nothing,
            nothing,
            nothing,
            b_field,
            fit5as4
        )
    end
end

function get_tuples_d(self::HelixFitOnGPU)
    return self.tuples_d
end

function get_tuple_multiplicity_d(self::HelixFitOnGPU)
    return self.tuple_multiplicity_d
end

function get_output_soa_d(self::HelixFitOnGPU)
    return self.output_soa_d
end

function set_b_field!(self::HelixFitOnGPU, b_field::Float64)
    self.b_field = b_field
end

function launch_riemann_kernels!(self::HelixFitOnGPU, hv::HitsView, nhits::UInt32, max_number_of_tuples::UInt32)
    println("Launching Riemann Kernels with b_field: ", self.b_field)
end

function launchBrokenLineKernelsOnCPU(self::HelixFitOnGPU, hv::HitsView, nhits::UInt32, max_number_of_tuples::UInt32)
    println("Launching Broken Line Kernels with b_field: ", self.b_field)
end

function allocate_on_gpu!(self::HelixFitOnGPU, tuples::Tuples, tuple_multiplicity::Tuple_Multiplicity, helix_fit_results::OutputSoA)
    self.tuples_d = tuples
    self.tuple_multiplicity_d = tuple_multiplicity
    self.output_soa_d = helix_fit_results

    @assert !isnothing(tuples)
    @assert !isnothing(tuple_multiplicity)
    @assert !isnothing(helix_fit_results)

end

function deallocate_on_gpu!(self::HelixFitOnGPU)
    self.tuples_d = nothing
    self.tuple_multiplicity_d = nothing
    self.output_soa_d = nothing
    println("Deallocated GPU resources")
end

end