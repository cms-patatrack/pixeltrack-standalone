module VertexSOA
const MAX_TRACKS::Int32 = 32 * 1024
const MAX_VTX::Int32 = 1024
mutable struct ZVertexSoA
    # Arrays for tracks and vertices
    idv::Vector{Int16}         # Vertex index for each associated track (-1 = not associated)
    zv::Vector{Float32}        # Output z-position of found vertices
    wv::Vector{Float32}        # Output weight (1/error^2)
    chi2::Vector{Float32}      # Chi-squared value for vertices
    ptv2::Vector{Float32}      # Transverse momentum squared of vertices
    ndof::Vector{Int32}        # Number of degrees of freedom (workspace for nearest neighbors)
    sortInd::Vector{UInt16}    # Sorted index (ascending order by pt^2)
    nv_final::UInt32            # Number of vertices
    # Constructor with default initialization
    function ZVertexSoA()
        new(
            fill(Int16(-1), MAX_TRACKS), # idv array initialized to -1
            zeros(Float32, MAX_VTX),   # zv array initialized to 0.0
            zeros(Float32, MAX_VTX),   # wv array initialized to 0.0
            zeros(Float32, MAX_VTX),   # chi2 array initialized to 0.0
            zeros(Float32, MAX_VTX),   # ptv2 array initialized to 0.0
            zeros(Int32, MAX_TRACKS), # ndof array initialized to 0
            zeros(UInt16, MAX_VTX),    # sortInd array initialized to 0
            UInt32(0)               # nvFinal initialized to 0
        )
    end
end
end