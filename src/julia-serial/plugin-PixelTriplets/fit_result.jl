module RecoPixelVertexing_PixelTrackFitting_interface_FitResult_h

using StaticArrays
export circle_fit, line_fit, helix_fit
"""
A fixed-size, stack-allocated circle_fit using StaticArrays
Fields:
  - par: 3-element parameter vector (X0, Y0, R)
  - cov: 3×3 covariance matrix
  - q:   particle charge (Int32)
  - chi2: χ² value (Float64)
"""
mutable struct circle_fit
    par  :: MVector{3,Float64}
    cov  :: MMatrix{3,3,Float64}
    q    :: Int32
    chi2 :: Float64

    # Default constructor
    function circle_fit()
        new(
            MVector{3,Float64}(0.0, 0.0, 0.0),
            MMatrix{3,3,Float64}(
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0
            ),
            0,
            0.0
        )
    end
end
# Struct for line_fit with a default constructor
mutable struct line_fit
    par  :: MVector{2,Float64}
    cov  :: MMatrix{2,2,Float64}
    chi2 :: Float64

    # Default constructor
    line_fit() = new(
        MVector{2,Float64}(0.0, 0.0),               # Default parameter vector
        MMatrix{2,2,Float64}(0.0, 0.0,
                             0.0, 0.0),            # Default 2×2 covariance
        0.0                                         # Default chi-squared
    )
end


# Struct for helix_fit with a default constructor
mutable struct helix_fit
    par::Vector{Float64} # (phi, Tip, pt, cotan(theta), Zip)
    cov::Matrix{Float64} # covariance matrix
    # < ()->cov() 
    # |(phi,phi)|(Tip,phi)|(p_t,phi)|(c_t,phi)|(Zip,phi)| 
    # |(phi,Tip)|(Tip,Tip)|(p_t,Tip)|(c_t,Tip)|(Zip,Tip)| 
    # |(phi,p_t)|(Tip,p_t)|(p_t,p_t)|(c_t,p_t)|(Zip,p_t)| 
    # |(phi,c_t)|(Tip,c_t)|(p_t,c_t)|(c_t,c_t)|(Zip,c_t)| 
    # |(phi,Zip)|(Tip,Zip)|(p_t,Zip)|(c_t,Zip)|(Zip,Zip)|
    chi2_circle::Float64 # chi-squared value for the circle fit
    chi2_line::Float64   # chi-squared value for the line fit
    q::Int32             # Particle charge

    # Default constructor
    helix_fit() = new(
        [0.0, 0.0, 0.0, 0.0, 0.0],   # Default parameter vector
        zeros(5, 5),                 # Default 5x5 covariance matrix
        0.0,                         # Default chi-squared value for the circle fit
        0.0,                         # Default chi-squared value for the line fit
        0                            # Default particle charge (neutral)
    )
end


end