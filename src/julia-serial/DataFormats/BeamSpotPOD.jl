module BeamSpotPOD_h
export BeamSpotPOD
struct BeamSpotPOD
    x::Float32  # position x
    y::Float32  # position y
    z::Float32  # position z
    sigmaZ::Float32
    beamWidthX::Float32
    beamWidthY::Float32
    dxdz::Float32
    dydz::Float32
    emittanceX::Float32
    emittanceY::Float32
    betaStar::Float32
end
end


