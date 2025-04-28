using .PixelGPU_h
using .Geometry_TrackerGeometryBuilder_phase1PixelTopology_h.phase1PixelTopology

struct PixelCPEFast
    m_detParamsGPU::Vector{DetParams}
    m_commonParamsGPU::CommonParams
    m_layerGeometry::LayerGeometry
    m_averageGeometry::AverageGeometry
    cpuData_::ParamsOnGPU
end



# Define the getCPUProduct method
function getCPUProduct(pixelCPE::PixelCPEFast)::ParamsOnGPU
    return pixelCPE.cpuData_
end

