#include <iostream>
#include <fstream>

#include <CL/sycl.hpp>

#include "Geometry/phase1PixelTopology.h"
#include "CondFormats/PixelCPEFast.h"

// Services
// this is needed to get errors from templates

namespace {
  [[maybe_unused]] constexpr float micronsToCm = 1.0e-4;
}

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEFast::PixelCPEFast(std::string const &path) {
  {
    std::ifstream in(path, std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    in.read(reinterpret_cast<char *>(&m_commonParamsGPU), sizeof(pixelCPEforGPU::CommonParams));
    unsigned int ndetParams;
    in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
    m_detParamsGPU.resize(ndetParams);
    in.read(reinterpret_cast<char *>(m_detParamsGPU.data()), ndetParams * sizeof(pixelCPEforGPU::DetParams));
    in.read(reinterpret_cast<char *>(&m_averageGeometry), sizeof(pixelCPEforGPU::AverageGeometry));
    in.read(reinterpret_cast<char *>(&m_layerGeometry), sizeof(pixelCPEforGPU::LayerGeometry));
  }
}

const pixelCPEforGPU::ParamsOnGPU *PixelCPEFast::getGPUProductAsync(sycl::queue stream) const {
  const auto &data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData &data, sycl::queue stream) {
    // and now copy to device...
    data.h_paramsOnGPU.m_commonParams = cms::sycltools::make_device_unique<pixelCPEforGPU::CommonParams>(stream);
    data.h_paramsOnGPU.m_detParams = cms::sycltools::make_device_unique_uninitialized<pixelCPEforGPU::DetParams[]>(
        this->m_detParamsGPU.size(), stream);
    data.h_paramsOnGPU.m_averageGeometry = cms::sycltools::make_device_unique<pixelCPEforGPU::AverageGeometry>(stream);
    data.h_paramsOnGPU.m_layerGeometry = cms::sycltools::make_device_unique<pixelCPEforGPU::LayerGeometry>(stream);
    data.d_paramsOnGPU = cms::sycltools::make_device_unique_uninitialized<pixelCPEforGPU::ParamsOnGPU>(stream);

    stream.memcpy(data.d_paramsOnGPU.get(), &data.h_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU));
    stream.memcpy((void *)data.h_paramsOnGPU.m_commonParams.get(),
                  &this->m_commonParamsGPU,
                  sizeof(pixelCPEforGPU::CommonParams));
    stream.memcpy((void *)data.h_paramsOnGPU.m_averageGeometry.get(),
                  &this->m_averageGeometry,
                  sizeof(pixelCPEforGPU::AverageGeometry));
    stream.memcpy((void *)data.h_paramsOnGPU.m_layerGeometry.get(),
                  &this->m_layerGeometry,
                  sizeof(pixelCPEforGPU::LayerGeometry));
    stream
        .memcpy((void *)data.h_paramsOnGPU.m_detParams.get(),
                this->m_detParamsGPU.data(),
                this->m_detParamsGPU.size() * sizeof(pixelCPEforGPU::DetParams))
        .wait();
  });
  return data.d_paramsOnGPU.get();
}

PixelCPEFast::GPUData::~GPUData() = default;
