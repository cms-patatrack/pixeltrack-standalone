#include <iostream>
#include <fstream>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

#include "Geometry/phase1PixelTopology.h"
#include "CUDACore/cudaCheck.h"
#include "CondFormats/PixelCPEFast.h"

// Services
// this is needed to get errors from templates

namespace {
  constexpr float micronsToCm = 1.0e-4;
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

  cpuData_ = {
      &m_commonParamsGPU,
      m_detParamsGPU.data(),
      &m_layerGeometry,
      &m_averageGeometry,
  };
}

const pixelCPEforGPU::ParamsOnGPU *PixelCPEFast::getGPUProductAsync(hipStream_t cudaStream) const {
  const auto &data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData &data, hipStream_t stream) {
    // and now copy to device...
    cudaCheck(hipMalloc((void **)&data.h_paramsOnGPU.m_commonParams, sizeof(pixelCPEforGPU::CommonParams)));
    cudaCheck(hipMalloc((void **)&data.h_paramsOnGPU.m_detParams,
                        this->m_detParamsGPU.size() * sizeof(pixelCPEforGPU::DetParams)));
    cudaCheck(hipMalloc((void **)&data.h_paramsOnGPU.m_averageGeometry, sizeof(pixelCPEforGPU::AverageGeometry)));
    cudaCheck(hipMalloc((void **)&data.h_paramsOnGPU.m_layerGeometry, sizeof(pixelCPEforGPU::LayerGeometry)));
    cudaCheck(hipMalloc((void **)&data.d_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU)));

    cudaCheck(hipMemcpyAsync(
        data.d_paramsOnGPU, &data.h_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU), hipMemcpyDefault, stream));
    cudaCheck(hipMemcpyAsync((void *)data.h_paramsOnGPU.m_commonParams,
                             &this->m_commonParamsGPU,
                             sizeof(pixelCPEforGPU::CommonParams),
                             hipMemcpyDefault,
                             stream));
    cudaCheck(hipMemcpyAsync((void *)data.h_paramsOnGPU.m_averageGeometry,
                             &this->m_averageGeometry,
                             sizeof(pixelCPEforGPU::AverageGeometry),
                             hipMemcpyDefault,
                             stream));
    cudaCheck(hipMemcpyAsync((void *)data.h_paramsOnGPU.m_layerGeometry,
                             &this->m_layerGeometry,
                             sizeof(pixelCPEforGPU::LayerGeometry),
                             hipMemcpyDefault,
                             stream));
    cudaCheck(hipMemcpyAsync((void *)data.h_paramsOnGPU.m_detParams,
                             this->m_detParamsGPU.data(),
                             this->m_detParamsGPU.size() * sizeof(pixelCPEforGPU::DetParams),
                             hipMemcpyDefault,
                             stream));
  });
  return data.d_paramsOnGPU;
}

PixelCPEFast::GPUData::~GPUData() {
  if (d_paramsOnGPU != nullptr) {
    hipFree((void *)h_paramsOnGPU.m_commonParams);
    hipFree((void *)h_paramsOnGPU.m_detParams);
    hipFree((void *)h_paramsOnGPU.m_averageGeometry);
    hipFree(d_paramsOnGPU);
  }
}
