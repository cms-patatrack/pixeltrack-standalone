#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

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

const pixelCPEforGPU::ParamsOnGPU *PixelCPEFast::getGPUProductAsync(cudaStream_t cudaStream) const {
  const auto &data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData &data, cudaStream_t stream) {
    // and now copy to device...
    cudaCheck(cudaMalloc((void **)&data.h_paramsOnGPU.m_commonParams, sizeof(pixelCPEforGPU::CommonParams)));
    cudaCheck(cudaMalloc((void **)&data.h_paramsOnGPU.m_detParams,
                         this->m_detParamsGPU.size() * sizeof(pixelCPEforGPU::DetParams)));
    cudaCheck(cudaMalloc((void **)&data.h_paramsOnGPU.m_averageGeometry, sizeof(pixelCPEforGPU::AverageGeometry)));
    cudaCheck(cudaMalloc((void **)&data.h_paramsOnGPU.m_layerGeometry, sizeof(pixelCPEforGPU::LayerGeometry)));
    cudaCheck(cudaMalloc((void **)&data.d_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU)));

    cudaCheck(cudaMemcpyAsync(
        data.d_paramsOnGPU, &data.h_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU), cudaMemcpyDefault, stream));
    cudaCheck(cudaMemcpyAsync((void *)data.h_paramsOnGPU.m_commonParams,
                              &this->m_commonParamsGPU,
                              sizeof(pixelCPEforGPU::CommonParams),
                              cudaMemcpyDefault,
                              stream));
    cudaCheck(cudaMemcpyAsync((void *)data.h_paramsOnGPU.m_averageGeometry,
                              &this->m_averageGeometry,
                              sizeof(pixelCPEforGPU::AverageGeometry),
                              cudaMemcpyDefault,
                              stream));
    cudaCheck(cudaMemcpyAsync((void *)data.h_paramsOnGPU.m_layerGeometry,
                              &this->m_layerGeometry,
                              sizeof(pixelCPEforGPU::LayerGeometry),
                              cudaMemcpyDefault,
                              stream));
    cudaCheck(cudaMemcpyAsync((void *)data.h_paramsOnGPU.m_detParams,
                              this->m_detParamsGPU.data(),
                              this->m_detParamsGPU.size() * sizeof(pixelCPEforGPU::DetParams),
                              cudaMemcpyDefault,
                              stream));
  });
  return data.d_paramsOnGPU;
}

PixelCPEFast::GPUData::~GPUData() {
  if (d_paramsOnGPU != nullptr) {
    cudaFree((void *)h_paramsOnGPU.m_commonParams);
    cudaFree((void *)h_paramsOnGPU.m_detParams);
    cudaFree((void *)h_paramsOnGPU.m_averageGeometry);
    cudaFree(d_paramsOnGPU);
  }
}
