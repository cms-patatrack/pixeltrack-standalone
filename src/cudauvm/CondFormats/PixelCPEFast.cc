#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Geometry/phase1PixelTopology.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/StreamCache.h"
#include "CondFormats/PixelCPEFast.h"

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEFast::PixelCPEFast(std::string const &path) {
  unsigned int ndetParams;

  cudaCheck(cudaMallocManaged(&m_params, sizeof(pixelCPEforGPU::ParamsOnGPU)));
  cudaCheck(cudaMallocManaged(&m_commonParams, sizeof(pixelCPEforGPU::CommonParams)));
  cudaCheck(cudaMallocManaged(&m_layerGeometry, sizeof(pixelCPEforGPU::LayerGeometry)));
  cudaCheck(cudaMallocManaged(&m_averageGeometry, sizeof(pixelCPEforGPU::AverageGeometry)));

  {
    std::ifstream in(path, std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    in.read(reinterpret_cast<char *>(m_commonParams), sizeof(pixelCPEforGPU::CommonParams));
    in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
    cudaCheck(cudaMallocManaged(&m_detParams, ndetParams * sizeof(pixelCPEforGPU::DetParams)));
    in.read(reinterpret_cast<char *>(m_detParams), ndetParams * sizeof(pixelCPEforGPU::DetParams));
    in.read(reinterpret_cast<char *>(m_averageGeometry), sizeof(pixelCPEforGPU::AverageGeometry));
    in.read(reinterpret_cast<char *>(m_layerGeometry), sizeof(pixelCPEforGPU::LayerGeometry));
  }

  m_params->m_commonParams = m_commonParams;
  m_params->m_detParams = m_detParams;
  m_params->m_layerGeometry = m_layerGeometry;
  m_params->m_averageGeometry = m_averageGeometry;

  for (int device = 0, ndev = cms::cuda::deviceCount(); device < ndev; ++device) {
#ifndef CUDAUVM_DISABLE_ADVISE
    cudaCheck(cudaMemAdvise(m_params, sizeof(pixelCPEforGPU::ParamsOnGPU), cudaMemAdviseSetReadMostly, device));
    cudaCheck(cudaMemAdvise(m_commonParams, sizeof(pixelCPEforGPU::CommonParams), cudaMemAdviseSetReadMostly, device));
    cudaCheck(
        cudaMemAdvise(m_detParams, ndetParams * sizeof(pixelCPEforGPU::DetParams), cudaMemAdviseSetReadMostly, device));
    cudaCheck(
        cudaMemAdvise(m_layerGeometry, sizeof(pixelCPEforGPU::LayerGeometry), cudaMemAdviseSetReadMostly, device));
    cudaCheck(
        cudaMemAdvise(m_averageGeometry, sizeof(pixelCPEforGPU::AverageGeometry), cudaMemAdviseSetReadMostly, device));
#endif
#ifndef CUDAUVM_DISABLE_PREFETCH
    cms::cuda::ScopedSetDevice guard{device};
    auto stream = cms::cuda::getStreamCache().get();
    cudaCheck(cudaMemPrefetchAsync(m_params, sizeof(pixelCPEforGPU::ParamsOnGPU), device, stream.get()));
    cudaCheck(cudaMemPrefetchAsync(m_commonParams, sizeof(pixelCPEforGPU::CommonParams), device, stream.get()));
    cudaCheck(cudaMemPrefetchAsync(m_detParams, ndetParams * sizeof(pixelCPEforGPU::DetParams), device, stream.get()));
    cudaCheck(cudaMemPrefetchAsync(m_layerGeometry, sizeof(pixelCPEforGPU::LayerGeometry), device, stream.get()));
    cudaCheck(cudaMemPrefetchAsync(m_averageGeometry, sizeof(pixelCPEforGPU::AverageGeometry), device, stream.get()));
#endif
  }
}

PixelCPEFast::~PixelCPEFast() {
  cudaFree(m_params);
  cudaFree(m_commonParams);
  cudaFree(m_detParams);
  cudaFree(m_layerGeometry);
  cudaFree(m_averageGeometry);
}
