#include <cstring>

#include <cuda.h>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/StreamCache.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> const& gainData) {
  cudaCheck(cudaMallocManaged(&gainForHLT_, sizeof(SiPixelGainForHLTonGPU)));
  *gainForHLT_ = gain;
  cudaCheck(cudaMallocManaged(&gainData_, gainData.size()));
  gainForHLT_->v_pedestals = gainData_;

  std::memcpy(gainData_, gainData.data(), gainData.size());
  for (int device = 0, ndev = cms::cuda::deviceCount(); device < ndev; ++device) {
#ifndef CUDAUVM_DISABLE_ADVISE
    cudaCheck(cudaMemAdvise(gainForHLT_, sizeof(SiPixelGainForHLTonGPU), cudaMemAdviseSetReadMostly, device));
    cudaCheck(cudaMemAdvise(gainData_, gainData.size(), cudaMemAdviseSetReadMostly, device));
#endif
#ifndef CUDAUVM_DISABLE_PREFETCH
    cms::cuda::ScopedSetDevice guard{device};
    auto stream = cms::cuda::getStreamCache().get();
    cudaCheck(cudaMemPrefetchAsync(gainForHLT_, sizeof(SiPixelGainForHLTonGPU), device, stream.get()));
    cudaCheck(cudaMemPrefetchAsync(gainData_, gainData.size(), device, stream.get()));
#endif
  }
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {
  cudaCheck(cudaFree(gainForHLT_));
  cudaCheck(cudaFree(gainData_));
}
