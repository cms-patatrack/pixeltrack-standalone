// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// CMSSW includes
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDADataFormats/gpuClusteringConstants.h"
#include "CondFormats/SiPixelROCsStatusAndMappingWrapper.h"

SiPixelROCsStatusAndMappingWrapper::SiPixelROCsStatusAndMappingWrapper(SiPixelROCsStatusAndMapping const& cablingMap,
                                                               std::vector<unsigned char> modToUnp)
    : modToUnpDefault(modToUnp.size()), hasQuality_(true) {
  cudaCheck(cudaMallocHost(&cablingMapHost, sizeof(SiPixelROCsStatusAndMapping)));
  std::memcpy(cablingMapHost, &cablingMap, sizeof(SiPixelROCsStatusAndMapping));

  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

SiPixelROCsStatusAndMappingWrapper::~SiPixelROCsStatusAndMappingWrapper() { cudaCheck(cudaFreeHost(cablingMapHost)); }

const SiPixelROCsStatusAndMapping* SiPixelROCsStatusAndMappingWrapper::getGPUProductAsync(cudaStream_t cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, cudaStream_t stream) {
    // allocate
    cudaCheck(cudaMalloc(&data.cablingMapDevice, sizeof(SiPixelROCsStatusAndMapping)));

    // transfer
    cudaCheck(cudaMemcpyAsync(
        data.cablingMapDevice, this->cablingMapHost, sizeof(SiPixelROCsStatusAndMapping), cudaMemcpyDefault, stream));
  });
  return data.cablingMapDevice;
}

const unsigned char* SiPixelROCsStatusAndMappingWrapper::getModToUnpAllAsync(cudaStream_t cudaStream) const {
  const auto& data =
      modToUnp_.dataForCurrentDeviceAsync(cudaStream, [this](ModulesToUnpack& data, cudaStream_t stream) {
        cudaCheck(cudaMalloc((void**)&data.modToUnpDefault, pixelgpudetails::MAX_SIZE_BYTE_BOOL));
        cudaCheck(cudaMemcpyAsync(data.modToUnpDefault,
                                  this->modToUnpDefault.data(),
                                  this->modToUnpDefault.size() * sizeof(unsigned char),
                                  cudaMemcpyDefault,
                                  stream));
      });
  return data.modToUnpDefault;
}

SiPixelROCsStatusAndMappingWrapper::GPUData::~GPUData() { cudaCheck(cudaFree(cablingMapDevice)); }

SiPixelROCsStatusAndMappingWrapper::ModulesToUnpack::~ModulesToUnpack() { cudaCheck(cudaFree(modToUnpDefault)); }
