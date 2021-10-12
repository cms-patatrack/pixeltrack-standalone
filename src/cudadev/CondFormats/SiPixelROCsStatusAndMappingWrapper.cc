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
#include "CUDACore/copyAsync.h"

SiPixelROCsStatusAndMappingWrapper::SiPixelROCsStatusAndMappingWrapper(SiPixelROCsStatusAndMapping const& cablingMap,
                                                                       std::vector<unsigned char> modToUnp)
    : modToUnpDefault(modToUnp.size()), hasQuality_(true) {
  // TODO: check if cudaStreamDefault is appropriate
  auto cablingMapMetadata = cablingMap.soaMetadata();
  cablingMapHostBuffer = cms::cuda::make_host_unique<std::byte[]>(cablingMapMetadata.byteSize(), cudaStreamDefault);
  std::memcpy(cablingMapHostBuffer.get(), cablingMapMetadata.data(), cablingMapMetadata.byteSize());
  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

const SiPixelROCsStatusAndMapping & SiPixelROCsStatusAndMappingWrapper::getGPUProductAsync(cudaStream_t cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, 
    [this](GPUData& data, cudaStream_t stream) {
      // allocate
      data.allocate(pixelgpudetails::MAX_SIZE, stream);
      // transfer
      cms::cuda::copyAsync(data.cablingMapDeviceBuffer, this->cablingMapHostBuffer, 
        data.cablingMapDevice.soaMetadata().byteSize(), stream);
    }
  );
  return data.cablingMapDevice;
}

const unsigned char* SiPixelROCsStatusAndMappingWrapper::getModToUnpAllAsync(cudaStream_t cudaStream) const {
  const auto& data =
      modToUnp_.dataForCurrentDeviceAsync(cudaStream, [this](ModulesToUnpack& data, cudaStream_t stream) {
        data.modToUnpDefault = cms::cuda::make_device_unique<unsigned char []>(pixelgpudetails::MAX_SIZE_BYTE_BOOL, stream);
        cudaCheck(cudaMemcpyAsync(data.modToUnpDefault.get(),
                                  this->modToUnpDefault.data(),
                                  this->modToUnpDefault.size() * sizeof(unsigned char),
                                  cudaMemcpyDefault,
                                  stream));
      });
  return data.modToUnpDefault.get();
}