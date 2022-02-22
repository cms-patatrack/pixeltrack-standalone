// C++ includes
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA includes
#include <hip/hip_runtime.h>

// CMSSW includes
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> modToUnp)
    : modToUnpDefault(modToUnp.size()), hasQuality_(true) {
  cudaCheck(hipHostMalloc(&cablingMapHost, sizeof(SiPixelFedCablingMapGPU)));
  std::memcpy(cablingMapHost, &cablingMap, sizeof(SiPixelFedCablingMapGPU));

  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() { cudaCheck(hipHostFree(cablingMapHost)); }

const SiPixelFedCablingMapGPU* SiPixelFedCablingMapGPUWrapper::getGPUProductAsync(hipStream_t cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, hipStream_t stream) {
    // allocate
    cudaCheck(hipMalloc(&data.cablingMapDevice, sizeof(SiPixelFedCablingMapGPU)));

    // transfer
    cudaCheck(hipMemcpyAsync(
        data.cablingMapDevice, this->cablingMapHost, sizeof(SiPixelFedCablingMapGPU), hipMemcpyDefault, stream));
  });
  return data.cablingMapDevice;
}

const unsigned char* SiPixelFedCablingMapGPUWrapper::getModToUnpAllAsync(hipStream_t cudaStream) const {
  const auto& data = modToUnp_.dataForCurrentDeviceAsync(cudaStream, [this](ModulesToUnpack& data, hipStream_t stream) {
    cudaCheck(hipMalloc((void**)&data.modToUnpDefault, pixelgpudetails::MAX_SIZE_BYTE_BOOL));
    cudaCheck(hipMemcpyAsync(data.modToUnpDefault,
                             this->modToUnpDefault.data(),
                             this->modToUnpDefault.size() * sizeof(unsigned char),
                             hipMemcpyDefault,
                             stream));
  });
  return data.modToUnpDefault;
}

SiPixelFedCablingMapGPUWrapper::GPUData::~GPUData() { cudaCheck(hipFree(cablingMapDevice)); }

SiPixelFedCablingMapGPUWrapper::ModulesToUnpack::~ModulesToUnpack() { cudaCheck(hipFree(modToUnpDefault)); }
