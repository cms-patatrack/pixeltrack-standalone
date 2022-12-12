// C++ includes
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// SYCL includes
#include <CL/sycl.hpp>

// CMSSW includes
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> modToUnp)
    : modToUnpDefault(modToUnp.size()), hasQuality_(true) {
  cablingMapHost_ = new SiPixelFedCablingMapGPU();
  std::memcpy(cablingMapHost_, &cablingMap, sizeof(SiPixelFedCablingMapGPU));
  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() { delete cablingMapHost_; }

const SiPixelFedCablingMapGPU* SiPixelFedCablingMapGPUWrapper::getGPUProductAsync(sycl::queue stream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData& data, sycl::queue& stream) {
    // allocate
    data.cablingMapDevice = cms::sycltools::make_device_unique_uninitialized<SiPixelFedCablingMapGPU>(stream);
    // transfer
    stream.memcpy(data.cablingMapDevice.get(), this->cablingMapHost_, sizeof(SiPixelFedCablingMapGPU));
  });
  return data.cablingMapDevice.get();
}

const unsigned char* SiPixelFedCablingMapGPUWrapper::getModToUnpAllAsync(sycl::queue stream) const {
  const auto& data = modToUnp_.dataForCurrentDeviceAsync(stream, [this](ModulesToUnpack& data, sycl::queue stream) {
    data.modToUnpDefault =
        cms::sycltools::make_device_unique<unsigned char[]>(pixelgpudetails::MAX_SIZE_BYTE_BOOL, stream);
    stream.memcpy(
        data.modToUnpDefault.get(), this->modToUnpDefault.data(), this->modToUnpDefault.size() * sizeof(unsigned char));
  });
  return data.modToUnpDefault.get();
}
