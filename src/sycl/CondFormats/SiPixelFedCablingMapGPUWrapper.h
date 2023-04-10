#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include "SYCLCore/ESProduct.h"
#include "SYCLCore/device_unique_ptr.h"
#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include <CL/sycl.hpp>

#include <set>

class SiPixelFedCablingMapGPUWrapper {
public:
  explicit SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const &cablingMap,
                                          std::vector<unsigned char> modToUnp);
  ~SiPixelFedCablingMapGPUWrapper();

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  const SiPixelFedCablingMapGPU *getGPUProductAsync(sycl::queue stream) const;

  // returns pointer to GPU memory
  const unsigned char *getModToUnpAllAsync(sycl::queue stream) const;

private:
  std::vector<unsigned char> modToUnpDefault;
  bool hasQuality_;

  SiPixelFedCablingMapGPU *cablingMapHost_ = nullptr;  // pointer to struct in CPU

  struct GPUData {
    GPUData() = default;
    ~GPUData() {}

    cms::sycltools::device::unique_ptr<SiPixelFedCablingMapGPU> cablingMapDevice;  // pointer to struct in GPU
  };
  cms::sycltools::ESProduct<GPUData> gpuData_;

  struct ModulesToUnpack {
    ModulesToUnpack() = default;
    ~ModulesToUnpack(){};

    cms::sycltools::device::unique_ptr<unsigned char[]> modToUnpDefault;  // pointer to GPU
  };
  cms::sycltools::ESProduct<ModulesToUnpack> modToUnp_;
};

#endif
