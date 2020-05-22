#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include "CUDACore/ESProduct.h"
#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include <cuda_runtime.h>

#include <set>

class SiPixelFedCablingMapGPUWrapper {
public:
  explicit SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                          std::vector<unsigned char> const& modToUnp);
  ~SiPixelFedCablingMapGPUWrapper();

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  const SiPixelFedCablingMapGPU* cablingMap() const { return cablingMap_; }

  // returns pointer to GPU memory
  const unsigned char* modToUnpAll() const { return modToUnpDefault_; }

private:
  bool hasQuality_;

  SiPixelFedCablingMapGPU* cablingMap_ = nullptr;
  unsigned char* modToUnpDefault_ = nullptr;
};

#endif
