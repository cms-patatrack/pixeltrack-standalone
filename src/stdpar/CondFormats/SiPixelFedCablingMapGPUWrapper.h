#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include <memory>

#include "CondFormats/SiPixelFedCablingMapGPU.h"


class SiPixelFedCablingMapGPUWrapper {
public:
  explicit SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                          std::vector<unsigned char> const& modToUnp);

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  const SiPixelFedCablingMapGPU* cablingMap() const { return cablingMap_.get(); }

  // returns pointer to GPU memory
  const unsigned char* modToUnpAll() const { return modToUnpDefault_.get(); }

private:
  bool hasQuality_;

  std::unique_ptr<SiPixelFedCablingMapGPU> cablingMap_;
  std::unique_ptr<unsigned char[]> modToUnpDefault_;
};

#endif
