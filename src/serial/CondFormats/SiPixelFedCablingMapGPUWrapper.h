#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include <set>

class SiPixelFedCablingMapGPUWrapper {
public:
  explicit SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const &cablingMap,
                                          std::vector<unsigned char> modToUnp);
  ~SiPixelFedCablingMapGPUWrapper() = default;

  bool hasQuality() const { return hasQuality_; }

  const SiPixelFedCablingMapGPU *getCPUProduct() const { return &cablingMapHost; }

  const unsigned char *getModToUnpAll() const { return modToUnpDefault.data(); }

private:
  std::vector<unsigned char> modToUnpDefault;
  bool hasQuality_;

  SiPixelFedCablingMapGPU cablingMapHost;
};

#endif
