#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include <set>
#include <span>
#include <vector>

class SiPixelFedCablingMapGPUWrapper {
public:
  explicit SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const &cablingMap,
                                          std::vector<unsigned char> modToUnp);
  ~SiPixelFedCablingMapGPUWrapper() = default;

  bool hasQuality() const { return hasQuality_; }

  const SiPixelFedCablingMapGPU *getCPUProduct() const { return &cablingMapHost; }

  const unsigned char *getModToUnpAll() const { return modToUnpDefault.data(); }

  /// The same two, in shapes a binding can be generated from: a pointer says
  /// nothing about what it points at, and a bare pointer nothing about how far
  /// it runs.
  SiPixelFedCablingMapGPU const &cablingMap() const { return cablingMapHost; }

  std::span<const unsigned char> modToUnpAll() const { return modToUnpDefault; }

private:
  std::vector<unsigned char> modToUnpDefault;
  bool hasQuality_;

  SiPixelFedCablingMapGPU cablingMapHost;
};

#endif
