#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelROCsStatusAndMappingWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelROCsStatusAndMappingWrapper_h

#include "CUDACore/device_unique_ptr.h"
#include "CondFormats/SiPixelROCsStatusAndMapping.h"

#include <cuda_runtime.h>

class SiPixelROCsStatusAndMappingWrapper {
public:
  explicit SiPixelROCsStatusAndMappingWrapper(cms::cuda::device::unique_ptr<SiPixelROCsStatusAndMapping> cablingMap,
                                              cms::cuda::device::unique_ptr<unsigned char[]> modToUnp)
      : cablingMap_(std::move(cablingMap)), modToUnp_(std::move(modToUnp)), hasQuality_(true) {}

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  const SiPixelROCsStatusAndMapping *getSiPixelROCsStatusAndMapping() const { return cablingMap_.get(); }

  // returns pointer to GPU memory
  const unsigned char *getModToUnpAll() const { return modToUnp_.get(); }

private:
  cms::cuda::device::unique_ptr<SiPixelROCsStatusAndMapping> cablingMap_;
  cms::cuda::device::unique_ptr<unsigned char[]> modToUnp_;

  bool hasQuality_;
};

#endif
