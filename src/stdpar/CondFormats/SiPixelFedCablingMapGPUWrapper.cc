// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// CMSSW includes
#include "CUDACore/cudaCheck.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/StreamCache.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> const& modToUnp)
    : hasQuality_(true) {
  cablingMap_ = new SiPixelFedCablingMapGPU(cablingMap);

  modToUnpDefault_ = new unsigned char[modToUnp.size()];
  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault_);
}

SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() {
  delete cablingMap_;
  delete[] modToUnpDefault_;
}
