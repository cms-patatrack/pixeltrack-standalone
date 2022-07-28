// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include <memory>
#include <ranges>
#include <cstddef>

// CMSSW includes
#include "CUDACore/cudaCheck.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/StreamCache.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> const& modToUnp)
    : hasQuality_(true),
      //rvalue ref
      cablingMap_{std::make_unique<SiPixelFedCablingMapGPU>(cablingMap)},
      modToUnpDefault_{std::make_unique<unsigned char[]>(modToUnp.size())} {
  auto iter = std::views::iota(std::size_t{0}, modToUnp.size());
  std::for_each(
      std::ranges::cbegin(iter), std::ranges::cend(iter), [&](const auto& i) { modToUnpDefault_[i] = modToUnp[i]; });
}

SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() {}
