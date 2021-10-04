// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// CMSSW includes
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU const& cablingMap,
                                                               std::vector<unsigned char> modToUnp)
  : modToUnpDefault(modToUnp.size()), hasQuality_(true), cablingMapHost(cablingMap) {
  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

