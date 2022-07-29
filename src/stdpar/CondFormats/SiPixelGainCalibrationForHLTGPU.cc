#include <algorithm>
#include <memory>
#include <vector>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/StreamCache.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<DecodingStructure> const& gainData)
    : gainForHLT_{std::make_unique<SiPixelGainForHLTonGPU>(gain)},
//make_shared for array
#if __GNUC__ >= 12
      gainData_{std::make_shared<DecodingStructure[]>(gainData.size())}
#else
      gainData_{new DecodingStructure[gainData.size()]}
#endif
{
  gainForHLT_->v_pedestals = gainData_;
  std::copy(gainData.cbegin(), gainData.cend(), gainData_.get());
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {}
