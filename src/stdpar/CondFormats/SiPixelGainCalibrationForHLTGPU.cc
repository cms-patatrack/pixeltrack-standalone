#include <cstring>
#include <memory>
#include <ranges>

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
  auto iter = std::views::iota(std::size_t{0}, gainData.size());
  std::for_each(std::ranges::cbegin(iter), std::ranges::cend(iter), [&](const auto& i) { gainData_[i] = gainData[i]; });
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {}
