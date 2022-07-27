#include <cstring>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/StreamCache.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(
    SiPixelGainForHLTonGPU const& gain, std::vector<DecodingStructure> const& gainData) {
  gainForHLT_ = new SiPixelGainForHLTonGPU(gain);
  gainData_ = new DecodingStructure[gainData.size()];
  gainForHLT_->v_pedestals = gainData_;
  std::memcpy(gainData_, gainData.data(), gainData.size());
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {
  delete gainForHLT_;
  delete[] gainData_;
}
