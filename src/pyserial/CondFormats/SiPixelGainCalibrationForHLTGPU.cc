#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> gainData)
    : gainData_(std::move(gainData)) {
  gainForHLTonHost_ = new SiPixelGainForHLTonGPU(gain);
  gainForHLTonHost_->v_pedestals = reinterpret_cast<SiPixelGainForHLTonGPU_DecodingStructure*>(gainData_.data());
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {
  delete gainForHLTonHost_;
}
