#ifndef CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
#define CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h

#include "CUDACore/ESProduct.h"

class SiPixelGainForHLTonGPU;
struct SiPixelGainForHLTonGPU_DecodingStructure;

class SiPixelGainCalibrationForHLTGPU {
  using DecodingStructure = SiPixelGainForHLTonGPU_DecodingStructure;

public:
  explicit SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                           std::vector<DecodingStructure> const& gainData);
  ~SiPixelGainCalibrationForHLTGPU();

  const SiPixelGainForHLTonGPU* get() const { return gainForHLT_; }

private:
  SiPixelGainForHLTonGPU* gainForHLT_ = nullptr;
  DecodingStructure* gainData_ = nullptr;
};

#endif  // CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
