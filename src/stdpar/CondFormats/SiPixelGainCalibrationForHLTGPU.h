#ifndef CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
#define CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h

#include "CUDACore/ESProduct.h"
#include <memory>

class SiPixelGainForHLTonGPU;
struct SiPixelGainForHLTonGPU_DecodingStructure;

class SiPixelGainCalibrationForHLTGPU {
  using DecodingStructure = SiPixelGainForHLTonGPU_DecodingStructure;

public:
  explicit SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                           std::vector<DecodingStructure> const& gainData);
  ~SiPixelGainCalibrationForHLTGPU();

  const SiPixelGainForHLTonGPU* get() const { return gainForHLT_.get(); }

private:
  std::unique_ptr<SiPixelGainForHLTonGPU> gainForHLT_;
  std::unique_ptr<DecodingStructure[]> gainData_;
};

#endif  // CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
