#ifndef CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
#define CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h

#include "CUDACore/device_unique_ptr.h"

class SiPixelGainForHLTonGPU;
struct SiPixelGainForHLTonGPU_DecodingStructure;

class SiPixelGainCalibrationForHLTGPU {
public:
  explicit SiPixelGainCalibrationForHLTGPU(cms::cuda::device::unique_ptr<SiPixelGainForHLTonGPU> gain,
                                           cms::cuda::device::unique_ptr<char[]> gainData)
      : gain_(std::move(gain)), gainData_(std::move(gainData)) {}

  const SiPixelGainForHLTonGPU *get() const { return gain_.get(); }

private:
  cms::cuda::device::unique_ptr<SiPixelGainForHLTonGPU> gain_;
  cms::cuda::device::unique_ptr<char[]> gainData_;
};

#endif  // CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
