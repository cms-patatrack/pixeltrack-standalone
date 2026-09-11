#ifndef CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
#define CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h

#include "CondFormats/SiPixelGainForHLTonGPU.h"

#include <span>
#include <vector>

class SiPixelGainCalibrationForHLTGPU {
public:
  explicit SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const &gain, std::vector<char> gainData);
  ~SiPixelGainCalibrationForHLTGPU();

  const SiPixelGainForHLTonGPU *getCPUProduct() const { return gainForHLTonHost_; }

  /// The same product by reference, and the block v_pedestals points into as a
  /// column: the pointer carries no length, and the length lives here.
  SiPixelGainForHLTonGPU const &gains() const { return *gainForHLTonHost_; }

  std::span<const unsigned char> pedestals() const {
    return {reinterpret_cast<unsigned char const *>(gainData_.data()), gainData_.size()};
  }

private:
  SiPixelGainForHLTonGPU *gainForHLTonHost_ = nullptr;
  std::vector<char> gainData_;
};

#endif  // CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
