#include <sycl/sycl.hpp>

#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU const& gain,
                                                                 std::vector<char> gainData)
    : gainData_(std::move(gainData)) {
  gainForHLTonHost_ = new SiPixelGainForHLTonGPU();
  *gainForHLTonHost_ = gain;
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() { delete gainForHLTonHost_; }

const SiPixelGainForHLTonGPU* SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(sycl::queue SYCLstream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(SYCLstream, [this](GPUData& data, sycl::queue stream) {
    data.gainForHLTonGPU = cms::sycltools::make_device_unique_uninitialized<SiPixelGainForHLTonGPU>(stream);
    data.gainDataOnGPU = cms::sycltools::make_device_unique_uninitialized<SiPixelGainForHLTonGPU_DecodingStructure[]>(
        this->gainData_.size(), stream);

    // those in CUDA were three memcpy: here they didn't work
    // this is another way to achieve the same result
    stream.memcpy(data.gainDataOnGPU.get(), this->gainData_.data(), this->gainData_.size());
#ifdef CPU_DEBUG
    stream.wait();
#endif
    this->gainForHLTonHost_->v_pedestals = data.gainDataOnGPU.get();

    stream.memcpy(data.gainForHLTonGPU.get(), this->gainForHLTonHost_, sizeof(SiPixelGainForHLTonGPU));
#ifdef CPU_DEBUG
    stream.wait();
#endif
  });
  return data.gainForHLTonGPU.get();
}
