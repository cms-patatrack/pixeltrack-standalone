#ifndef CondFormats_alpaka_SiPixelGainCalibrationForHLTGPU_h
#define CondFormats_alpaka_SiPixelGainCalibrationForHLTGPU_h

#include <vector>

#include "AlpakaCore/ESProduct.h"
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"
#include "CondFormats/alpaka/SiPixelGainForHLTonGPU.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelGainCalibrationForHLTGPU {
  public:
    using DecodingStructure = SiPixelGainForHLTonGPU::DecodingStructure;

    SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU gain, std::vector<DecodingStructure> gainData)
        : gainForHLTonHost_{cms::alpakatools::make_host_buffer<SiPixelGainForHLTonGPU, Platform>()},
          gainData_(gainData),
          numDecodingStructures_(gainData.size()) {
      *gainForHLTonHost_ = gain;
    };

    ~SiPixelGainCalibrationForHLTGPU() = default;

    const SiPixelGainForHLTonGPU* getGPUProductAsync(Queue& queue) const {
      const auto& data = gpuData_.dataForDeviceAsync(queue, [this](Queue& queue) {
        GPUData gpuData(queue, numDecodingStructures_);
        auto gainDataView(cms::alpakatools::make_host_view(gainData_.data(), numDecodingStructures_));
        alpaka::memcpy(queue, gpuData.v_pedestalsGPU, gainDataView);

        *gpuData.gainDataOnGPU = *gainForHLTonHost_;
        gpuData.gainDataOnGPU->pedestals_ = gpuData.v_pedestalsGPU.data();

        alpaka::memcpy(queue, gpuData.gainForHLTonGPU, gpuData.gainDataOnGPU);
        return gpuData;
      });
      return data.gainForHLTonGPU.data();
    };

  private:
    struct GPUData {
    public:
      GPUData() = delete;
      GPUData(Queue const& queue, unsigned int numDecodingStructures)
          : gainForHLTonGPU{cms::alpakatools::make_device_buffer<SiPixelGainForHLTonGPU>(queue)},
            gainDataOnGPU{cms::alpakatools::make_host_buffer<SiPixelGainForHLTonGPU>(queue)},
            v_pedestalsGPU{cms::alpakatools::make_device_buffer<DecodingStructure[]>(queue, numDecodingStructures)} {};
      ~GPUData() = default;

      cms::alpakatools::device_buffer<Device, SiPixelGainForHLTonGPU> gainForHLTonGPU;
      cms::alpakatools::host_buffer<SiPixelGainForHLTonGPU> gainDataOnGPU;
      cms::alpakatools::device_buffer<Device, DecodingStructure[]> v_pedestalsGPU;
    };

    cms::alpakatools::host_buffer<SiPixelGainForHLTonGPU> gainForHLTonHost_;
    std::vector<DecodingStructure> gainData_;
    cms::alpakatools::ESProduct<Queue, GPUData> gpuData_;
    uint32_t numDecodingStructures_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CondFormats_alpaka_SiPixelGainCalibrationForHLTGPU_h
