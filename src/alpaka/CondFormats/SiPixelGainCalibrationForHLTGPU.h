#ifndef CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
#define CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h

#include <vector>

#include "AlpakaCore/ESProduct.h"
#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaCore/alpakaMemoryHelper.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class SiPixelGainCalibrationForHLTGPU {
  public:
    using DecodingStructure = SiPixelGainForHLTonGPU::DecodingStructure;

    SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU gain, std::vector<DecodingStructure> gainData)
        : gainForHLTonHost_{::cms::alpakatools::allocHostBuf<SiPixelGainForHLTonGPU>(1u)},
          gainData_(gainData),
          numDecodingStructures_(gainData.size()) {
      *alpaka::getPtrNative(gainForHLTonHost_) = gain;
      alpaka::prepareForAsyncCopy(gainForHLTonHost_);
    };

    ~SiPixelGainCalibrationForHLTGPU() = default;

    const SiPixelGainForHLTonGPU* getGPUProductAsync(Queue& queue) const {
      const auto& data = gpuData_.dataForDeviceAsync(queue, [this](Queue& queue) {
        GPUData gpuData(queue, numDecodingStructures_);
        auto gainDataView(::cms::alpakatools::createHostView(gainData_.data(), numDecodingStructures_));
        alpaka::memcpy(queue, gpuData.v_pedestalsGPU, gainDataView, numDecodingStructures_);

        *alpaka::getPtrNative(gpuData.gainDataOnGPU) = *alpaka::getPtrNative(gainForHLTonHost_);
        alpaka::getPtrNative(gpuData.gainDataOnGPU)->pedestals_ = alpaka::getPtrNative(gpuData.v_pedestalsGPU);

        alpaka::memcpy(queue, gpuData.gainForHLTonGPU, gpuData.gainDataOnGPU, 1u);
        return gpuData;
      });
      return alpaka::getPtrNative(data.gainForHLTonGPU);
    };

  private:
    struct GPUData {
    public:
      GPUData() = delete;
      GPUData(Queue const& queue, unsigned int numDecodingStructures)
          : gainForHLTonGPU{::cms::alpakatools::allocDeviceBuf<SiPixelGainForHLTonGPU>(alpaka::getDev(queue), 1u)},
            gainDataOnGPU{::cms::alpakatools::allocHostBuf<SiPixelGainForHLTonGPU>(1u)},
            v_pedestalsGPU{
                ::cms::alpakatools::allocDeviceBuf<DecodingStructure>(alpaka::getDev(queue), numDecodingStructures)} {
        alpaka::prepareForAsyncCopy(gainDataOnGPU);
      };
      ~GPUData() = default;

      AlpakaDeviceBuf<SiPixelGainForHLTonGPU> gainForHLTonGPU;
      AlpakaHostBuf<SiPixelGainForHLTonGPU> gainDataOnGPU;
      AlpakaDeviceBuf<DecodingStructure> v_pedestalsGPU;
    };

    AlpakaHostBuf<SiPixelGainForHLTonGPU> gainForHLTonHost_;
    std::vector<DecodingStructure> gainData_;
    ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::ESProduct<GPUData> gpuData_;
    uint32_t numDecodingStructures_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
