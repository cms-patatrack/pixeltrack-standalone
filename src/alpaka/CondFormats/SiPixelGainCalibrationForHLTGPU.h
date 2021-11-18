#ifndef CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
#define CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h

#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaCore/ESProduct.h"
#include "AlpakaCore/alpakaMemoryHelper.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"

class SiPixelGainForHLTonGPU;
struct SiPixelGainForHLTonGPU_DecodingStructure;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class SiPixelGainCalibrationForHLTGPU {
  public:
    SiPixelGainCalibrationForHLTGPU(SiPixelGainForHLTonGPU gain,
                                    std::vector<SiPixelGainForHLTonGPU_DecodingStructure> gainData)
        : gainForHLTonHost{::cms::alpakatools::allocHostBuf<SiPixelGainForHLTonGPU>(1u)},
          h_gainData(gainData),
          numDecodingStructures_(gainData.size()) {
      *alpaka::getPtrNative(gainForHLTonHost) = gain;
      alpaka::prepareForAsyncCopy(gainForHLTonHost);
    };

    ~SiPixelGainCalibrationForHLTGPU() = default;

    const SiPixelGainForHLTonGPU *getGPUProductAsync(Queue& queue) const {
      const auto &data = gpuData_.dataForDeviceAsync(queue, [this](Queue& queue) {
        GPUData gpuData(queue, numDecodingStructures_);
        auto gainDataView(::cms::alpakatools::createHostView(h_gainData.data(), numDecodingStructures_));
        alpaka::memcpy(queue, gpuData.v_pedestalsGPU, gainDataView, numDecodingStructures_);

        *alpaka::getPtrNative(gpuData.gainDataOnGPU) = *alpaka::getPtrNative(gainForHLTonHost);
        alpaka::getPtrNative(gpuData.gainDataOnGPU)->v_pedestals = alpaka::getPtrNative(gpuData.v_pedestalsGPU);

        alpaka::memcpy(queue, gpuData.gainForHLTonGPU, gpuData.gainDataOnGPU, 1u);
        return gpuData;
      });
      return alpaka::getPtrNative(data.gainForHLTonGPU);
    };

  private:
    AlpakaHostBuf<SiPixelGainForHLTonGPU> gainForHLTonHost;
    std::vector<SiPixelGainForHLTonGPU_DecodingStructure> h_gainData;
    uint32_t numDecodingStructures_;

    struct GPUData {
    public:
      GPUData() = delete;
      GPUData(Queue const& queue, unsigned int numDecodingStructures)
          : gainForHLTonGPU{::cms::alpakatools::allocDeviceBuf<SiPixelGainForHLTonGPU>(alpaka::getDev(queue), 1u)},
            gainDataOnGPU{::cms::alpakatools::allocHostBuf<SiPixelGainForHLTonGPU>(1u)},
            v_pedestalsGPU{::cms::alpakatools::allocDeviceBuf<SiPixelGainForHLTonGPU_DecodingStructure>(
                alpaka::getDev(queue), numDecodingStructures)} {
        alpaka::prepareForAsyncCopy(gainDataOnGPU);
      };

      ~GPUData(){};

    public:
      AlpakaDeviceBuf<SiPixelGainForHLTonGPU> gainForHLTonGPU;
      AlpakaHostBuf<SiPixelGainForHLTonGPU> gainDataOnGPU;
      AlpakaDeviceBuf<SiPixelGainForHLTonGPU_DecodingStructure> v_pedestalsGPU;
    };
    ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::ESProduct<GPUData> gpuData_;
  };
};      // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif  // CalibTracker_SiPixelESProducers_interface_SiPixelGainCalibrationForHLTGPU_h
