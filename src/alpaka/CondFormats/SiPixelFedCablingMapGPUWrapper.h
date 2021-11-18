#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h

#include <utility>

#include "CondFormats/SiPixelFedCablingMapGPU.h"

#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaCore/ESProduct.h"
#include "AlpakaCore/alpakaMemoryHelper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelFedCablingMapGPUWrapper {
  public:
    using CablingMapDeviceBuf = AlpakaDeviceBuf<SiPixelFedCablingMapGPU>;
    using CablingMapHostBuf = AlpakaHostBuf<SiPixelFedCablingMapGPU>;

    explicit SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMapGPU cablingMap, std::vector<unsigned char> modToUnp)
        : modToUnpDefault_(modToUnp.size()),
          cablingMapHost_{::cms::alpakatools::allocHostBuf<SiPixelFedCablingMapGPU>(1u)},
          hasQuality_{true} {
      std::memcpy(alpaka::getPtrNative(cablingMapHost_), &cablingMap, sizeof(SiPixelFedCablingMapGPU));
      std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault_.begin());
    }
    ~SiPixelFedCablingMapGPUWrapper() = default;

    bool hasQuality() const { return hasQuality_; }

    const SiPixelFedCablingMapGPU* cablingMap() const { return alpaka::getPtrNative(cablingMapHost_); }

    const SiPixelFedCablingMapGPU* getGPUProductAsync(Queue& queue) const {
      const auto& data = gpuData_.dataForDeviceAsync(queue, [this](Queue& queue) {
        // allocate
        GPUData gpuData(queue);
        alpaka::memcpy(queue, gpuData.cablingMapDevice, cablingMapHost_, 1u);
        return gpuData;
      });
      return alpaka::getPtrNative(data.cablingMapDevice);
    }

    const unsigned char* getModToUnpAllAsync(Queue& queue) const {
      const auto& data = modToUnp_.dataForDeviceAsync(queue, [this](Queue& queue) {
        unsigned int modToUnpSize = modToUnpDefault_.size();
        ModulesToUnpack modToUnp(queue, modToUnpSize);
        auto modToUnpDefault_view =
            ::cms::alpakatools::createHostView<const unsigned char>(modToUnpDefault_.data(), modToUnpSize);
        alpaka::memcpy(queue, modToUnp.modToUnpDefault, modToUnpDefault_view, modToUnpSize);
        return modToUnp;
      });
      return alpaka::getPtrNative(data.modToUnpDefault);
    }

  private:
    std::vector<unsigned char> modToUnpDefault_;
    CablingMapHostBuf cablingMapHost_;
    bool hasQuality_;

    struct GPUData {
    public:
      GPUData() = delete;
      GPUData(Queue const& queue)
          : cablingMapDevice{::cms::alpakatools::allocDeviceBuf<SiPixelFedCablingMapGPU>(alpaka::getDev(queue), 1u)} {
        alpaka::prepareForAsyncCopy(cablingMapDevice);
      };
      ~GPUData() = default;

      AlpakaDeviceBuf<SiPixelFedCablingMapGPU> cablingMapDevice;  // pointer to struct in GPU
    };
    ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::ESProduct<GPUData> gpuData_;

    struct ModulesToUnpack {
    public:
      ModulesToUnpack() = delete;
      ModulesToUnpack(Queue const& queue, unsigned int modToUnpSize)
          : modToUnpDefault{::cms::alpakatools::allocDeviceBuf<unsigned char>(alpaka::getDev(queue), modToUnpSize)} {};
      ~ModulesToUnpack() = default;

      AlpakaDeviceBuf<unsigned char> modToUnpDefault;  // pointer to GPU
    };

    ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::ESProduct<ModulesToUnpack> modToUnp_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
