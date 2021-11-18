#ifndef RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
#define RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h

#include <utility>

#include "CondFormats/pixelCPEforGPU.h"

#include "AlpakaCore/alpakaCommon.h"
#include "AlpakaCore/ESProduct.h"
#include "AlpakaCore/alpakaMemoryHelper.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelCPEFast {
  public:
    PixelCPEFast(std::string const &path)
        : m_commonParamsGPU(::cms::alpakatools::allocHostBuf<pixelCPEforGPU::CommonParams>(1u)),
          m_layerGeometry(::cms::alpakatools::allocHostBuf<pixelCPEforGPU::LayerGeometry>(1u)),
          m_averageGeometry(::cms::alpakatools::allocHostBuf<pixelCPEforGPU::AverageGeometry>(1u))

    {
      std::ifstream in(path, std::ios::binary);
      in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
      in.read(reinterpret_cast<char *>(alpaka::getPtrNative(m_commonParamsGPU)), sizeof(pixelCPEforGPU::CommonParams));
      unsigned int ndetParams;
      in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
      m_detParamsGPU.resize(ndetParams);
      in.read(reinterpret_cast<char *>(m_detParamsGPU.data()), ndetParams * sizeof(pixelCPEforGPU::DetParams));
      in.read(reinterpret_cast<char *>(alpaka::getPtrNative(m_averageGeometry)),
              sizeof(pixelCPEforGPU::AverageGeometry));
      in.read(reinterpret_cast<char *>(alpaka::getPtrNative(m_layerGeometry)), sizeof(pixelCPEforGPU::LayerGeometry));

      alpaka::prepareForAsyncCopy(m_commonParamsGPU);
      alpaka::prepareForAsyncCopy(m_layerGeometry);
      alpaka::prepareForAsyncCopy(m_averageGeometry);
    }

    ~PixelCPEFast() = default;

    // The return value can only be used safely in kernels launched on
    // the same cudaStream, or after cudaStreamSynchronize.
    const pixelCPEforGPU::ParamsOnGPU *getGPUProductAsync(Queue &queue) const {
      const auto &data = gpuData_.dataForDeviceAsync(queue, [this](Queue &queue) {
        unsigned int ndetParams = m_detParamsGPU.size();
        GPUData gpuData(queue, ndetParams);

        alpaka::memcpy(queue, gpuData.d_commonParams, m_commonParamsGPU, 1u);
        alpaka::getPtrNative(gpuData.h_paramsOnGPU)->m_commonParams = alpaka::getPtrNative(gpuData.d_commonParams);

        alpaka::memcpy(queue, gpuData.d_layerGeometry, m_layerGeometry, 1u);
        alpaka::getPtrNative(gpuData.h_paramsOnGPU)->m_layerGeometry = alpaka::getPtrNative(gpuData.d_layerGeometry);

        alpaka::memcpy(queue, gpuData.d_averageGeometry, m_averageGeometry, 1u);
        alpaka::getPtrNative(gpuData.h_paramsOnGPU)->m_averageGeometry =
            alpaka::getPtrNative(gpuData.d_averageGeometry);

        auto detParams_h =
            ::cms::alpakatools::createHostView<const pixelCPEforGPU::DetParams>(m_detParamsGPU.data(), ndetParams);
        alpaka::memcpy(queue, gpuData.d_detParams, detParams_h, ndetParams);
        alpaka::getPtrNative(gpuData.h_paramsOnGPU)->m_detParams = alpaka::getPtrNative(gpuData.d_detParams);

        alpaka::memcpy(queue, gpuData.d_paramsOnGPU, gpuData.h_paramsOnGPU, 1u);

        return gpuData;
      });
      return alpaka::getPtrNative(data.d_paramsOnGPU);
    }

  private:
    // allocate it with posix malloc to be compatible with cpu wf
    std::vector<pixelCPEforGPU::DetParams> m_detParamsGPU;
    AlpakaHostBuf<pixelCPEforGPU::CommonParams> m_commonParamsGPU;
    AlpakaHostBuf<pixelCPEforGPU::LayerGeometry> m_layerGeometry;
    AlpakaHostBuf<pixelCPEforGPU::AverageGeometry> m_averageGeometry;

    struct GPUData {
      // not needed if not used on CPU...
    public:
      GPUData() = delete;
      GPUData(Queue &queue, unsigned int ndetParams)
          : h_paramsOnGPU{::cms::alpakatools::allocHostBuf<pixelCPEforGPU::ParamsOnGPU>(1u)},
            d_paramsOnGPU{::cms::alpakatools::allocDeviceBuf<pixelCPEforGPU::ParamsOnGPU>(alpaka::getDev(queue), 1u)},
            d_commonParams{::cms::alpakatools::allocDeviceBuf<pixelCPEforGPU::CommonParams>(alpaka::getDev(queue), 1u)},
            d_layerGeometry{
                ::cms::alpakatools::allocDeviceBuf<pixelCPEforGPU::LayerGeometry>(alpaka::getDev(queue), 1u)},
            d_averageGeometry{
                ::cms::alpakatools::allocDeviceBuf<pixelCPEforGPU::AverageGeometry>(alpaka::getDev(queue), 1u)},
            d_detParams{
                ::cms::alpakatools::allocDeviceBuf<pixelCPEforGPU::DetParams>(alpaka::getDev(queue), ndetParams)} {
        alpaka::prepareForAsyncCopy(h_paramsOnGPU);
      };
      ~GPUData() = default;

    public:
      AlpakaHostBuf<pixelCPEforGPU::ParamsOnGPU> h_paramsOnGPU;
      AlpakaDeviceBuf<pixelCPEforGPU::ParamsOnGPU> d_paramsOnGPU;  // copy of the above on the Device
      AlpakaDeviceBuf<pixelCPEforGPU::CommonParams> d_commonParams;
      AlpakaDeviceBuf<pixelCPEforGPU::LayerGeometry> d_layerGeometry;
      AlpakaDeviceBuf<pixelCPEforGPU::AverageGeometry> d_averageGeometry;
      AlpakaDeviceBuf<pixelCPEforGPU::DetParams> d_detParams;
    };

    ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::ESProduct<GPUData> gpuData_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_PixelCPEFast_h
