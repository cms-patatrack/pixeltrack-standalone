#ifndef CondFormats_alpaka_PixelCPEFast_h
#define CondFormats_alpaka_PixelCPEFast_h

#include <utility>

#include "AlpakaCore/ESProduct.h"
#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"
#include "CondFormats/pixelCPEforGPU.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelCPEFast {
  public:
    PixelCPEFast(std::string const &path)
        : m_commonParamsGPU(cms::alpakatools::make_host_buffer<pixelCPEforGPU::CommonParams, Platform>()),
          m_layerGeometry(cms::alpakatools::make_host_buffer<pixelCPEforGPU::LayerGeometry, Platform>()),
          m_averageGeometry(cms::alpakatools::make_host_buffer<pixelCPEforGPU::AverageGeometry, Platform>())

    {
      std::ifstream in(path, std::ios::binary);
      in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
      in.read(reinterpret_cast<char *>(m_commonParamsGPU.data()), sizeof(pixelCPEforGPU::CommonParams));
      unsigned int ndetParams;
      in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
      m_detParamsGPU.resize(ndetParams);
      in.read(reinterpret_cast<char *>(m_detParamsGPU.data()), ndetParams * sizeof(pixelCPEforGPU::DetParams));
      in.read(reinterpret_cast<char *>(m_averageGeometry.data()), sizeof(pixelCPEforGPU::AverageGeometry));
      in.read(reinterpret_cast<char *>(m_layerGeometry.data()), sizeof(pixelCPEforGPU::LayerGeometry));
    }

    ~PixelCPEFast() = default;

    // The return value can only be used safely in kernels launched on
    // the same cudaStream, or after cudaStreamSynchronize.
    const pixelCPEforGPU::ParamsOnGPU *getGPUProductAsync(Queue &queue) const {
      const auto &data = gpuData_.dataForDeviceAsync(queue, [this](Queue &queue) {
        unsigned int ndetParams = m_detParamsGPU.size();
        GPUData gpuData(queue, ndetParams);

        alpaka::memcpy(queue, gpuData.d_commonParams, m_commonParamsGPU);
        gpuData.h_paramsOnGPU->m_commonParams = gpuData.d_commonParams.data();

        alpaka::memcpy(queue, gpuData.d_layerGeometry, m_layerGeometry);
        gpuData.h_paramsOnGPU->m_layerGeometry = gpuData.d_layerGeometry.data();

        alpaka::memcpy(queue, gpuData.d_averageGeometry, m_averageGeometry);
        gpuData.h_paramsOnGPU->m_averageGeometry = gpuData.d_averageGeometry.data();

        auto detParams_h = cms::alpakatools::make_host_view(m_detParamsGPU.data(), ndetParams);
        alpaka::memcpy(queue, gpuData.d_detParams, detParams_h);
        gpuData.h_paramsOnGPU->m_detParams = gpuData.d_detParams.data();

        alpaka::memcpy(queue, gpuData.d_paramsOnGPU, gpuData.h_paramsOnGPU);

        return gpuData;
      });
      return data.d_paramsOnGPU.data();
    }

  private:
    // allocate it with posix malloc to be compatible with cpu wf
    std::vector<pixelCPEforGPU::DetParams> m_detParamsGPU;
    cms::alpakatools::host_buffer<pixelCPEforGPU::CommonParams> m_commonParamsGPU;
    cms::alpakatools::host_buffer<pixelCPEforGPU::LayerGeometry> m_layerGeometry;
    cms::alpakatools::host_buffer<pixelCPEforGPU::AverageGeometry> m_averageGeometry;

    struct GPUData {
      // not needed if not used on CPU...
    public:
      GPUData() = delete;
      GPUData(Queue &queue, unsigned int ndetParams)
          : h_paramsOnGPU{cms::alpakatools::make_host_buffer<pixelCPEforGPU::ParamsOnGPU>(queue)},
            d_paramsOnGPU{cms::alpakatools::make_device_buffer<pixelCPEforGPU::ParamsOnGPU>(queue)},
            d_commonParams{cms::alpakatools::make_device_buffer<pixelCPEforGPU::CommonParams>(queue)},
            d_layerGeometry{cms::alpakatools::make_device_buffer<pixelCPEforGPU::LayerGeometry>(queue)},
            d_averageGeometry{cms::alpakatools::make_device_buffer<pixelCPEforGPU::AverageGeometry>(queue)},
            d_detParams{cms::alpakatools::make_device_buffer<pixelCPEforGPU::DetParams[]>(queue, ndetParams)} {};
      ~GPUData() = default;

    public:
      cms::alpakatools::host_buffer<pixelCPEforGPU::ParamsOnGPU> h_paramsOnGPU;
      cms::alpakatools::device_buffer<Device, pixelCPEforGPU::ParamsOnGPU>
          d_paramsOnGPU;  // copy of the above on the Device
      cms::alpakatools::device_buffer<Device, pixelCPEforGPU::CommonParams> d_commonParams;
      cms::alpakatools::device_buffer<Device, pixelCPEforGPU::LayerGeometry> d_layerGeometry;
      cms::alpakatools::device_buffer<Device, pixelCPEforGPU::AverageGeometry> d_averageGeometry;
      cms::alpakatools::device_buffer<Device, pixelCPEforGPU::DetParams[]> d_detParams;
    };

    cms::alpakatools::ESProduct<Queue, GPUData> gpuData_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CondFormats_alpaka_PixelCPEFast_h
