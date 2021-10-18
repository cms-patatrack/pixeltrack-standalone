#include "CondFormats/PixelCPEFast.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include "AlpakaCore/device_unique_ptr.h"

#include <fstream>
#include <iostream>
#include <memory>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class PixelCPEFastESProducer : public edm::ESProducer {
  public:
    explicit PixelCPEFastESProducer(std::string const &datadir) : data_(datadir) {}
    void produce(edm::EventSetup &eventSetup);

  private:
    std::string data_;
  };

  void PixelCPEFastESProducer::produce(edm::EventSetup &eventSetup) {
    std::ifstream in((data_ + "/cpefast.bin").c_str(), std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);

    Queue queue(device);

    pixelCPEforGPU::CommonParams commonParams;
    in.read(reinterpret_cast<char *>(&commonParams), sizeof(pixelCPEforGPU::CommonParams));
    auto commonParams_h{cms::alpakatools::createHostView<pixelCPEforGPU::CommonParams>(&commonParams, 1u)};
    auto commonParams_d{cms::alpakatools::make_device_unique<pixelCPEforGPU::CommonParams>(1u)};
    auto commonParams_d_view{
        cms::alpakatools::createDeviceView<pixelCPEforGPU::CommonParams>(commonParams_d.get(), 1u)};
    alpaka::memcpy(queue, commonParams_d_view, commonParams_h, 1u);

    unsigned int ndetParams;
    in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
    //std::vector<pixelCPEforGPU::DetParams> detParams;
    //detParams.resize(ndetParams);
    std::vector<pixelCPEforGPU::DetParams> detParams(ndetParams);
    in.read(reinterpret_cast<char *>(detParams.data()), ndetParams * sizeof(pixelCPEforGPU::DetParams));
    auto detParams_h{cms::alpakatools::createHostView<pixelCPEforGPU::DetParams>(detParams.data(), ndetParams)};
    auto detParams_d{cms::alpakatools::make_device_unique<pixelCPEforGPU::DetParams>(ndetParams)};
    auto detParams_d_view{cms::alpakatools::createDeviceView<pixelCPEforGPU::DetParams>(detParams_d.get(), ndetParams)};
    alpaka::memcpy(queue, detParams_d_view, detParams_h, ndetParams);

    pixelCPEforGPU::AverageGeometry averageGeometry;
    in.read(reinterpret_cast<char *>(&averageGeometry), sizeof(pixelCPEforGPU::AverageGeometry));
    auto averageGeometry_h{cms::alpakatools::createHostView<pixelCPEforGPU::AverageGeometry>(&averageGeometry, 1u)};
    auto averageGeometry_d{cms::alpakatools::make_device_unique<pixelCPEforGPU::AverageGeometry>(1u)};
    auto averageGeometry_d_view{
        cms::alpakatools::createDeviceView<pixelCPEforGPU::AverageGeometry>(averageGeometry_d.get(), 1u)};
    alpaka::memcpy(queue, averageGeometry_d_view, averageGeometry_h, 1u);

    pixelCPEforGPU::LayerGeometry layerGeometry;
    in.read(reinterpret_cast<char *>(&layerGeometry), sizeof(pixelCPEforGPU::LayerGeometry));
    auto layerGeometry_h{cms::alpakatools::createHostView<pixelCPEforGPU::LayerGeometry>(&layerGeometry, 1u)};
    auto layerGeometry_d{cms::alpakatools::make_device_unique<pixelCPEforGPU::LayerGeometry>(1u)};
    auto layerGeometry_d_view{
        cms::alpakatools::createDeviceView<pixelCPEforGPU::LayerGeometry>(layerGeometry_d.get(), 1u)};
    alpaka::memcpy(queue, layerGeometry_d_view, layerGeometry_h, 1u);

    pixelCPEforGPU::ParamsOnGPU params;
    params.m_commonParams = commonParams_d.get();
    params.m_detParams = detParams_d.get();
    params.m_layerGeometry = layerGeometry_d.get();
    params.m_averageGeometry = averageGeometry_d.get();
    auto params_h{cms::alpakatools::createHostView<pixelCPEforGPU::ParamsOnGPU>(&params, 1u)};
    auto params_d{cms::alpakatools::make_device_unique<pixelCPEforGPU::ParamsOnGPU>(1u)};
    auto params_d_view{cms::alpakatools::createDeviceView<pixelCPEforGPU::ParamsOnGPU>(params_d.get(), 1u)};
    alpaka::memcpy(queue, params_d_view, params_h, 1u);

    alpaka::wait(queue);

    eventSetup.put(std::make_unique<PixelCPEFast>(std::move(commonParams_d),
                                                  std::move(detParams_d),
                                                  std::move(layerGeometry_d),
                                                  std::move(averageGeometry_d),
                                                  std::move(params_d)));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_EVENTSETUP_MODULE(PixelCPEFastESProducer);
