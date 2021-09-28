#include "CondFormats/PixelCPEFast.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/ViewHelpers.h"

#include <fstream>
#include <iostream>
#include <memory>

namespace KOKKOS_NAMESPACE {
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

    Kokkos::View<pixelCPEforGPU::CommonParams, KokkosDeviceMemSpace> commonParams_d(
        Kokkos::ViewAllocateWithoutInitializing("commonParams_d"));
    auto commonParams_h = cms::kokkos::create_mirror_view(commonParams_d);
    in.read(reinterpret_cast<char *>(commonParams_h.data()), sizeof(pixelCPEforGPU::CommonParams));
    Kokkos::deep_copy(KokkosExecSpace(), commonParams_d, commonParams_h);

    unsigned int ndetParams;
    in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
    Kokkos::View<pixelCPEforGPU::DetParams *, KokkosDeviceMemSpace> detParams_d(
        Kokkos::ViewAllocateWithoutInitializing("detParams_d"), ndetParams);
    auto detParams_h = cms::kokkos::create_mirror_view(detParams_d);
    in.read(reinterpret_cast<char *>(detParams_h.data()), ndetParams * sizeof(pixelCPEforGPU::DetParams));
    Kokkos::deep_copy(KokkosExecSpace(), detParams_d, detParams_h);

    Kokkos::View<pixelCPEforGPU::AverageGeometry, KokkosDeviceMemSpace> averageGeometry_d(
        Kokkos::ViewAllocateWithoutInitializing("averageGeometry_d"));
    auto averageGeometry_h = cms::kokkos::create_mirror_view(averageGeometry_d);
    in.read(reinterpret_cast<char *>(averageGeometry_h.data()), sizeof(pixelCPEforGPU::AverageGeometry));
    Kokkos::deep_copy(KokkosExecSpace(), averageGeometry_d, averageGeometry_h);

    Kokkos::View<pixelCPEforGPU::LayerGeometry, KokkosDeviceMemSpace> layerGeometry_d(
        Kokkos::ViewAllocateWithoutInitializing("layerGeometry_d"));
    auto layerGeometry_h = cms::kokkos::create_mirror_view(layerGeometry_d);
    in.read(reinterpret_cast<char *>(layerGeometry_h.data()), sizeof(pixelCPEforGPU::LayerGeometry));
    Kokkos::deep_copy(KokkosExecSpace(), layerGeometry_d, layerGeometry_h);

    Kokkos::View<pixelCPEforGPU::ParamsOnGPU, KokkosDeviceMemSpace> params_d(
        Kokkos::ViewAllocateWithoutInitializing("params_d"));
    auto params_h = cms::kokkos::create_mirror_view(params_d);
    params_h().m_commonParams = commonParams_d.data();
    params_h().m_detParams = detParams_d.data();
    params_h().m_layerGeometry = layerGeometry_d.data();
    params_h().m_averageGeometry = averageGeometry_d.data();
    Kokkos::deep_copy(KokkosExecSpace(), params_d, params_h);

    KokkosExecSpace().fence();
    eventSetup.put(std::make_unique<PixelCPEFast<KokkosDeviceMemSpace>>(std::move(commonParams_d),
                                                                        std::move(detParams_d),
                                                                        std::move(layerGeometry_d),
                                                                        std::move(averageGeometry_d),
                                                                        std::move(params_d)));
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_EVENTSETUP_MODULE(PixelCPEFastESProducer);
