#include "CondFormats/SiPixelFedCablingMapGPU.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include "KokkosCore/kokkosConfig.h"

#include <fstream>
#include <memory>

namespace KOKKOS_NAMESPACE {
  class SiPixelFedCablingMapESProducer : public edm::ESProducer {
  public:
    explicit SiPixelFedCablingMapESProducer(std::string const& datadir) : data_(datadir) {}
    void produce(edm::EventSetup& eventSetup);

  private:
    std::string data_;
  };

  void SiPixelFedCablingMapESProducer::produce(edm::EventSetup& eventSetup) {
    std::ifstream in((data_ + "/cablingMap.bin").c_str(), std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    SiPixelFedCablingMapGPU obj;
    in.read(reinterpret_cast<char*>(&obj), sizeof(SiPixelFedCablingMapGPU));
    unsigned int modToUnpDefSize;
    in.read(reinterpret_cast<char*>(&modToUnpDefSize), sizeof(unsigned int));
    std::vector<unsigned char> modToUnpDefault(modToUnpDefSize);
    in.read(reinterpret_cast<char*>(modToUnpDefault.data()), modToUnpDefSize);

    Kokkos::View<SiPixelFedCablingMapGPU, KokkosExecSpace> cablingMap_d("cablingMap_d");
    auto cablingMap_h = Kokkos::create_mirror_view(cablingMap_d);
    (*cablingMap_h.data()) = obj;
    Kokkos::deep_copy(cablingMap_d, cablingMap_h);
    eventSetup.put(std::make_unique<SiPixelFedCablingMapGPUWrapper<KokkosExecSpace>>(std::move(cablingMap_d), true));

    Kokkos::View<unsigned char*, KokkosExecSpace> modToUnp_d("modToUnp_d", modToUnpDefSize);
    auto modToUnp_h = Kokkos::create_mirror_view(modToUnp_d);
    std::copy(modToUnpDefault.begin(), modToUnpDefault.end(), modToUnp_h.data());
    Kokkos::deep_copy(modToUnp_d, modToUnp_h);
    eventSetup.put(std::make_unique<Kokkos::View<const unsigned char*, KokkosExecSpace>>(std::move(modToUnp_d)));
    Kokkos::fence();
  }
}  // namespace KOKKOS_NAMESPACE

DEFINE_FWK_KOKKOS_EVENTSETUP_MODULE(SiPixelFedCablingMapESProducer);
