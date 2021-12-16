#ifndef KokkosDataFormats_BeamSpot_interface_BeamSpotKokkos_h
#define KokkosDataFormats_BeamSpot_interface_BeamSpotKokkos_h

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/deep_copy.h"
#include "KokkosCore/shared_ptr.h"
#include "KokkosCore/ViewHelpers.h"
#include "DataFormats/BeamSpotPOD.h"

#include <cstring>

template <typename MemorySpace>
class BeamSpotKokkos {
public:
  BeamSpotKokkos() = default;
  template <typename ExecSpace>
  BeamSpotKokkos(BeamSpotPOD const* data, ExecSpace const& execSpace)
      : data_d{cms::kokkos::make_shared<BeamSpotPOD, MemorySpace>(execSpace)} {
    auto data_h = cms::kokkos::make_mirror_shared(data_d, execSpace);
    std::memcpy(data_h.get(), data, sizeof(BeamSpotPOD));
    cms::kokkos::deep_copy(execSpace, data_d, data_h);
  }

  BeamSpotPOD const* data() const { return data_d.get(); }

private:
  cms::kokkos::shared_ptr<BeamSpotPOD, MemorySpace> data_d;
};

#endif
