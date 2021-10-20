#ifndef KokkosDataFormats_BeamSpot_interface_BeamSpotKokkos_h
#define KokkosDataFormats_BeamSpot_interface_BeamSpotKokkos_h

#include "KokkosCore/kokkosConfig.h"
#include "KokkosCore/ViewHelpers.h"
#include "DataFormats/BeamSpotPOD.h"

#include <cstring>

template <typename MemorySpace>
class BeamSpotKokkos {
public:
  BeamSpotKokkos() = default;
  template <typename ExecSpace>
  BeamSpotKokkos(BeamSpotPOD const* data, ExecSpace const& execSpace)
      : data_d{Kokkos::ViewAllocateWithoutInitializing("data_d")} {
    auto data_h = cms::kokkos::create_mirror_view(data_d);
    std::memcpy(data_h.data(), data, sizeof(BeamSpotPOD));
    Kokkos::deep_copy(execSpace, data_d, data_h);
  }

  KOKKOS_INLINE_FUNCTION BeamSpotPOD const* data() const { return data_d.data(); }

private:
  Kokkos::View<BeamSpotPOD, MemorySpace> data_d;
};

#endif
