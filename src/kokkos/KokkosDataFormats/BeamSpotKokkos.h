#ifndef KokkosDataFormats_BeamSpot_interface_BeamSpotKokkos_h
#define KokkosDataFormats_BeamSpot_interface_BeamSpotKokkos_h

#include "KokkosCore/kokkosConfig.h"
#include "DataFormats/BeamSpotPOD.h"

#include <cstring>

template <typename MemorySpace>
class BeamSpotKokkos {
public:
  BeamSpotKokkos() = default;
  BeamSpotKokkos(BeamSpotPOD const* data) : data_d{"data_d"} {
    typename Kokkos::View<BeamSpotPOD, MemorySpace>::HostMirror data_h = Kokkos::create_mirror_view(data_d);
    std::memcpy(data_h.data(), data, sizeof(BeamSpotPOD));
    Kokkos::deep_copy(data_d, data_h);
  }

  KOKKOS_INLINE_FUNCTION BeamSpotPOD const* data() const { return data_d.data(); }

private:
  Kokkos::View<BeamSpotPOD, MemorySpace> data_d;
};

#endif
