#ifndef CUDADataFormats_BeamSpot_interface_BeamSpotKokkos_h
#define CUDADataFormats_BeamSpot_interface_BeamSpotKokkos_h

#include "KokkosCore/kokkosConfig.h"

#include <cstring>

template<typename MemorySpace>
class BeamSpotKokkos {
public:
  struct Data {
    float x, y, z;  // position
    // TODO: add covariance matrix

    float sigmaZ;
    float beamWidthX, beamWidthY;
    float dxdz, dydz;
    float emittanceX, emittanceY;
    float betaStar;
  };

  BeamSpotKokkos() = default;
  BeamSpotKokkos(Data const* data): data_d{"data_d"} {
    typename Kokkos::View<Data, MemorySpace>::HostMirror data_h = Kokkos::create_mirror_view(data_d);
    std::memcpy(data_h.data(), data, sizeof(Data));
    Kokkos::deep_copy(data_d, data_h);
  }

  KOKKOS_INLINE_FUNCTION Data const* data() const { return data_d.data(); }

private:
  Kokkos::View<Data, MemorySpace> data_d;
};

#endif
