#ifndef RecoPixelVertexing_PixelTriplets_plugins_PixelDoubletsConstants_h
#define RecoPixelVertexing_PixelTriplets_plugins_PixelDoubletsConstants_h

#include "CAConstants.h"

namespace PixelDoubletsConstants {
  constexpr int nPairs = 13 + 2 + 4;
  static_assert(nPairs <= CAConstants::maxNumberOfLayerPairs());

  // constants via functor member variables
  class LayerPairs {
  private:
    const uint8_t layerPairs[2 * nPairs] = {
        0, 1, 0, 4, 0, 7,              // BPIX1 (3)
        1, 2, 1, 4, 1, 7,              // BPIX2 (5)
        4, 5, 7, 8,                    // FPIX1 (8)
        2, 3, 2, 4, 2, 7, 5, 6, 8, 9,  // BPIX3 & FPIX2 (13)
        0, 2, 1, 3,                    // Jumping Barrel (15)
        0, 5, 0, 8,                    // Jumping Forward (BPIX1,FPIX2)
        4, 6, 7, 9                     // Jumping Forward (19)
    };

  public:
    KOKKOS_INLINE_FUNCTION uint8_t operator[](int i) const { return layerPairs[i]; };
  };

  constexpr int16_t phi0p05 = 522;  // round(521.52189...) = phi2short(0.05);
  constexpr int16_t phi0p06 = 626;  // round(625.82270...) = phi2short(0.06);
  constexpr int16_t phi0p07 = 730;  // round(730.12648...) = phi2short(0.07);

  class PhiCuts {
  private:
    const int16_t phicuts[nPairs] = {phi0p05,
                                     phi0p07,
                                     phi0p07,
                                     phi0p05,
                                     phi0p06,
                                     phi0p06,
                                     phi0p05,
                                     phi0p05,
                                     phi0p06,
                                     phi0p06,
                                     phi0p06,
                                     phi0p05,
                                     phi0p05,
                                     phi0p05,
                                     phi0p05,
                                     phi0p05,
                                     phi0p05,
                                     phi0p05,
                                     phi0p05};

  public:
    KOKKOS_INLINE_FUNCTION int16_t operator[](int i) const { return phicuts[i]; };
  };
  //   phi0p07, phi0p07, phi0p06,phi0p06, phi0p06,phi0p06};  // relaxed cuts

  class MinZ {
  private:
    float const minz[nPairs] = {
        -20., 0., -30., -22., 10., -30., -70., -70., -22., 15., -30, -70., -70., -20., -22., 0, -30., -70., -70.};

  public:
    KOKKOS_INLINE_FUNCTION float operator[](int i) const { return minz[i]; };
  };

  class MaxZ {
  private:
    float const maxz[nPairs] = {
        20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.};

  public:
    KOKKOS_INLINE_FUNCTION float operator[](int i) const { return maxz[i]; };
  };

  class MaxR {
  private:
    float const maxr[nPairs] = {20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.};

  public:
    KOKKOS_INLINE_FUNCTION float operator[](int i) const { return maxr[i]; };
  };

}  // namespace PixelDoubletsConstants

#endif  // RecoPixelVertexing_PixelTriplets_plugins_PixelDoubletsConstants_h
