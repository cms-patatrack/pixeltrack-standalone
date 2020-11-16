#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h

#include <cstdint>
#include <cstdio>

#include "KokkosCore/kokkosConfig.h"

#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "KokkosDataFormats/gpuClusteringConstants.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuCalibPixel {

    constexpr uint16_t InvId = 9999;  // must be > MaxNumModules

    // valid for run2
    constexpr float VCaltoElectronGain = 47;         // L2-4: 47 +- 4.7
    constexpr float VCaltoElectronGain_L1 = 50;      // L1:   49.6 +- 2.6
    constexpr float VCaltoElectronOffset = -60;      // L2-4: -60 +- 130
    constexpr float VCaltoElectronOffset_L1 = -670;  // L1:   -670 +- 220

    KOKKOS_INLINE_FUNCTION void calibDigis(
        bool isRun2,
        Kokkos::View<uint16_t*, KokkosExecSpace> id,
        Kokkos::View<uint16_t const*, KokkosExecSpace> x,
        Kokkos::View<uint16_t const*, KokkosExecSpace> y,
        Kokkos::View<uint16_t*, KokkosExecSpace> adc,
        SiPixelGainForHLTonGPU<KokkosExecSpace> ped,
        int numElements,
        Kokkos::View<uint32_t*, KokkosExecSpace> moduleStart,        // just to zero first
        Kokkos::View<uint32_t*, KokkosExecSpace> nClustersInModule,  // just to zero them
        Kokkos::View<uint32_t*, KokkosExecSpace> clusModuleStart,    // just to zero first
        const size_t index) {
      // zero for next kernels...
      if (0 == index) {
        clusModuleStart[0] = moduleStart[0] = 0;
      }
      if (index < gpuClustering::MaxNumModules) {
        nClustersInModule[index] = 0;
      }

      if (InvId == id[index]) {
        return;
      }

      float conversionFactor = (isRun2) ? (id[index] < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain) : 1.f;
      float offset = (isRun2) ? (id[index] < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset) : 0;

      bool isDeadColumn = false, isNoisyColumn = false;

      int row = x[index];
      int col = y[index];
      auto ret = ped.getPedAndGain(id[index], col, row, isDeadColumn, isNoisyColumn);
      float pedestal = ret.first;
      float gain = ret.second;
      // float pedestal = 0; float gain = 1.;
      if (isDeadColumn | isNoisyColumn) {
        id[index] = InvId;
        adc[index] = 0;
        printf("bad pixel at %d in %d\n", int(index), int(id[index]));
      } else {
        float vcal = adc[index] * gain - pedestal * gain;
        adc[index] = std::max(100, int(vcal * conversionFactor + offset));
      }
    }
  }  // namespace gpuCalibPixel
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
