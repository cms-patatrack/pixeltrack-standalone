#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h

#include <algorithm>
#include <ranges>
#include <execution>
#include <cstdint>
#include <cstdio>

#include "CondFormats/SiPixelGainForHLTonGPU.h"

#include "gpuClusteringConstants.h"

namespace gpuCalibPixel {

  constexpr uint16_t InvId = 9999;  // must be > MaxNumModules

  // valid for run2
  constexpr float VCaltoElectronGain = 47;         // L2-4: 47 +- 4.7
  constexpr float VCaltoElectronGain_L1 = 50;      // L1:   49.6 +- 2.6
  constexpr float VCaltoElectronOffset = -60;      // L2-4: -60 +- 130
  constexpr float VCaltoElectronOffset_L1 = -670;  // L1:   -670 +- 220

  void calibDigis(bool isRun2,
                             uint16_t* id,
                             uint16_t const* __restrict__ x,
                             uint16_t const* __restrict__ y,
                             uint16_t* adc,
                             SiPixelGainForHLTonGPU const* __restrict__ ped,
                             int numElements,
                             uint32_t* __restrict__ moduleStart,        // just to zero first
                             uint32_t* __restrict__ nClustersInModule,  // just to zero them
                             uint32_t* __restrict__ clusModuleStart     // just to zero first
  ) {
    int first = 0;

    clusModuleStart[0] = moduleStart[0] = 0;
    std::fill(std::execution::par, nClustersInModule, nClustersInModule + gpuClustering::MaxNumModules, 0);

    auto iter{std::views::iota(first, numElements)};
    std::for_each(std::execution::par, std::ranges::cbegin(iter), std::ranges::cend(iter), [=](const auto i) {
      if (InvId != id[i]){
        float conversionFactor = (isRun2) ? (id[i] < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain) : 1.f;
        float offset = (isRun2) ? (id[i] < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset) : 0;

        bool isDeadColumn = false, isNoisyColumn = false;

        int row = x[i];
        int col = y[i];
        auto ret = ped->getPedAndGain(id[i], col, row, isDeadColumn, isNoisyColumn);
        float pedestal = ret.first;
        float gain = ret.second;
        // float pedestal = 0; float gain = 1.;
        if (isDeadColumn | isNoisyColumn) {
          id[i] = InvId;
          adc[i] = 0;
          printf("bad pixel at %d in %d\n", i, id[i]);
        } else {
          float vcal = adc[i] * gain - pedestal * gain;
          adc[i] = std::max(100, int(vcal * conversionFactor + offset));
        }
      }
    });
  }
}  // namespace gpuCalibPixel

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
