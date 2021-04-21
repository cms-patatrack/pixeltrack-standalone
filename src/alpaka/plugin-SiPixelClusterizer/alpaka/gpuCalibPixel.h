#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h

#include <cstdint>
#include <cstdio>

#include "AlpakaCore/alpakaKernelCommon.h"

#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "AlpakaDataFormats/gpuClusteringConstants.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace gpuCalibPixel {

    constexpr uint16_t InvId = 9999;  // must be > MaxNumModules

    // valid for run2
    constexpr float VCaltoElectronGain = 47;         // L2-4: 47 +- 4.7
    constexpr float VCaltoElectronGain_L1 = 50;      // L1:   49.6 +- 2.6
    constexpr float VCaltoElectronOffset = -60;      // L2-4: -60 +- 130
    constexpr float VCaltoElectronOffset_L1 = -670;  // L1:   -670 +- 220

    struct calibDigis {
      template <typename T_Acc>
      ALPAKA_FN_ACC void operator()(const T_Acc& acc,
                                    bool isRun2,
                                    uint16_t* id,
                                    uint16_t const* __restrict__ x,
                                    uint16_t const* __restrict__ y,
                                    uint16_t* adc,
                                    //SiPixelGainForHLTonGPU const* __restrict__ ped,
                                    const SiPixelGainForHLTonGPU::DecodingStructure* __restrict__ v_pedestals,
                                    const SiPixelGainForHLTonGPU::RangeAndCols* __restrict__ rangeAndCols,
                                    const SiPixelGainForHLTonGPU::Fields* __restrict__ fields,
                                    int numElements,
                                    uint32_t* __restrict__ moduleStart,        // just to zero first
                                    uint32_t* __restrict__ nClustersInModule,  // just to zero them
                                    uint32_t* __restrict__ clusModuleStart     // just to zero first
      ) const {
        const uint32_t threadIdxGlobal(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);

        // zero for next kernels...
        if (threadIdxGlobal == 0) {
          clusModuleStart[0] = moduleStart[0] = 0;
        }

        cms::alpakatools::for_each_element_in_grid_strided(
            acc, gpuClustering::MaxNumModules, [&](uint32_t i) { nClustersInModule[i] = 0; });

        cms::alpakatools::for_each_element_in_grid_strided(acc, numElements, [&](uint32_t i) {
          if (id[i] != InvId) {
            float conversionFactor = (isRun2) ? (id[i] < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain) : 1.f;
            float offset = (isRun2) ? (id[i] < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset) : 0;

            bool isDeadColumn = false, isNoisyColumn = false;

            int row = x[i];
            int col = y[i];
            auto ret = SiPixelGainForHLTonGPU::getPedAndGain(
                v_pedestals, rangeAndCols, fields, id[i], col, row, isDeadColumn, isNoisyColumn);
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
    };
  }  // namespace gpuCalibPixel
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
