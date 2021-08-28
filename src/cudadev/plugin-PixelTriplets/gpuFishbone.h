#ifndef RecoPixelVertexing_PixelTriplets_plugins_gpuFishbone_h
#define RecoPixelVertexing_PixelTriplets_plugins_gpuFishbone_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "DataFormats/approx_atan2.h"
#include "Geometry/phase1PixelTopology.h"
#include "CUDACore/VecArray.h"
#include "CUDACore/cuda_assert.h"

#include "GPUCACell.h"

namespace gpuPixelDoublets {

  __global__ void fishbone(GPUCACell::Hits const* __restrict__ hits_p,
                           GPUCACell* cells,
                           uint32_t const* __restrict__ nCells,
                           GPUCACell::OuterHitOfCell const* __restrict__ isOuterHitOfCell,
                           uint32_t nHits,
                           bool checkTrack) {
    constexpr auto maxCellsPerHit = GPUCACell::maxCellsPerHit;

    auto const& hits = *hits_p;

    // the x index runs faster
    auto threadsPerHit = blockDim.x;
    auto hitsPerBlock = blockDim.y;
    auto numberOfBlocks = gridDim.y;
    auto hitsPerGrid = numberOfBlocks * hitsPerBlock;
    auto firstHit = threadIdx.y + blockIdx.y * hitsPerBlock;
    auto firstThread = threadIdx.x;

    float x[maxCellsPerHit];
    float y[maxCellsPerHit];
    float z[maxCellsPerHit];
    float length2[maxCellsPerHit];
    uint32_t cc[maxCellsPerHit];
    uint16_t detId[maxCellsPerHit];

    // outer loop: parallelize over the hits
    for (int hit = firstHit; hit < (int)nHits; hit += hitsPerGrid) {
      auto const& vc = isOuterHitOfCell[hit];
      auto size = vc.size();
      if (size < 2)
        continue;

      // if alligned kill one of the two.
      // in principle one could try to relax the cut (only in r-z?) for jumping-doublets
      auto const& c0 = cells[vc[0]];
      auto xo = c0.outer_x(hits);
      auto yo = c0.outer_y(hits);
      auto zo = c0.outer_z(hits);
      auto doublets = 0;
      for (int32_t i = 0; i < size; ++i) {
        auto& cell = cells[vc[i]];
        if (cell.unused())
          continue;  // for triplets equivalent to next
        if (checkTrack && cell.tracks().empty())
          continue;
        cc[doublets] = vc[i];
        detId[doublets] = cell.inner_detIndex(hits);
        x[doublets] = cell.inner_x(hits) - xo;
        y[doublets] = cell.inner_y(hits) - yo;
        z[doublets] = cell.inner_z(hits) - zo;
        length2[doublets] = x[doublets] * x[doublets] + y[doublets] * y[doublets] + z[doublets] * z[doublets];
        ++doublets;
      }
      if (doublets < 2)
        continue;

      // inner loop: parallelize over the doublets
      for (int32_t i = firstThread; i < doublets - 1; i += threadsPerHit) {
        auto& cell_i = cells[cc[i]];
        for (auto j = i + 1; j < doublets; ++j) {
          // must be different detectors (potentially in the same layer)
          if (detId[i] == detId[j])
            continue;
          auto& cell_j = cells[cc[j]];
          auto cos12 = x[i] * x[j] + y[i] * y[j] + z[i] * z[j];
          if (cos12 * cos12 >= 0.99999f * length2[i] * length2[j]) {
            // alligned: kill farthest (prefer consecutive layers)
            if (length2[i] > length2[j]) {
              cell_i.kill();
              break;
            } else {
              cell_j.kill();
            }
          }
        }  // j
      }    // i
    }      // hit
  }
}  // namespace gpuPixelDoublets

#endif  // RecoPixelVertexing_PixelTriplets_plugins_gpuFishbone_h
