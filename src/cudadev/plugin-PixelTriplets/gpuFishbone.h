#ifndef RecoPixelVertexing_PixelTriplets_plugins_gpuFishbone_h
#define RecoPixelVertexing_PixelTriplets_plugins_gpuFishbone_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "DataFormats/approx_atan2.h"
#include "Geometry/phase1PixelTopology.h"
#include "CUDACore/VecArray.h"
#include "CUDACore/cuda_assert.h"

#include "GPUCACell.h"

namespace gpuPixelDoublets {

  template <int hitsPerBlock, int threadsPerHit>
  __global__ __launch_bounds__(hitsPerBlock* threadsPerHit) void fishbone(
      GPUCACell::Hits const* __restrict__ hits_p,
      GPUCACell* cells,
      uint32_t const* __restrict__ nCells,
      GPUCACell::OuterHitOfCell const* __restrict__ isOuterHitOfCell,
      uint32_t nHits,
      bool checkTrack) {
    constexpr auto maxCellsPerHit = GPUCACell::maxCellsPerHit;

    auto const& hits = *hits_p;

    // blockDim must be hitsPerBlock times threadsPerHit
    assert(blockDim.x == hitsPerBlock * threadsPerHit);

    // partition the thread block into threadsPerHit-sized tiles
    auto tile = cg::tiled_partition<threadsPerHit>(cg::this_thread_block());
    assert(hitsPerBlock == tile.meta_group_size());

    auto numberOfBlocks = gridDim.x;
    auto hitsPerGrid = numberOfBlocks * hitsPerBlock;
    auto hitInBlock = tile.meta_group_rank();  // 0 .. hitsPerBlock-1
    auto innerThread = tile.thread_rank();
    auto firstHit = hitInBlock + blockIdx.x * hitsPerBlock;

    // hitsPerBlock buffers in shared memory
    __shared__ float s_x[hitsPerBlock][maxCellsPerHit];
    __shared__ float s_y[hitsPerBlock][maxCellsPerHit];
    __shared__ float s_z[hitsPerBlock][maxCellsPerHit];
    __shared__ float s_length2[hitsPerBlock][maxCellsPerHit];
    __shared__ uint32_t s_cc[hitsPerBlock][maxCellsPerHit];
    __shared__ uint16_t s_detId[hitsPerBlock][maxCellsPerHit];
    __shared__ uint32_t s_doublets[hitsPerBlock];

    // buffer used by the current thread
    float(&x)[maxCellsPerHit] = s_x[hitInBlock];
    float(&y)[maxCellsPerHit] = s_y[hitInBlock];
    float(&z)[maxCellsPerHit] = s_z[hitInBlock];
    float(&length2)[maxCellsPerHit] = s_length2[hitInBlock];
    uint32_t(&cc)[maxCellsPerHit] = s_cc[hitInBlock];
    uint16_t(&detId)[maxCellsPerHit] = s_detId[hitInBlock];
    uint32_t& doublets = s_doublets[hitInBlock];

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
      if (innerThread == 0) {
        doublets = 0;
      }
      tile.sync();
      for (int32_t i = innerThread; i < size; i += threadsPerHit) {
        auto& cell = cells[vc[i]];
        if (cell.unused())
          continue;  // for triplets equivalent to next
        if (checkTrack && cell.tracks().empty())
          continue;
        auto index = atomicInc(&doublets, 0xFFFFFFFF);
        cc[index] = vc[i];
        detId[index] = cell.inner_detIndex(hits);
        x[index] = cell.inner_x(hits) - xo;
        y[index] = cell.inner_y(hits) - yo;
        z[index] = cell.inner_z(hits) - zo;
        length2[index] = x[index] * x[index] + y[index] * y[index] + z[index] * z[index];
      }

      tile.sync();
      if (doublets < 2)
        continue;

      // inner loop: parallelize over the doublets
      for (uint32_t i = innerThread; i < doublets - 1; i += threadsPerHit) {
        auto& cell_i = cells[cc[i]];
        for (uint32_t j = i + 1; j < doublets; ++j) {
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
