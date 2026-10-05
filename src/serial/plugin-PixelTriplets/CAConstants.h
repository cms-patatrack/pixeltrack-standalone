#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h

#include <cstdint>

#include "CUDACore/cudaCompat.h"

#include "CUDACore/HistoContainer.h"
#include "CUDACore/SimpleVector.h"
#include "CUDACore/VecArray.h"
#include "CUDADataFormats/gpuClusteringConstants.h"

// #define ONLY_PHICUT

namespace CAConstants {

  // constants
#ifndef ONLY_PHICUT
#ifdef GPU_SMALL_EVENTS
  constexpr uint32_t maxNumberOfTuples() { return 3 * 1024; }
#else
  constexpr uint32_t maxNumberOfTuples() { return 24 * 1024; }
#endif
#else
  constexpr uint32_t maxNumberOfTuples() { return 48 * 1024; }
#endif
  constexpr uint32_t maxNumberOfQuadruplets() { return maxNumberOfTuples(); }
#ifndef ONLY_PHICUT
#ifndef GPU_SMALL_EVENTS
  constexpr uint32_t maxNumberOfDoublets() { return 512 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 128; }
#else
  constexpr uint32_t maxNumberOfDoublets() { return 128 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 128 / 2; }
#endif
#else
  constexpr uint32_t maxNumberOfDoublets() { return 2 * 1024 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 8 * 128; }
#endif
  constexpr uint32_t maxNumOfActiveDoublets() { return maxNumberOfDoublets() / 8; }

  constexpr uint32_t maxNumberOfLayerPairs() { return 20; }
  constexpr uint32_t maxNumberOfLayers() { return 10; }
  constexpr uint32_t maxTuples() { return maxNumberOfTuples(); }

  // types
  using hindex_type = uint16_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
  using tindex_type = uint16_t;  //  for tuples

#ifndef ONLY_PHICUT
  using CellNeighbors = cms::cuda::VecArray<uint32_t, 36>;
  using CellTracks = cms::cuda::VecArray<tindex_type, 48>;
#else
  using CellNeighbors = cms::cuda::VecArray<uint32_t, 64>;
  using CellTracks = cms::cuda::VecArray<tindex_type, 64>;
#endif

  using CellNeighborsVector = cms::cuda::SimpleVector<CellNeighbors>;
  using CellTracksVector = cms::cuda::SimpleVector<CellTracks>;

  using OuterHitOfCell = cms::cuda::VecArray<uint32_t, maxCellsPerHit()>;

  // Compact (CSR) storage of the cells whose outer hit is each hit, used instead of an array of OuterHitOfCell to
  // reduce the memory footprint: the cells of hit i are cells[offsets[i]], ..., cells[offsets[i + 1] - 1], in
  // increasing order, with at most maxCellsPerHit() cells per hit (like OuterHitOfCell::push_back()).
  struct OuterHitOfCellContainer {
    struct Range {
      uint32_t const* begin_;
      uint32_t const* end_;

      constexpr int size() const { return end_ - begin_; }
      constexpr bool empty() const { return begin_ == end_; }
      constexpr bool full() const { return size() == static_cast<int>(maxCellsPerHit()); }
      constexpr uint32_t const* data() const { return begin_; }
      constexpr uint32_t operator[](int i) const { return begin_[i]; }
    };

    Range operator[](uint32_t hit) const { return Range{cells + offsets[hit], cells + offsets[hit + 1]}; }

    uint32_t* offsets = nullptr;  // nHits + 1 elements
    uint32_t* cells = nullptr;    // up to maxNumberOfDoublets() elements
  };
  using TuplesContainer = cms::cuda::OneToManyAssoc<hindex_type, maxTuples(), 5 * maxTuples()>;
  using HitToTuple =
      cms::cuda::OneToManyAssoc<tindex_type, pixelGPUConstants::maxNumberOfHits, 4 * maxTuples()>;  // 3.5 should be enough
  using TupleMultiplicity = cms::cuda::OneToManyAssoc<tindex_type, 8, maxTuples()>;

}  // namespace CAConstants

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h
