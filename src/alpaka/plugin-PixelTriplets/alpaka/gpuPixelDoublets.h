#ifndef plugin_PixelTriplets_alpaka_gpuPixelDoublets_h
#define plugin_PixelTriplets_alpaka_gpuPixelDoublets_h

#include "gpuPixelDoubletsAlgos.h"

#if ALPAKA_ACC_SYCL_ENABLED
#define CONSTANT_VAR constexpr
#else
#define CONSTANT_VAR ALPAKA_STATIC_ACC_MEM_CONSTANT
#endif

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace gpuPixelDoublets {

    constexpr int nPairs = 13 + 2 + 4;
    static_assert(nPairs <= CAConstants::maxNumberOfLayerPairs());

    // start constants
    // clang-format off

  CONSTANT_VAR const uint8_t layerPairs[2 * nPairs] = {
      0, 1, 0, 4, 0, 7,              // BPIX1 (3)
      1, 2, 1, 4, 1, 7,              // BPIX2 (5)
      4, 5, 7, 8,                    // FPIX1 (8)
      2, 3, 2, 4, 2, 7, 5, 6, 8, 9,  // BPIX3 & FPIX2 (13)
      0, 2, 1, 3,                    // Jumping Barrel (15)
      0, 5, 0, 8,                    // Jumping Forward (BPIX1,FPIX2)
      4, 6, 7, 9                     // Jumping Forward (19)
  };

  constexpr int16_t phi0p05 = 522;  // round(521.52189...) = phi2short(0.05);
  constexpr int16_t phi0p06 = 626;  // round(625.82270...) = phi2short(0.06);
  constexpr int16_t phi0p07 = 730;  // round(730.12648...) = phi2short(0.07);

  CONSTANT_VAR const int16_t phicuts[nPairs]{phi0p05,
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
  //   phi0p07, phi0p07, phi0p06,phi0p06, phi0p06,phi0p06};  // relaxed cuts

  CONSTANT_VAR float const minz[nPairs] = {
      -20., 0., -30., -22., 10., -30., -70., -70., -22., 15., -30, -70., -70., -20., -22., 0, -30., -70., -70.};
  CONSTANT_VAR float const maxz[nPairs] = {
      20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.};
  CONSTANT_VAR float const maxr[nPairs] = {
      20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.};

  // end constants
    // clang-format on

    using CellNeighbors = CAConstants::CellNeighbors;
    using CellTracks = CAConstants::CellTracks;
    using CellNeighborsVector = CAConstants::CellNeighborsVector;
    using CellTracksVector = CAConstants::CellTracksVector;

    struct initDoublets {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                                    int nHits,
                                    CellNeighborsVector* cellNeighbors,
                                    CellNeighbors* cellNeighborsContainer,
                                    CellTracksVector* cellTracks,
                                    CellTracks* cellTracksContainer) const {
        ALPAKA_ASSERT_OFFLOAD(isOuterHitOfCell);
        cms::alpakatools::for_each_element_in_grid_strided(
            acc, nHits, [&](uint32_t i) { isOuterHitOfCell[i].reset(); });

        const uint32_t threadIdx(alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        if (0 == threadIdx) {
          cellNeighbors->construct(CAConstants::maxNumOfActiveDoublets(), cellNeighborsContainer);
          cellTracks->construct(CAConstants::maxNumOfActiveDoublets(), cellTracksContainer);
          // NB: Increases cellNeighbors size by 1, returns previous size which should be 0.
          [[maybe_unused]] auto i = cellNeighbors->extend(acc);
          ALPAKA_ASSERT_OFFLOAD(0 == i);
          (*cellNeighbors)[0].reset();
          // NB: Increases cellTracks size by 1, returns previous size which should be 0
          [[maybe_unused]] auto ii = cellTracks->extend(acc);
          ALPAKA_ASSERT_OFFLOAD(0 == ii);
          (*cellTracks)[0].reset();
        }
      }  // initDoublets kernel operator()
    };   // initDoublets

    constexpr auto getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
    constexpr auto getDoubletsFromHistoMinBlocksPerMP = 16;

    /*#ifdef __CUDACC__
  __launch_bounds__(getDoubletsFromHistoMaxBlockSize, getDoubletsFromHistoMinBlocksPerMP)
  #endif*/
    // TO DO: NB: Alpaka equivalent for this does not seem to exit.
    struct getDoubletsFromHisto {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    GPUCACell* cells,
                                    uint32_t* nCells,
                                    CellNeighborsVector* cellNeighbors,
                                    CellTracksVector* cellTracks,
                                    TrackingRecHit2DSoAView const* __restrict__ hhp,
                                    GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                                    int nActualPairs,
                                    bool ideal_cond,
                                    bool doClusterCut,
                                    bool doZ0Cut,
                                    bool doPtCut,
                                    uint32_t maxNumOfDoublets) const {
        auto const& __restrict__ hh = *hhp;
        doubletsFromHisto(acc,
                          layerPairs,
                          nActualPairs,
                          cells,
                          nCells,
                          cellNeighbors,
                          cellTracks,
                          hh,
                          isOuterHitOfCell,
                          phicuts,
                          minz,
                          maxz,
                          maxr,
                          ideal_cond,
                          doClusterCut,
                          doZ0Cut,
                          doPtCut,
                          maxNumOfDoublets);
      }  // getDoubletsFromHisto kernel operator()
    };   // getDoubletsFromHisto

  }  // namespace gpuPixelDoublets
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

template <typename TAcc>
struct alpaka::trait::WarpSize<ALPAKA_ACCELERATOR_NAMESPACE::gpuPixelDoublets::getDoubletsFromHisto, TAcc>
    : std::integral_constant<std::uint32_t, 32> {};

#endif  // plugin_PixelTriplets_alpaka_gpuPixelDoublets_h
