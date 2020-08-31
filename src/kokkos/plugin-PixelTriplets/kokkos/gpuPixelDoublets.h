#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h

#include "gpuPixelDoubletsAlgos.h"

namespace KOKKOS_NAMESPACE {
  namespace gpuPixelDoublets {

    using namespace gpuPixelDoubletsAlgos;
    using namespace PixelDoubletsConstants;

    using CellNeighbors = CAConstants::CellNeighbors;
    using CellTracks = CAConstants::CellTracks;
    using CellNeighborsVector = CAConstants::CellNeighborsVector;
    using CellTracksVector = CAConstants::CellTracksVector;

    KOKKOS_INLINE_FUNCTION void initDoublets(
        Kokkos::View<GPUCACell::OuterHitOfCell*, KokkosExecSpace> isOuterHitOfCell,
        int nHits,
        Kokkos::View<CAConstants::CellNeighborsVector, KokkosExecSpace> cellNeighbors,  // not used at the moment
        Kokkos::View<CAConstants::CellTracksVector, KokkosExecSpace> cellTracks,        // not used at the moment
        const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& teamMember) {
      assert(isOuterHitOfCell.data());
      const int leagueSize = teamMember.league_size();
      const int teamSize = teamMember.team_size();
      int first = teamMember.league_rank() * teamSize + teamMember.team_rank();
      for (int i = first; i < nHits; i += leagueSize * teamSize)
        isOuterHitOfCell(i).reset();
    }

    constexpr auto getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
    constexpr auto getDoubletsFromHistoMinBlocksPerMP = 16;

    KOKKOS_INLINE_FUNCTION void getDoubletsFromHisto(
        Kokkos::View<GPUCACell*, KokkosExecSpace> cells,
        Kokkos::View<uint32_t, KokkosExecSpace> nCells,
        Kokkos::View<CAConstants::CellNeighborsVector, KokkosExecSpace> cellNeighbors,  // not used at the moment
        Kokkos::View<CAConstants::CellTracksVector, KokkosExecSpace> cellTracks,        // not used at the moment
        TrackingRecHit2DSOAView const* __restrict__ hhp,
        Kokkos::View<GPUCACell::OuterHitOfCell*, KokkosExecSpace> isOuterHitOfCell,
        int nActualPairs,
        bool ideal_cond,
        bool doClusterCut,
        bool doZ0Cut,
        bool doPtCut,
        uint32_t maxNumOfDoublets,
        const int stride,
        const Kokkos::TeamPolicy<KokkosExecSpace>::member_type& teamMember) {
      auto const& __restrict__ hh = *hhp;

      LayerPairs layerPairs;
      PhiCuts phicuts;
      MinZ minz;
      MaxZ maxz;
      MaxR maxr;

      doubletsFromHisto(layerPairs,
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
                        maxNumOfDoublets,
                        stride,
                        teamMember);
    }
  }  // namespace gpuPixelDoublets
}  // namespace KOKKOS_NAMESPACE

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h
