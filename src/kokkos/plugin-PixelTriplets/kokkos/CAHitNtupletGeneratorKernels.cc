#include "CAHitNtupletGeneratorKernels.h"
#include "CAHitNtupletGeneratorKernelsImpl.h"

namespace KOKKOS_NAMESPACE {
  void CAHitNtupletGeneratorKernels::fillHitDetIndices(HitsView const *hv,
                                                       Kokkos::View<TkSoA, KokkosExecSpace> tracks_d,
                                                       KokkosExecSpace const &execSpace) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, HitContainer::capacity()), KOKKOS_LAMBDA(size_t i) {
          kernel_fillHitDetIndices(&(tracks_d().hitIndices), hv, &(tracks_d().detIndices), i);
        });
#ifdef GPU_DEBUG
    execSpace.fence();
#endif
  }

  void CAHitNtupletGeneratorKernels::launchKernels(HitsOnCPU const &hh,
                                                   Kokkos::View<TkSoA, KokkosExecSpace> tracks_d,
                                                   KokkosExecSpace const &execSpace) {
    // these are pointer on GPU!
    auto *tuples_d = &tracks_d().hitIndices;
    auto *quality_d = (Quality *)(&tracks_d().m_quality);

    // zero tuples
    cms::kokkos::launchZero(tuples_d, execSpace);

    auto nhits = hh.nHits();
    assert(nhits <= pixelGPUConstants::maxNumberOfHits);

    // std::cout << "N hits " << nhits << std::endl;
    // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

    //
    // applying conbinatoric cleaning such as fishbone at this stage is too expensive
    //

#ifdef TODO
    auto nthTot = 64;
    auto stride = 4;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    auto rescale = numberOfBlocks / 65536;
    blockSize *= (rescale + 1);
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    assert(numberOfBlocks < 65536);
    assert(blockSize > 0 && 0 == blockSize % 16);
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);

    kernel_connect<<<blks, thrs, 0, cudaStream>>>(
        device_hitTuple_apc_,
        device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
        hh.view(),
        device_theCells_.get(),
        device_nCells_,
        device_theCellNeighbors_,
        device_isOuterHitOfCell_.get(),
        m_params.hardCurvCut_,
        m_params.ptmin_,
        m_params.CAThetaCutBarrel_,
        m_params.CAThetaCutForward_,
        m_params.dcaCutInnerTriplet_,
        m_params.dcaCutOuterTriplet_);
    cudaCheck(cudaGetLastError());

    if (nhits > 1 && m_params.earlyFishbone_) {
      auto nthTot = 128;
      auto stride = 16;
      auto blockSize = nthTot / stride;
      auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
      dim3 blks(1, numberOfBlocks, 1);
      dim3 thrs(stride, blockSize, 1);
      fishbone<<<blks, thrs, 0, cudaStream>>>(
          hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, false);
      cudaCheck(cudaGetLastError());
    }

    blockSize = 64;
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    kernel_find_ntuplets<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(),
                                                                       device_theCells_.get(),
                                                                       device_nCells_,
                                                                       device_theCellTracks_,
                                                                       tuples_d,
                                                                       device_hitTuple_apc_,
                                                                       quality_d,
                                                                       m_params.minHitsPerNtuplet_);
    cudaCheck(cudaGetLastError());

    if (m_params.doStats_)
      kernel_mark_used<<<numberOfBlocks, blockSize, 0, cudaStream>>>(hh.view(), device_theCells_.get(), device_nCells_);
    cudaCheck(cudaGetLastError());
#endif

#ifdef GPU_DEBUG
    execSpace.fence();
#endif

#ifdef TODO
    blockSize = 128;
    numberOfBlocks = (HitContainer::totbins() + blockSize - 1) / blockSize;
    cms::cuda::finalizeBulk<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_hitTuple_apc_, tuples_d);

    // remove duplicates (tracks that share a doublet)
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    kernel_earlyDuplicateRemover<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        device_theCells_.get(), device_nCells_, tuples_d, quality_d);
    cudaCheck(cudaGetLastError());

    blockSize = 128;
    numberOfBlocks = (3 * CAConstants::maxTuples() / 4 + blockSize - 1) / blockSize;
    kernel_countMultiplicity<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        tuples_d, quality_d, device_tupleMultiplicity_.get());
    cms::cuda::launchFinalize(device_tupleMultiplicity_.get(), device_tmws_, cudaStream);
    kernel_fillMultiplicity<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        tuples_d, quality_d, device_tupleMultiplicity_.get());
    cudaCheck(cudaGetLastError());

    if (nhits > 1 && m_params.lateFishbone_) {
      auto nthTot = 128;
      auto stride = 16;
      auto blockSize = nthTot / stride;
      auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
      dim3 blks(1, numberOfBlocks, 1);
      dim3 thrs(stride, blockSize, 1);
      fishbone<<<blks, thrs, 0, cudaStream>>>(
          hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, true);
      cudaCheck(cudaGetLastError());
    }

    if (m_params.doStats_) {
      numberOfBlocks = (std::max(nhits, m_params.maxNumberOfDoublets_) + blockSize - 1) / blockSize;
      kernel_checkOverflows<<<numberOfBlocks, blockSize, 0, cudaStream>>>(tuples_d,
                                                                          device_tupleMultiplicity_.get(),
                                                                          device_hitTuple_apc_,
                                                                          device_theCells_.get(),
                                                                          device_nCells_,
                                                                          device_theCellNeighbors_,
                                                                          device_theCellTracks_,
                                                                          device_isOuterHitOfCell_.get(),
                                                                          nhits,
                                                                          m_params.maxNumberOfDoublets_,
                                                                          counters_);
      cudaCheck(cudaGetLastError());
    }
#endif
#ifdef GPU_DEBUG
    execSpace.fence();
#endif
  }

  void CAHitNtupletGeneratorKernels::buildDoublets(HitsOnCPU const &hh, KokkosExecSpace const &execSpace) {
    auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
    std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
    execSpace.fence();
#endif

    // in principle we can use "nhits" to heuristically dimension the workspace...
    device_isOuterHitOfCell_ =
        Kokkos::View<GPUCACell::OuterHitOfCell *, KokkosExecSpace>("device_isOuterHitOfCell_", std::max(1U, nhits));

    // Current multi-team algorithms are usually based on CUDA implementations which
    // may not be flexible enough to become a generic parallel solution for all backends.
    // Thus team policy should be manually handled for each specific backend.
#ifdef KOKKOS_BACKEND_CUDA
    int teamSize = 128;
    // at least one league
    int leagueSize = (std::max(1U, nhits) + teamSize - 1) / teamSize;
    Kokkos::TeamPolicy<KokkosExecSpace> policy{execSpace, leagueSize, teamSize};
#else  // serial
    Kokkos::TeamPolicy<KokkosExecSpace> policy{execSpace, 1, Kokkos::AUTO()};
#endif
    Kokkos::parallel_for(
        "initDoublets", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<KokkosExecSpace>::member_type &teamMember) {
          gpuPixelDoublets::initDoublets(
              device_isOuterHitOfCell_, nhits, device_theCellNeighbors_, device_theCellTracks_, teamMember);
        });

    device_theCells_ = Kokkos::View<GPUCACell *, KokkosExecSpace>("device_theCells_", m_params.maxNumberOfDoublets_);

#ifdef GPU_DEBUG
    execSpace.fence();
#endif

    if (0 == nhits)
      return;  // protect against empty events

    // FIXME avoid magic numbers
    auto nActualPairs = gpuPixelDoublets::nPairs;
    if (!m_params.includeJumpingForwardDoublets_)
      nActualPairs = 15;
    if (m_params.minHitsPerNtuplet_ > 3) {
      nActualPairs = 13;
    }

    assert(nActualPairs <= gpuPixelDoublets::nPairs);
#ifdef KOKKOS_BACKEND_CUDA
    int stride = 4;
    teamSize = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
    leagueSize = (4 * nhits + teamSize - 1) / teamSize;
    Kokkos::TeamPolicy<KokkosExecSpace,
                       Kokkos::LaunchBounds<gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize,
                                            gpuPixelDoublets::getDoubletsFromHistoMinBlocksPerMP>>
        tempPolicy{execSpace, leagueSize, teamSize * stride};
#else
    int stride = 1;
    Kokkos::TeamPolicy<KokkosExecSpace,
                       Kokkos::LaunchBounds<gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize,
                                            gpuPixelDoublets::getDoubletsFromHistoMinBlocksPerMP>>
        tempPolicy{execSpace, 1, Kokkos::AUTO()};
#endif
    tempPolicy.set_scratch_size(0, Kokkos::PerTeam(84));  // 21 x 4 = 84 bytes is required
    const auto *hhp = hh.view();

    gpuPixelDoublets::getDoubletsFromHisto getdoublets(device_theCells_,
                                                       device_nCells_,
                                                       device_theCellNeighbors_,
                                                       device_theCellTracks_,
                                                       hhp,
                                                       device_isOuterHitOfCell_,
                                                       nActualPairs,
                                                       m_params.idealConditions_,
                                                       m_params.doClusterCut_,
                                                       m_params.doZ0Cut_,
                                                       m_params.doPtCut_,
                                                       m_params.maxNumberOfDoublets_,
                                                       stride);
    Kokkos::parallel_for("getDoubletsFromHisto", tempPolicy, getdoublets);

#ifdef GPU_DEBUG
    execSpace.fence();
#endif
  }

  void CAHitNtupletGeneratorKernels::classifyTuples(HitsOnCPU const &hh,
                                                    Kokkos::View<TkSoA, KokkosExecSpace> tracks_d,
                                                    KokkosExecSpace const &execSpace) {
    // these are pointer on GPU!
    auto const *tuples_d = &tracks_d().hitIndices;
    auto *quality_d = (Quality *)(&tracks_d().m_quality);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
        KOKKOS_LAMBDA(const size_t i) {
          kernel_classifyTracks(tuples_d, tracks_d.data(), m_params.cuts_, quality_d, i);
        });

    if (m_params.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < device_nCells_()) {
              kernel_fishboneCleaner(device_theCells_.data(), quality_d, i);
            }
          });
    }

    // remove duplicates (tracks that share a doublet)
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
        KOKKOS_LAMBDA(const size_t i) {
          if (i < device_nCells_()) {
            kernel_fastDuplicateRemover(device_theCells_.data(), tuples_d, tracks_d.data(), i);
          }
        });

    if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
      // fill hit->track "map"
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < tuples_d->nbins()) {
              kernel_countHitInTracks(tuples_d, quality_d, device_hitToTuple_.data(), i);
            }
          });
      cms::kokkos::launchFinalize(device_hitToTuple_, execSpace);
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < tuples_d->nbins()) {
              kernel_fillHitInTracks(tuples_d, quality_d, device_hitToTuple_.data(), i);
            }
          });
    }

    if (m_params.minHitsPerNtuplet_ < 4) {
      // remove duplicates (tracks that share a hit)
      auto hh_view = hh.view();
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, HitToTuple::capacity()), KOKKOS_LAMBDA(const size_t i) {
            if (i < device_hitToTuple_().nbins()) {
              kernel_tripletCleaner(hh_view, tuples_d, tracks_d.data(), quality_d, device_hitToTuple_.data(), i);
            }
          });
    }

    if (m_params.doStats_) {
      // counters (add flag???)
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, HitToTuple::capacity()), KOKKOS_LAMBDA(const size_t i) {
            if (i < device_hitToTuple_().nbins()) {
              kernel_doStatsForHitInTracks(device_hitToTuple_.data(), counters_, i);
            }
          });
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < tuples_d->nbins()) {
              kernel_doStatsForTracks(tuples_d, quality_d, counters_, i);
            }
          });
    }

#ifdef GPU_DEBUG
    execSpace.fence();
#endif

#ifdef DUMP_GPU_TK_TUPLES
    static std::atomic<int> iev(0);
    ++iev;
    auto hh_view = hh.view();
    auto const maxPrint = 100;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
        KOKKOS_LAMBDA(const size_t i) {
          if (i < std::min(maxPrint, tuples_d->nbins() =)) {
            kernel_print_found_ntuplets(hh_view, tuples_d, tracks_d, quality_d, device_hitToTuple_data(), 100, iev, i);
          }
        });
#endif
  }

  void CAHitNtupletGeneratorKernels::printCounters(Kokkos::View<Counters const, KokkosExecSpace> counters) {
#ifdef TODO
    kernel_printCounters<<<1, 1>>>(counters);
#endif
  }

  void CAHitNtupletGeneratorKernels::allocateOnGPU(KokkosExecSpace const &execSpace) {
    //////////////////////////////////////////////////////////
    // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
    //////////////////////////////////////////////////////////

    device_hitToTuple_ = Kokkos::View<HitToTuple, KokkosExecSpace>("device_hitToTuple_");

    device_tupleMultiplicity_ = Kokkos::View<TupleMultiplicity, KokkosExecSpace>("device_tupleMultiplicity_");

    device_hitTuple_apc_ = Kokkos::View<AtomicPairCounter, KokkosExecSpace>("device_hitTuple_apc_");
    device_hitToTuple_apc_ = Kokkos::View<AtomicPairCounter, KokkosExecSpace>("device_hitToTuple_apc_");
    device_nCells_ = Kokkos::View<uint32_t, KokkosExecSpace>("device_nCells_");
    device_tmws_ = Kokkos::View<uint8_t *, KokkosExecSpace>(
        "device_tmws_", std::max(TupleMultiplicity::wsSize(), HitToTuple::wsSize()));

    Kokkos::deep_copy(execSpace, device_nCells_, 0);

    cms::kokkos::launchZero(device_tupleMultiplicity_, execSpace);
    cms::kokkos::launchZero(device_hitToTuple_, execSpace);  // we may wish to keep it in the edm...
  }

}  // namespace KOKKOS_NAMESPACE
