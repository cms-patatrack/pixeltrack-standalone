#include "CAHitNtupletGeneratorKernels.h"
#include "CAHitNtupletGeneratorKernelsImpl.h"

#include "KokkosCore/hintLightWeight.h"

namespace KOKKOS_NAMESPACE {
  void CAHitNtupletGeneratorKernels::fillHitDetIndices(HitsView const *hv,
                                                       cms::kokkos::shared_ptr<TkSoA, KokkosDeviceMemSpace> &tracks_d,
                                                       KokkosExecSpace const &execSpace) {
    auto *hitIndices = &tracks_d->hitIndices;
    auto *detIndices = &tracks_d->detIndices;
    Kokkos::parallel_for(
        hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, HitContainer::capacity())),
        KOKKOS_LAMBDA(size_t i) { kernel_fillHitDetIndices(hitIndices, hv, detIndices, i); });
#ifdef GPU_DEBUG
    execSpace.fence();
#endif
  }

  void CAHitNtupletGeneratorKernels::launchKernels(HitsOnCPU const &hh,
                                                   cms::kokkos::shared_ptr<TkSoA, KokkosDeviceMemSpace> &tracks_d,
                                                   KokkosExecSpace const &execSpace) {
    // these are pointer on GPU!
    auto *tuples_d = &tracks_d->hitIndices;
    auto *quality_d = (Quality *)(&tracks_d->m_quality);

    // zero tuples
    cms::kokkos::launchZero(tuples_d, execSpace);

    auto nhits = hh.nHits();
    assert(nhits <= pixelGPUConstants::maxNumberOfHits);

    // std::cout << "N hits " << nhits << std::endl;
    // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

    //
    // applying conbinatoric cleaning such as fishbone at this stage is too expensive
    //

    int nthTot = 64;
    int stride = 4;
    int teamSize = nthTot / stride;
    int leagueSize = (3 * m_params.maxNumberOfDoublets_ / 4 + teamSize - 1) / teamSize;
    int rescale = leagueSize / 65536;
    teamSize *= (rescale + 1);
    leagueSize = (3 * m_params.maxNumberOfDoublets_ / 4 + teamSize - 1) / teamSize;
    assert(leagueSize < 65536);
    assert(teamSize > 0 && 0 == teamSize % 16);
    teamSize *= stride;

#if defined KOKKOS_BACKEND_SERIAL || defined KOKKOS_BACKEND_PTHREAD
    // unit team size and stride loop for host execution
    auto policy = hintLightWeight(Kokkos::TeamPolicy<KokkosExecSpace>{execSpace, leagueSize, 1});
    stride = 1;
#else
    auto policy = hintLightWeight(Kokkos::TeamPolicy<KokkosExecSpace>{execSpace, leagueSize, teamSize});
#endif
    const auto *hhp = hh.view();

    // Kokkos::View as local variables to pass to the lambda
    auto d_hitTuple_apc_ = cms::kokkos::to_view(device_hitTuple_apc_);
    auto d_hitToTuple_apc_ = cms::kokkos::to_view(device_hitToTuple_apc_);
    auto d_theCells_ = cms::kokkos::to_view(device_theCells_);
    auto d_nCells_ = cms::kokkos::to_view(device_nCells_);
    auto d_theCellNeighbors_ = cms::kokkos::to_view(device_theCellNeighbors_);
    auto d_theCellTracks_ = cms::kokkos::to_view(device_theCellTracks_);
    auto d_isOuterHitOfCell_ = cms::kokkos::to_view(device_isOuterHitOfCell_);
    auto d_tupleMultiplicity_ = cms::kokkos::to_view(device_tupleMultiplicity_);

    {
      // capturing this by the lambda leads to illegal memory access with CUDA
      auto const hardCurvCut = m_params.hardCurvCut_;
      auto const ptmin = m_params.ptmin_;
      auto const CAThetaCutBarrel = m_params.CAThetaCutBarrel_;
      auto const CAThetaCutForward = m_params.CAThetaCutForward_;
      auto const dcaCutInnerTriplet = m_params.dcaCutInnerTriplet_;
      auto const dcaCutOuterTriplet = m_params.dcaCutOuterTriplet_;
      Kokkos::parallel_for(
          "kernel_connect", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<KokkosExecSpace>::member_type &teamMember) {
            kernel_connect(d_hitTuple_apc_,
                           d_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
                           hhp,
                           d_theCells_,
                           d_nCells_,
                           d_theCellNeighbors_,
                           d_isOuterHitOfCell_,
                           hardCurvCut,
                           ptmin,
                           CAThetaCutBarrel,
                           CAThetaCutForward,
                           dcaCutInnerTriplet,
                           dcaCutOuterTriplet,
                           stride,
                           teamMember);
          });
    }

    if (nhits > 1 && m_params.earlyFishbone_) {
      int teamSize = 128;
      int stride = 16;
      int blockSize = teamSize / stride;
      int leagueSize = (nhits + blockSize - 1) / blockSize;
#if defined KOKKOS_BACKEND_SERIAL || defined KOKKOS_BACKEND_PTHREAD
      // unit team size and stride loop for host execution
      auto policy = hintLightWeight(Kokkos::TeamPolicy<KokkosExecSpace>{execSpace, leagueSize, 1});
      stride = 1;
#else
      auto policy = hintLightWeight(Kokkos::TeamPolicy<KokkosExecSpace>{execSpace, leagueSize, teamSize});
#endif
      Kokkos::parallel_for(
          "earlyfishbone", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<KokkosExecSpace>::member_type &teamMember) {
            gpuPixelDoublets::fishbone(
                hhp, d_theCells_, d_nCells_, d_isOuterHitOfCell_, nhits, false, stride, teamMember);
          });
    }

    {
      auto const minHitsPerNtuplet = m_params.minHitsPerNtuplet_;
      Kokkos::parallel_for(
          "kernel_find_ntuplets",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, m_params.maxNumberOfDoublets_)),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < d_nCells_()) {
              kernel_find_ntuplets(
                  hhp, d_theCells_, d_theCellTracks_, tuples_d, d_hitTuple_apc_, quality_d, minHitsPerNtuplet, i);
            }
          });
    }

    if (m_params.doStats_)
      Kokkos::parallel_for(
          "kernel_mark_used",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, m_params.maxNumberOfDoublets_)),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < d_nCells_()) {
              kernel_mark_used(hhp, d_theCells_, i);
            }
          });

#ifdef GPU_DEBUG
    execSpace.fence();
#endif

    cms::kokkos::finalizeBulk<HitContainer, KokkosExecSpace>(d_hitTuple_apc_, tuples_d, execSpace);

    // remove duplicates (tracks that share a doublet)
    Kokkos::parallel_for(
        "kernel_earlyDuplicateRemover",
        hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, m_params.maxNumberOfDoublets_)),
        KOKKOS_LAMBDA(const size_t i) {
          if (i < d_nCells_()) {
            kernel_earlyDuplicateRemover(d_theCells_, tuples_d, quality_d, i);
          }
        });

    Kokkos::parallel_for(
        "kernel_countMultiplicity",
        hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxTuples())),
        KOKKOS_LAMBDA(const size_t i) {
          if (i < tuples_d->nbins()) {
            kernel_countMultiplicity(tuples_d, quality_d, d_tupleMultiplicity_.data(), i);
          }
        });

    cms::kokkos::launchFinalize(d_tupleMultiplicity_, execSpace);

    Kokkos::parallel_for(
        "kernel_fillMultiplicity",
        hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxTuples())),
        KOKKOS_LAMBDA(const size_t i) {
          if (i < tuples_d->nbins()) {
            kernel_fillMultiplicity(tuples_d, quality_d, d_tupleMultiplicity_.data(), i);
          }
        });

    if (nhits > 1 && m_params.lateFishbone_) {
      int teamSize = 128;
      int stride = 16;
      int blockSize = teamSize / stride;
      int leagueSize = (nhits + blockSize - 1) / blockSize;
#if defined KOKKOS_BACKEND_SERIAL || KOKKOS_BACKEND_PTHREAD
      // unit team size and stride loop for host execution
      auto policy = hintLightWeight(Kokkos::TeamPolicy<KokkosExecSpace>{execSpace, leagueSize, 1});
      stride = 1;
#else
      auto policy = hintLightWeight(Kokkos::TeamPolicy<KokkosExecSpace>{execSpace, leagueSize, teamSize});
#endif
      Kokkos::parallel_for(
          "latefishbone", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<KokkosExecSpace>::member_type &teamMember) {
            gpuPixelDoublets::fishbone(
                hhp, d_theCells_, d_nCells_, d_isOuterHitOfCell_, nhits, true, stride, teamMember);
          });
    }
    if (m_params.doStats_) {
      teamSize = 128;
      leagueSize = (std::max(nhits, m_params.maxNumberOfDoublets_) + teamSize - 1) / teamSize;
#if defined KOKKOS_BACKEND_SERIAL || defined KOKKOS_BACKEND_PTHREAD
      policy = hintLightWeight(Kokkos::TeamPolicy<KokkosExecSpace>(execSpace, leagueSize, Kokkos::AUTO()));
#else
      policy = hintLightWeight(Kokkos::TeamPolicy<KokkosExecSpace>(execSpace, leagueSize, teamSize));
#endif
      auto maxNumberOfDoublets = m_params.maxNumberOfDoublets_;
      auto d_counters = counters_;
      Kokkos::parallel_for(
          "kernel_checkOverflows",
          policy,
          KOKKOS_LAMBDA(const Kokkos::TeamPolicy<KokkosExecSpace>::member_type &teamMember) {
            kernel_checkOverflows(tuples_d,
                                  d_tupleMultiplicity_,
                                  d_hitTuple_apc_,
                                  d_theCells_,
                                  d_nCells_,
                                  d_theCellNeighbors_,
                                  d_theCellTracks_,
                                  d_isOuterHitOfCell_,
                                  nhits,
                                  maxNumberOfDoublets,
                                  d_counters,
                                  teamMember);
          });
    }
#ifdef GPU_DEBUG
    execSpace.fence();
#endif
  }

  void CAHitNtupletGeneratorKernels::buildDoublets(HitsOnCPU const &hh, KokkosExecSpace const &execSpace) {
    std::cout << "Il fatto che io non printi questa cosa è molto strano" << '\n';
    auto nhits = hh.nHits();
    std::cout << "Vediamo se setto il numero di hits" << '\n';
#ifdef NTUPLE_DEBUG
    std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
    execSpace.fence();
#endif

    // in principle we can use "nhits" to heuristically dimension the workspace...
    device_isOuterHitOfCell_ =
        cms::kokkos::make_shared<GPUCACell::OuterHitOfCell[], KokkosDeviceMemSpace>(std::max(1U, nhits), execSpace);

    device_theCellNeighborsContainer_ = cms::kokkos::make_shared<CAConstants::CellNeighbors[], KokkosDeviceMemSpace>(
        CAConstants::maxNumOfActiveDoublets(), execSpace);
    device_theCellTracksContainer_ = cms::kokkos::make_shared<CAConstants::CellTracks[], KokkosDeviceMemSpace>(
        CAConstants::maxNumOfActiveDoublets(), execSpace);
    std::cout << "Stranamente questi device cosi non hanno dato problemi" << '\n';

    {
      auto isOuterHitOfCell = cms::kokkos::to_view(device_isOuterHitOfCell_);
      auto cellNeighbors = cms::kokkos::to_view(device_theCellNeighbors_);
      auto cellNeighborsContainer = cms::kokkos::to_view(device_theCellNeighborsContainer_);
      auto cellTracks = cms::kokkos::to_view(device_theCellTracks_);
      auto cellTracksContainer = cms::kokkos::to_view(device_theCellTracksContainer_);
      Kokkos::parallel_for(
          "initDoublets",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, nhits)),
          KOKKOS_LAMBDA(const size_t i) {
            assert(isOuterHitOfCell.data());
            isOuterHitOfCell(i).reset();

            if (0 == i) {
              cellNeighbors().construct(CAConstants::maxNumOfActiveDoublets(), cellNeighborsContainer.data());
              cellTracks().construct(CAConstants::maxNumOfActiveDoublets(), cellTracksContainer.data());
              auto j = cellNeighbors().extend();
              assert(0 == j);
              cellNeighbors()[0].reset();
              j = cellTracks().extend();
              assert(0 == j);
              cellTracks()[0].reset();
            }
          });
    }

    device_theCells_ =
        cms::kokkos::make_shared<GPUCACell[], KokkosDeviceMemSpace>(m_params.maxNumberOfDoublets_, execSpace);

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
#if defined KOKKOS_BACKEND_SERIAL || defined KOKKOS_BACKEND_PTHREAD
    int stride = 1;
    Kokkos::TeamPolicy<KokkosExecSpace,
                       Kokkos::LaunchBounds<gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize,
                                            gpuPixelDoublets::getDoubletsFromHistoMinBlocksPerMP>>
        tempPolicy{execSpace, KokkosExecSpace::impl_thread_pool_size(), 1};
#else
    int stride = 4;
    int teamSize = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
    int leagueSize = (4 * nhits + teamSize - 1) / teamSize;
    Kokkos::TeamPolicy<KokkosExecSpace,
                       Kokkos::LaunchBounds<gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize,
                                            gpuPixelDoublets::getDoubletsFromHistoMinBlocksPerMP>>
        tempPolicy{execSpace, leagueSize, teamSize * stride};
#endif
    // TODO: I do not understand why +2 is needed, the code allocates
    // one uint32_t in addition of the
    // CAConstants::maxNumberOfLayerPairs()
    tempPolicy.set_scratch_size(0, Kokkos::PerTeam((CAConstants::maxNumberOfLayerPairs() + 2) * sizeof(uint32_t)));
    const auto *hhp = hh.view();

    gpuPixelDoublets::getDoubletsFromHisto getdoublets(cms::kokkos::to_view(device_theCells_),
                                                       cms::kokkos::to_view(device_nCells_),
                                                       cms::kokkos::to_view(device_theCellNeighbors_),
                                                       cms::kokkos::to_view(device_theCellTracks_),
                                                       hhp,
                                                       cms::kokkos::to_view(device_isOuterHitOfCell_),
                                                       nActualPairs,
                                                       m_params.idealConditions_,
                                                       m_params.doClusterCut_,
                                                       m_params.doZ0Cut_,
                                                       m_params.doPtCut_,
                                                       m_params.maxNumberOfDoublets_,
                                                       stride);
    Kokkos::parallel_for("getDoubletsFromHisto", hintLightWeight(tempPolicy), getdoublets);

#ifdef GPU_DEBUG
    execSpace.fence();
#endif
  }

  void CAHitNtupletGeneratorKernels::classifyTuples(HitsOnCPU const &hh,
                                                    cms::kokkos::shared_ptr<TkSoA, KokkosDeviceMemSpace> &tracks_d,
                                                    KokkosExecSpace const &execSpace) {
    // these are pointer on GPU!
    auto *tksoa_d = tracks_d.get();
    auto const *tuples_d = &tracks_d->hitIndices;
    auto *quality_d = (Quality *)(&tracks_d->m_quality);

    {
      auto const cuts = m_params.cuts_;
      Kokkos::parallel_for(
          "kernel_classifyTracks",
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
          KOKKOS_LAMBDA(const size_t i) { kernel_classifyTracks(tuples_d, tksoa_d, cuts, quality_d, i); });
    }

    auto *theCells = device_theCells_.get();
    auto *d_nCells = device_nCells_.get();
    if (m_params.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      Kokkos::parallel_for(
          "kernel_fishboneCleaner",
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, m_params.maxNumberOfDoublets_),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < *d_nCells) {
              kernel_fishboneCleaner(theCells, quality_d, i);
            }
          });
    }

    // remove duplicates (tracks that share a doublet)
    {
      Kokkos::parallel_for(
          "kernel_fastDuplicateRemover",
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, m_params.maxNumberOfDoublets_),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < *d_nCells) {
              kernel_fastDuplicateRemover(theCells, tuples_d, tksoa_d, i);
            }
          });
    }

    auto *hitToTuple = device_hitToTuple_.get();
    if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
      // fill hit->track "map"
      Kokkos::parallel_for(
          "kernel_countHitInTracks",
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < tuples_d->nbins()) {
              kernel_countHitInTracks(tuples_d, quality_d, hitToTuple, i);
            }
          });
      cms::kokkos::launchFinalize(cms::kokkos::to_view(device_hitToTuple_), execSpace);
      Kokkos::parallel_for(
          "kernel_fillHitInTracks",
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < tuples_d->nbins()) {
              kernel_fillHitInTracks(tuples_d, quality_d, hitToTuple, i);
            }
          });
    }

    if (m_params.minHitsPerNtuplet_ < 4) {
      // remove duplicates (tracks that share a hit)
      auto hh_view = hh.view();
      Kokkos::parallel_for(
          "kernel_tripletCleaner",
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, HitToTuple::capacity()),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < hitToTuple->nbins()) {
              kernel_tripletCleaner(hh_view, tuples_d, tksoa_d, quality_d, hitToTuple, i);
            }
          });
    }

    if (m_params.doStats_) {
      // counters (add flag???)
      auto d_counters = counters_;
      Kokkos::parallel_for(
          "kernel_doStatsForHitInTracks",
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, HitToTuple::capacity()),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < hitToTuple->nbins()) {
              kernel_doStatsForHitInTracks(hitToTuple, d_counters, i);
            }
          });
      Kokkos::parallel_for(
          "kernel_doStatsForTracks",
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, CAConstants::maxNumberOfQuadruplets()),
          KOKKOS_LAMBDA(const size_t i) {
            if (i < tuples_d->nbins()) {
              kernel_doStatsForTracks(tuples_d, quality_d, d_counters, i);
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
            kernel_print_found_ntuplets(hh_view, tuples_d, tracks_d, quality_d, hitToTuple, 100, iev, i);
          }
        });
#endif
  }

  void CAHitNtupletGeneratorKernels::printCounters(
      const cms::kokkos::shared_ptr<Counters, KokkosDeviceMemSpace> &counters) {
#ifdef TODO
    kernel_printCounters<<<1, 1>>>(counters);
#endif
  }

  void CAHitNtupletGeneratorKernels::allocateOnGPU(KokkosExecSpace const &execSpace) {
    //////////////////////////////////////////////////////////
    // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
    //////////////////////////////////////////////////////////

    device_theCellNeighbors_ =
        cms::kokkos::make_shared<CAConstants::CellNeighborsVector, KokkosDeviceMemSpace>(execSpace);
    device_theCellTracks_ = cms::kokkos::make_shared<CAConstants::CellTracksVector, KokkosDeviceMemSpace>(execSpace);

    device_hitToTuple_ = cms::kokkos::make_shared<HitToTuple, KokkosDeviceMemSpace>(execSpace);

    device_tupleMultiplicity_ = cms::kokkos::make_shared<TupleMultiplicity, KokkosDeviceMemSpace>(execSpace);

    device_hitTuple_apc_ = cms::kokkos::make_shared<cms::kokkos::AtomicPairCounter, KokkosDeviceMemSpace>(execSpace);
    device_hitToTuple_apc_ = cms::kokkos::make_shared<cms::kokkos::AtomicPairCounter, KokkosDeviceMemSpace>(execSpace);
    device_nCells_ = cms::kokkos::make_shared<uint32_t, KokkosDeviceMemSpace>(execSpace);
    device_tmws_ = cms::kokkos::make_shared<uint8_t[], KokkosDeviceMemSpace>(
        std::max(TupleMultiplicity::wsSize(), HitToTuple::wsSize()), execSpace);

    Kokkos::deep_copy(execSpace, cms::kokkos::to_view(device_nCells_), 0);

    cms::kokkos::launchZero(cms::kokkos::to_view(device_tupleMultiplicity_), execSpace);
    cms::kokkos::launchZero(cms::kokkos::to_view(device_hitToTuple_),
                            execSpace);  // we may wish to keep it in the edm...
  }

}  // namespace KOKKOS_NAMESPACE
