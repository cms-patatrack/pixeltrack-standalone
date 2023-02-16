#include <memory>

#include "CAHitNtupletGeneratorKernelsImpl.h"

void CAHitNtupletGeneratorKernelsGPU::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d) {
  kernel_fillHitDetIndices(&tracks_d->hitIndices, hv, &tracks_d->detIndices);
}

void CAHitNtupletGeneratorKernelsGPU::launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d) {
  // these are pointer on GPU!
  auto *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  // zero tuples
  cms::cuda::launchZero(tuples_d);

  auto nhits = hh.nHits();
  assert(nhits <= pixelGPUConstants::maxNumberOfHits);

  // std::cout << "N hits " << nhits << std::endl;
  // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

  //
  // applying conbinatoric cleaning such as fishbone at this stage is too expensive
  //

  kernel_connect(device_hitTuple_apc_,
                 device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
                 hh.view(),
                 device_theCells_.get(),
                 device_nCells_,
                 device_theCellNeighbors_.get(),
                 device_isOuterHitOfCell_.get(),
                 m_params.hardCurvCut_,
                 m_params.ptmin_,
                 m_params.CAThetaCutBarrel_,
                 m_params.CAThetaCutForward_,
                 m_params.dcaCutInnerTriplet_,
                 m_params.dcaCutOuterTriplet_);

  if (nhits > 1 && m_params.earlyFishbone_) {
    gpuPixelDoublets::fishbone(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, false);
  }

  kernel_find_ntuplets(hh.view(),
                       device_theCells_.get(),
                       device_nCells_,
                       device_theCellTracks_.get(),
                       tuples_d,
                       device_hitTuple_apc_,
                       quality_d,
                       m_params.minHitsPerNtuplet_);

  if (m_params.doStats_)
    kernel_mark_used(hh.view(), device_theCells_.get(), device_nCells_);

  cms::cuda::finalizeBulk(device_hitTuple_apc_, tuples_d);

  // remove duplicates (tracks that share a doublet)
  kernel_earlyDuplicateRemover(device_theCells_.get(), device_nCells_, tuples_d, quality_d);

  kernel_countMultiplicity(tuples_d, quality_d, device_tupleMultiplicity_.get());
  cms::cuda::launchFinalize(device_tupleMultiplicity_.get());
  kernel_fillMultiplicity(tuples_d, quality_d, device_tupleMultiplicity_.get());

  if (nhits > 1 && m_params.lateFishbone_) {
    gpuPixelDoublets::fishbone(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, true);
  }

  if (m_params.doStats_) {
    kernel_checkOverflows(tuples_d,
                          device_tupleMultiplicity_.get(),
                          device_hitTuple_apc_,
                          device_theCells_.get(),
                          device_nCells_,
                          device_theCellNeighbors_.get(),
                          device_theCellTracks_.get(),
                          device_isOuterHitOfCell_.get(),
                          nhits,
                          m_params.maxNumberOfDoublets_,
                          counters_);
  }

  // free space asap
  // device_isOuterHitOfCell_.reset();
}

void CAHitNtupletGeneratorKernelsGPU::buildDoublets(HitsOnCPU const &hh) {
  auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

  // in principle we can use "nhits" to heuristically dimension the workspace...
  device_isOuterHitOfCell_ = std::make_unique<GPUCACell::OuterHitOfCell[]>(std::max(1U, nhits));
  assert(device_isOuterHitOfCell_.get());

  cellStorage_ =
      std::make_unique<unsigned char[]>(CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellNeighbors) +
                                        CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellTracks));
  device_theCellNeighborsContainer_ = (GPUCACell::CellNeighbors *)cellStorage_.get();
  device_theCellTracksContainer_ =
      (GPUCACell::CellTracks *)(cellStorage_.get() +
                                CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellNeighbors));

  {
    gpuPixelDoublets::initDoublets(device_isOuterHitOfCell_.get(),
                                   nhits,
                                   device_theCellNeighbors_.get(),
                                   device_theCellNeighborsContainer_,
                                   device_theCellTracks_.get(),
                                   device_theCellTracksContainer_);
  }

  device_theCells_ = std::make_unique<GPUCACell[]>(m_params.maxNumberOfDoublets_);

  if (0 == nhits)
    return;  // protect against empty events

  // FIXME avoid magic numbers
  auto nActualPairs = gpuPixelDoublets::nPairs;
  if (!m_params.includeJumpingForwardDoublets_)
    nActualPairs = 15;
  if (m_params.minHitsPerNtuplet_ > 3) {
    nActualPairs = 13;
  }
  gpuPixelDoublets::getDoubletsFromHisto(device_theCells_.get(),
                                         device_nCells_,
                                         device_theCellNeighbors_.get(),
                                         device_theCellTracks_.get(),
                                         hh.view(),
                                         device_isOuterHitOfCell_.get(),
                                         nActualPairs,
                                         m_params.idealConditions_,
                                         m_params.doClusterCut_,
                                         m_params.doZ0Cut_,
                                         m_params.doPtCut_,
                                         m_params.maxNumberOfDoublets_);
}

void CAHitNtupletGeneratorKernelsGPU::classifyTuples(HitsOnCPU const &hh, TkSoA *tracks_d) {
  // these are pointer on GPU!
  auto const *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  // classify tracks based on kinematics
  kernel_classifyTracks(tuples_d, tracks_d, m_params.cuts_, quality_d);

  if (m_params.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    kernel_fishboneCleaner(device_theCells_.get(), device_nCells_, quality_d);
  }

  // remove duplicates (tracks that share a doublet)
  kernel_fastDuplicateRemover(device_theCells_.get(), device_nCells_, tuples_d, tracks_d);

  if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
    // fill hit->track "map"
    kernel_countHitInTracks(tuples_d, quality_d, device_hitToTuple_.get());
    cms::cuda::launchFinalize(device_hitToTuple_.get());
    kernel_fillHitInTracks(tuples_d, quality_d, device_hitToTuple_.get());
  }
  if (m_params.minHitsPerNtuplet_ < 4) {
    // remove duplicates (tracks that share a hit)
    kernel_tripletCleaner(hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get());
  }

  if (m_params.doStats_) {
    // counters (add flag???)
    kernel_doStatsForHitInTracks(device_hitToTuple_.get(), counters_);
    kernel_doStatsForTracks(tuples_d, quality_d, counters_);
  }

#ifdef DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  ++iev;
  kernel_print_found_ntuplets(hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get(), 100, iev);
#endif
}

void CAHitNtupletGeneratorKernelsGPU::printCounters(Counters const *counters) { kernel_printCounters(counters); }
