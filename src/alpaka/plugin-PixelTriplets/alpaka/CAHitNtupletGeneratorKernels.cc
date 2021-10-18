#include "CAHitNtupletGeneratorKernels.h"
#include "CAHitNtupletGeneratorKernelsImpl.h"

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void CAHitNtupletGeneratorKernels::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d, Queue &queue) {
    // NB: MPORTANT: This could be tuned to benefit from innermost loop.
    const auto blockSize = 128;
    const auto numberOfBlocks = (HitContainer::capacity() + blockSize - 1) / blockSize;
    const WorkDiv1 fillHitDetWorkDiv = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc1>(
            fillHitDetWorkDiv, kernel_fillHitDetIndices(), &tracks_d->hitIndices, hv, &tracks_d->detIndices));
    alpaka::wait(queue);
  }

  void CAHitNtupletGeneratorKernels::launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d, Queue &queue) {
    // these are pointer on GPU!
    auto *tuples_d = &tracks_d->hitIndices;
    auto *quality_d = (Quality *)(&tracks_d->m_quality);

    // zero tuples
    launchZero(tuples_d, queue);

    auto nhits = hh.nHits();
    assert(nhits <= pixelGPUConstants::maxNumberOfHits);

    // std::cout << "N hits " << nhits << std::endl;
    // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

    //
    // applying conbinatoric cleaning such as fishbone at this stage is too expensive
    //

    const uint32_t nthTot = 64;
    const uint32_t stride = 4;
    uint32_t blockSize = nthTot / stride;
    uint32_t numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    const uint32_t rescale = numberOfBlocks / 65536;
    blockSize *= (rescale + 1);
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    assert(numberOfBlocks < 65536);
    assert(blockSize > 0 && 0 == blockSize % 16);
    const Vec2 blks(numberOfBlocks, 1u);
    const Vec2 thrs(blockSize, stride);
    const WorkDiv2 kernelConnectWorkDiv = cms::alpakatools::make_workdiv(blks, thrs);
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc2>(kernelConnectWorkDiv,
                                       kernel_connect(),
                                       device_hitTuple_apc_.get(),
                                       device_hitToTuple_apc_.get(),  // needed only to be reset, ready for next kernel
                                       hh.view(),
                                       device_theCells_.get(),
                                       device_nCells_.get(),
                                       device_theCellNeighbors_.get(),
                                       device_isOuterHitOfCell_.get(),
                                       m_params.hardCurvCut_,
                                       m_params.ptmin_,
                                       m_params.CAThetaCutBarrel_,
                                       m_params.CAThetaCutForward_,
                                       m_params.dcaCutInnerTriplet_,
                                       m_params.dcaCutOuterTriplet_));

    if (nhits > 1 && m_params.earlyFishbone_) {
      const uint32_t nthTot = 128;
      const uint32_t stride = 16;
      const uint32_t blockSize = nthTot / stride;
      const uint32_t numberOfBlocks = (nhits + blockSize - 1) / blockSize;
      const Vec2 blks(numberOfBlocks, 1u);
      const Vec2 thrs(blockSize, stride);
      const WorkDiv2 fishboneWorkDiv = cms::alpakatools::make_workdiv(blks, thrs);

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc2>(fishboneWorkDiv,
                                                     gpuPixelDoublets::fishbone(),
                                                     hh.view(),
                                                     device_theCells_.get(),
                                                     device_nCells_.get(),
                                                     device_isOuterHitOfCell_.get(),
                                                     nhits,
                                                     false));
      alpaka::wait(queue);
    }

    blockSize = 64;
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    WorkDiv1 workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1>(workDiv1D,
                                                   kernel_find_ntuplets(),
                                                   hh.view(),
                                                   device_theCells_.get(),
                                                   device_nCells_.get(),
                                                   device_theCellTracks_.get(),
                                                   tuples_d,
                                                   device_hitTuple_apc_.get(),
                                                   quality_d,
                                                   m_params.minHitsPerNtuplet_));

    if (m_params.doStats_) {
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(
                          workDiv1D, kernel_mark_used(), hh.view(), device_theCells_.get(), device_nCells_.get()));
    }

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    blockSize = 128;
    numberOfBlocks = (HitContainer::totbins() + blockSize - 1) / blockSize;
    workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1>(
                        workDiv1D, cms::alpakatools::finalizeBulk(), device_hitTuple_apc_.get(), tuples_d));

    // remove duplicates (tracks that share a doublet)
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1>(workDiv1D,
                                                   kernel_earlyDuplicateRemover(),
                                                   device_theCells_.get(),
                                                   device_nCells_.get(),
                                                   tuples_d,
                                                   quality_d));

    blockSize = 128;
    numberOfBlocks = (3 * CAConstants::maxTuples() / 4 + blockSize - 1) / blockSize;
    workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1>(
                        workDiv1D, kernel_countMultiplicity(), tuples_d, quality_d, device_tupleMultiplicity_.get()));

    cms::alpakatools::launchFinalize(device_tupleMultiplicity_.get(), queue);

    workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1>(
                        workDiv1D, kernel_fillMultiplicity(), tuples_d, quality_d, device_tupleMultiplicity_.get()));

    if (nhits > 1 && m_params.lateFishbone_) {
      const uint32_t nthTot = 128;
      const uint32_t stride = 16;
      const uint32_t blockSize = nthTot / stride;
      const uint32_t numberOfBlocks = (nhits + blockSize - 1) / blockSize;

      const Vec2 blks(numberOfBlocks, 1u);
      const Vec2 thrs(blockSize, stride);
      const WorkDiv2 workDiv2D = cms::alpakatools::make_workdiv(blks, thrs);
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc2>(workDiv2D,
                                                     gpuPixelDoublets::fishbone(),
                                                     hh.view(),
                                                     device_theCells_.get(),
                                                     device_nCells_.get(),
                                                     device_isOuterHitOfCell_.get(),
                                                     nhits,
                                                     true));
      alpaka::wait(queue);
    }

    if (m_params.doStats_) {
      numberOfBlocks = (std::max(nhits, m_params.maxNumberOfDoublets_) + blockSize - 1) / blockSize;
      workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(workDiv1D,
                                                     kernel_checkOverflows(),
                                                     tuples_d,
                                                     device_tupleMultiplicity_.get(),
                                                     device_hitTuple_apc_.get(),
                                                     device_theCells_.get(),
                                                     device_nCells_.get(),
                                                     device_theCellNeighbors_.get(),
                                                     device_theCellTracks_.get(),
                                                     device_isOuterHitOfCell_.get(),
                                                     nhits,
                                                     m_params.maxNumberOfDoublets_,
                                                     counters_.get()));
      alpaka::wait(queue);
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    // free space asap
    // device_isOuterHitOfCell_.reset();
  }

  void CAHitNtupletGeneratorKernels::buildDoublets(HitsOnCPU const &hh, Queue &queue) {
    auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
    std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    assert(device_isOuterHitOfCell_.get());

    {
      int threadsPerBlock = 128;
      // at least one block!
      int blocks = (std::max(1U, nhits) + threadsPerBlock - 1) / threadsPerBlock;
      const WorkDiv1 workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(blocks), Vec1::all(threadsPerBlock));
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(workDiv1D,
                                                     gpuPixelDoublets::initDoublets(),
                                                     device_isOuterHitOfCell_.get(),
                                                     nhits,
                                                     device_theCellNeighbors_.get(),
                                                     device_theCellNeighborsContainer_.get(),
                                                     device_theCellTracks_.get(),
                                                     device_theCellTracksContainer_.get()));
      alpaka::wait(queue);
    }

#ifdef GPU_DEBUG
    alpaka::wait(queue);
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
    const uint32_t stride = 4;
    const uint32_t threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
    const uint32_t blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
    const Vec2 blks(blocks, 1u);
    const Vec2 thrs(threadsPerBlock, stride);
    const WorkDiv2 workDiv2D = cms::alpakatools::make_workdiv(blks, thrs);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc2>(workDiv2D,
                                                   gpuPixelDoublets::getDoubletsFromHisto(),
                                                   device_theCells_.get(),
                                                   device_nCells_.get(),
                                                   device_theCellNeighbors_.get(),
                                                   device_theCellTracks_.get(),
                                                   hh.view(),
                                                   device_isOuterHitOfCell_.get(),
                                                   nActualPairs,
                                                   m_params.idealConditions_,
                                                   m_params.doClusterCut_,
                                                   m_params.doZ0Cut_,
                                                   m_params.doPtCut_,
                                                   m_params.maxNumberOfDoublets_));
    alpaka::wait(queue);

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif
  }

  void CAHitNtupletGeneratorKernels::classifyTuples(HitsOnCPU const &hh, TkSoA *tracks_d, Queue &queue) {
    // these are pointer on GPU!
    auto const *tuples_d = &tracks_d->hitIndices;
    auto *quality_d = (Quality *)(&tracks_d->m_quality);

    const auto blockSize = 64;

    // classify tracks based on kinematics
    auto numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    WorkDiv1 workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1>(
                        workDiv1D, kernel_classifyTracks(), tuples_d, tracks_d, m_params.cuts_, quality_d));

    if (m_params.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
      workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
      alpaka::enqueue(
          queue,
          alpaka::createTaskKernel<Acc1>(
              workDiv1D, kernel_fishboneCleaner(), device_theCells_.get(), device_nCells_.get(), quality_d));
      alpaka::wait(queue);
    }

    // remove duplicates (tracks that share a doublet)
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc1>(
            workDiv1D, kernel_fastDuplicateRemover(), device_theCells_.get(), device_nCells_.get(), tuples_d, tracks_d));

    if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
      // fill hit->track "map"
      numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
      workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(
                          workDiv1D, kernel_countHitInTracks(), tuples_d, quality_d, device_hitToTuple_.get()));

      cms::alpakatools::launchFinalize(device_hitToTuple_.get(), queue);

      workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(
                          workDiv1D, kernel_fillHitInTracks(), tuples_d, quality_d, device_hitToTuple_.get()));
      alpaka::wait(queue);
    }
    if (m_params.minHitsPerNtuplet_ < 4) {
      // remove duplicates (tracks that share a hit)
      numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
      workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
      alpaka::enqueue(
          queue,
          alpaka::createTaskKernel<Acc1>(
              workDiv1D, kernel_tripletCleaner(), hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get()));
      alpaka::wait(queue);
    }

    if (m_params.doStats_) {
      // counters (add flag???)
      numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
      workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(
                          workDiv1D, kernel_doStatsForHitInTracks(), device_hitToTuple_.get(), counters_.get()));

      numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
      workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
      alpaka::enqueue(
          queue,
          alpaka::createTaskKernel<Acc1>(workDiv1D, kernel_doStatsForTracks(), tuples_d, quality_d, counters_.get()));
      alpaka::wait(queue);
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

#ifdef DUMP_GPU_TK_TUPLES
    static std::atomic<int> iev(0);
    ++iev;
    workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(1u), Vec1::all(32u));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1>(workDiv1D,
                                                   kernel_print_found_ntuplets(),
                                                   hh.view(),
                                                   tuples_d,
                                                   tracks_d,
                                                   quality_d,
                                                   device_hitToTuple_.get(),
                                                   100,
                                                   iev));
#endif
  }

  void CAHitNtupletGeneratorKernels::printCounters(Queue &queue) {
    const WorkDiv1 workDiv1D = cms::alpakatools::make_workdiv(Vec1::all(1u), Vec1::all(1u));
    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1>(workDiv1D, kernel_printCounters(), counters_.get()));
    alpaka::wait(queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
