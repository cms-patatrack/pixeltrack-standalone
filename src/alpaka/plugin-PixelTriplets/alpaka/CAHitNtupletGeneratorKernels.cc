#include <algorithm>
#include <atomic>
#ifdef NTUPLE_DEBUG
#include <iostream>
#endif

#include "AlpakaCore/config.h"
#include "AlpakaCore/workdivision.h"

#include "CAHitNtupletGeneratorKernels.h"
#include "CAHitNtupletGeneratorKernelsImpl.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void CAHitNtupletGeneratorKernels::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d, Queue &queue) {
    // NB: MPORTANT: This could be tuned to benefit from innermost loop.
    const auto blockSize = 128;
    const auto numberOfBlocks = cms::alpakatools::divide_up_by(HitContainer::capacity(), blockSize);
    const auto fillHitDetWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc1D>(
            fillHitDetWorkDiv, kernel_fillHitDetIndices(), &tracks_d->hitIndices, hv, &tracks_d->detIndices));
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif
  }

  void CAHitNtupletGeneratorKernels::launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d, Queue &queue) {
    // these are pointer on GPU!
    auto *tuples_d = &tracks_d->hitIndices;
    auto *quality_d = (Quality *)(&tracks_d->m_quality);

    // zero tuples
    cms::alpakatools::launchZero<Acc1D>(tuples_d, queue);

    auto nhits = hh.nHits();
    ALPAKA_ASSERT_ACC(nhits <= pixelGPUConstants::maxNumberOfHits);

    // std::cout << "N hits " << nhits << std::endl;
    // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

    //
    // applying conbinatoric cleaning such as fishbone at this stage is too expensive
    //

    const uint32_t nthTot = 64;
    const uint32_t stride = 4;
    uint32_t blockSize = nthTot / stride;
    uint32_t numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.maxNumberOfDoublets_ / 4, blockSize);
    const uint32_t rescale = numberOfBlocks / 65536;
    blockSize *= (rescale + 1);
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.maxNumberOfDoublets_ / 4, blockSize);
    ALPAKA_ASSERT_ACC(numberOfBlocks < 65536);
    ALPAKA_ASSERT_ACC(blockSize > 0 && 0 == blockSize % 16);
    const Vec2D blks(numberOfBlocks, 1u);
    const Vec2D thrs(blockSize, stride);
    const auto kernelConnectWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc2D>(kernelConnectWorkDiv,
                                        kernel_connect(),
                                        device_hitTuple_apc_,
                                        device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
                                        hh.view(),
                                        device_theCells_.data(),
                                        device_nCells_.data(),
                                        device_theCellNeighbors_.data(),
                                        device_isOuterHitOfCell_.data(),
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
      const uint32_t numberOfBlocks = cms::alpakatools::divide_up_by(nhits, blockSize);
      const Vec2D blks{numberOfBlocks, 1u};
      const Vec2D thrs{blockSize, stride};
      const auto fishboneWorkDiv = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc2D>(fishboneWorkDiv,
                                                      gpuPixelDoublets::fishbone(),
                                                      hh.view(),
                                                      device_theCells_.data(),
                                                      device_nCells_.data(),
                                                      device_isOuterHitOfCell_.data(),
                                                      nhits,
                                                      false));
    }

    blockSize = 64;
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.maxNumberOfDoublets_ / 4, blockSize);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                    kernel_find_ntuplets(),
                                                    hh.view(),
                                                    device_theCells_.data(),
                                                    device_nCells_.data(),
                                                    device_theCellTracks_.data(),
                                                    tuples_d,
                                                    device_hitTuple_apc_,
                                                    quality_d,
                                                    m_params.minHitsPerNtuplet_));

    if (m_params.doStats_) {
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(
                          workDiv1D, kernel_mark_used(), hh.view(), device_theCells_.data(), device_nCells_.data()));
    }

#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(HitContainer::totbins(), blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc1D>(workDiv1D, cms::alpakatools::finalizeBulk(), device_hitTuple_apc_, tuples_d));

    // remove duplicates (tracks that share a doublet)
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.maxNumberOfDoublets_ / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                    kernel_earlyDuplicateRemover(),
                                                    device_theCells_.data(),
                                                    device_nCells_.data(),
                                                    tuples_d,
                                                    quality_d));

    blockSize = 128;
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * CAConstants::maxTuples() / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(
                        workDiv1D, kernel_countMultiplicity(), tuples_d, quality_d, device_tupleMultiplicity_.data()));

    cms::alpakatools::launchFinalize<Acc1D>(device_tupleMultiplicity_.data(), queue);

    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(
                        workDiv1D, kernel_fillMultiplicity(), tuples_d, quality_d, device_tupleMultiplicity_.data()));

    if (nhits > 1 && m_params.lateFishbone_) {
      const uint32_t nthTot = 128;
      const uint32_t stride = 16;
      const uint32_t blockSize = nthTot / stride;
      const uint32_t numberOfBlocks = cms::alpakatools::divide_up_by(nhits, blockSize);

      const Vec2D blks{numberOfBlocks, 1u};
      const Vec2D thrs{blockSize, stride};
      const auto workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc2D>(workDiv2D,
                                                      gpuPixelDoublets::fishbone(),
                                                      hh.view(),
                                                      device_theCells_.data(),
                                                      device_nCells_.data(),
                                                      device_isOuterHitOfCell_.data(),
                                                      nhits,
                                                      true));
    }

    if (m_params.doStats_) {
      numberOfBlocks = cms::alpakatools::divide_up_by(std::max(nhits, m_params.maxNumberOfDoublets_), blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                      kernel_checkOverflows(),
                                                      tuples_d,
                                                      device_tupleMultiplicity_.data(),
                                                      device_hitTuple_apc_,
                                                      device_theCells_.data(),
                                                      device_nCells_.data(),
                                                      device_theCellNeighbors_.data(),
                                                      device_theCellTracks_.data(),
                                                      device_isOuterHitOfCell_.data(),
                                                      nhits,
                                                      m_params.maxNumberOfDoublets_,
                                                      counters_.data()));
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

    ALPAKA_ASSERT_ACC(device_isOuterHitOfCell_.data());

    {
      int threadsPerBlock = 128;
      // at least one block!
      int blocks = std::max(1u, cms::alpakatools::divide_up_by(nhits, threadsPerBlock));
      const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(blocks, threadsPerBlock);
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                      gpuPixelDoublets::initDoublets(),
                                                      device_isOuterHitOfCell_.data(),
                                                      nhits,
                                                      device_theCellNeighbors_.data(),
                                                      device_theCellNeighborsContainer_,
                                                      device_theCellTracks_.data(),
                                                      device_theCellTracksContainer_));
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

    ALPAKA_ASSERT_ACC(nActualPairs <= gpuPixelDoublets::nPairs);
    const uint32_t stride = 4;
    const uint32_t threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
    const uint32_t blocks = cms::alpakatools::divide_up_by(4 * nhits, threadsPerBlock);
    const Vec2D blks(blocks, 1u);
    const Vec2D thrs(threadsPerBlock, stride);
    const auto workDiv2D = cms::alpakatools::make_workdiv<Acc2D>(blks, thrs);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc2D>(workDiv2D,
                                                    gpuPixelDoublets::getDoubletsFromHisto(),
                                                    device_theCells_.data(),
                                                    device_nCells_.data(),
                                                    device_theCellNeighbors_.data(),
                                                    device_theCellTracks_.data(),
                                                    hh.view(),
                                                    device_isOuterHitOfCell_.data(),
                                                    nActualPairs,
                                                    m_params.idealConditions_,
                                                    m_params.doClusterCut_,
                                                    m_params.doZ0Cut_,
                                                    m_params.doPtCut_,
                                                    m_params.maxNumberOfDoublets_));

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
    auto numberOfBlocks = cms::alpakatools::divide_up_by(3 * CAConstants::maxNumberOfQuadruplets() / 4, blockSize);
    auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(
                        workDiv1D, kernel_classifyTracks(), tuples_d, tracks_d, m_params.cuts_, quality_d));

    if (m_params.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.maxNumberOfDoublets_ / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::enqueue(
          queue,
          alpaka::createTaskKernel<Acc1D>(
              workDiv1D, kernel_fishboneCleaner(), device_theCells_.data(), device_nCells_.data(), quality_d));
    }

    // remove duplicates (tracks that share a doublet)
    numberOfBlocks = cms::alpakatools::divide_up_by(3 * m_params.maxNumberOfDoublets_ / 4, blockSize);
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                    kernel_fastDuplicateRemover(),
                                                    device_theCells_.data(),
                                                    device_nCells_.data(),
                                                    tuples_d,
                                                    tracks_d));

    if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
      // fill hit->track "map"
      numberOfBlocks = cms::alpakatools::divide_up_by(3 * CAConstants::maxNumberOfQuadruplets() / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(
                          workDiv1D, kernel_countHitInTracks(), tuples_d, quality_d, device_hitToTuple_.data()));

      cms::alpakatools::launchFinalize<Acc1D>(device_hitToTuple_.data(), queue);

      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(
                          workDiv1D, kernel_fillHitInTracks(), tuples_d, quality_d, device_hitToTuple_.data()));
    }
    if (m_params.minHitsPerNtuplet_ < 4) {
      // remove duplicates (tracks that share a hit)
      numberOfBlocks = cms::alpakatools::divide_up_by(HitToTuple::capacity(), blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::enqueue(
          queue,
          alpaka::createTaskKernel<Acc1D>(
              workDiv1D, kernel_tripletCleaner(), hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.data()));
    }

    if (m_params.doStats_) {
      // counters (add flag???)
      numberOfBlocks = cms::alpakatools::divide_up_by(HitToTuple::capacity(), blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(
                          workDiv1D, kernel_doStatsForHitInTracks(), device_hitToTuple_.data(), counters_.data()));

      numberOfBlocks = cms::alpakatools::divide_up_by(3 * CAConstants::maxNumberOfQuadruplets() / 4, blockSize);
      workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
      alpaka::enqueue(
          queue,
          alpaka::createTaskKernel<Acc1D>(workDiv1D, kernel_doStatsForTracks(), tuples_d, quality_d, counters_.data()));
    }
#ifdef GPU_DEBUG
    alpaka::wait(queue);
#endif

#ifdef DUMP_GPU_TK_TUPLES
    static std::atomic<int> iev(0);
    ++iev;
    workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1u, 32u);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                    kernel_print_found_ntuplets(),
                                                    hh.view(),
                                                    tuples_d,
                                                    tracks_d,
                                                    quality_d,
                                                    device_hitToTuple_.data(),
                                                    100,
                                                    iev));
#endif
  }

  void CAHitNtupletGeneratorKernels::printCounters(Queue &queue) {
    const auto workDiv1D = cms::alpakatools::make_workdiv<Acc1D>(1u, 1u);
    alpaka::enqueue(queue, alpaka::createTaskKernel<Acc1D>(workDiv1D, kernel_printCounters(), counters_.data()));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
