#include "CAHitNtupletGeneratorKernels.h"
#include "CAHitNtupletGeneratorKernelsImpl.h"

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void CAHitNtupletGeneratorKernels::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d, Queue &queue) {
    // NB: MPORTANT: This could be tuned to benefit from innermost loop.
    const auto blockSize = 128;
    const auto numberOfBlocks = (HitContainer::capacity() + blockSize - 1) / blockSize;
    const WorkDiv1D fillHitDetWorkDiv = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(
        Vec1D::all(numberOfBlocks), Vec1D::all(blockSize));
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc1D>(
            fillHitDetWorkDiv, kernel_fillHitDetIndices(), &tracks_d->hitIndices, hv, &tracks_d->detIndices));
  }

  void CAHitNtupletGeneratorKernels::launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d, Queue &queue) {
    // these are pointer on GPU!
    auto *tuples_d = &tracks_d->hitIndices;
    auto *quality_d = (Quality *)(&tracks_d->m_quality);

    // zero tuples
    launchZero(tuples_d, queue);

    auto nhits = hh.nHits();
    ALPAKA_ASSERT_OFFLOAD(nhits <= pixelGPUConstants::maxNumberOfHits);

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
    ALPAKA_ASSERT_OFFLOAD(numberOfBlocks < 65536);
    ALPAKA_ASSERT_OFFLOAD(blockSize > 0 && 0 == blockSize % 16);
    const Vec2D blks(numberOfBlocks, 1u);
    const Vec2D thrs(blockSize, stride);
    const WorkDiv2D kernelConnectWorkDiv = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(blks, thrs);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc2D>(
                        kernelConnectWorkDiv,
                        kernel_connect(),
                        alpaka::getPtrNative(device_hitTuple_apc_),
                        alpaka::getPtrNative(device_hitToTuple_apc_),  // needed only to be reset, ready for next kernel
                        hh.view(),
                        alpaka::getPtrNative(device_theCells_),
                        alpaka::getPtrNative(device_nCells_),
                        alpaka::getPtrNative(device_theCellNeighbors_),
                        alpaka::getPtrNative(device_isOuterHitOfCell_),
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
      const Vec2D blks(numberOfBlocks, 1u);
      const Vec2D thrs(blockSize, stride);
      const WorkDiv2D fishboneWorkDiv = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(blks, thrs);

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc2D>(fishboneWorkDiv,
                                                      gpuPixelDoublets::fishbone(),
                                                      hh.view(),
                                                      alpaka::getPtrNative(device_theCells_),
                                                      alpaka::getPtrNative(device_nCells_),
                                                      alpaka::getPtrNative(device_isOuterHitOfCell_),
                                                      nhits,
                                                      false));
    }

    blockSize = 64;
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    WorkDiv1D workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                                         Vec1D::all(blockSize));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                    kernel_find_ntuplets(),
                                                    hh.view(),
                                                    alpaka::getPtrNative(device_theCells_),
                                                    alpaka::getPtrNative(device_nCells_),
                                                    alpaka::getPtrNative(device_theCellTracks_),
                                                    tuples_d,
                                                    alpaka::getPtrNative(device_hitTuple_apc_),
                                                    quality_d,
                                                    m_params.minHitsPerNtuplet_));

    if (m_params.doStats_) {
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                      kernel_mark_used(),
                                                      hh.view(),
                                                      alpaka::getPtrNative(device_theCells_),
                                                      alpaka::getPtrNative(device_nCells_)));
    }

    blockSize = 128;
    numberOfBlocks = (HitContainer::totbins() + blockSize - 1) / blockSize;
    workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                               Vec1D::all(blockSize));
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc1D>(
            workDiv1D, ::cms::alpakatools::finalizeBulk(), alpaka::getPtrNative(device_hitTuple_apc_), tuples_d));

    // remove duplicates (tracks that share a doublet)
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                               Vec1D::all(blockSize));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                    kernel_earlyDuplicateRemover(),
                                                    alpaka::getPtrNative(device_theCells_),
                                                    alpaka::getPtrNative(device_nCells_),
                                                    tuples_d,
                                                    quality_d));

    blockSize = 128;
    numberOfBlocks = (3 * CAConstants::maxTuples() / 4 + blockSize - 1) / blockSize;
    workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                               Vec1D::all(blockSize));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                    kernel_countMultiplicity(),
                                                    tuples_d,
                                                    quality_d,
                                                    alpaka::getPtrNative(device_tupleMultiplicity_)));

    ::cms::alpakatools::launchFinalize(alpaka::getPtrNative(device_tupleMultiplicity_), queue);

    workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                               Vec1D::all(blockSize));
    alpaka::enqueue(
        queue,
        alpaka::createTaskKernel<Acc1D>(
            workDiv1D, kernel_fillMultiplicity(), tuples_d, quality_d, alpaka::getPtrNative(device_tupleMultiplicity_)));

    if (nhits > 1 && m_params.lateFishbone_) {
      const uint32_t nthTot = 128;
      const uint32_t stride = 16;
      const uint32_t blockSize = nthTot / stride;
      const uint32_t numberOfBlocks = (nhits + blockSize - 1) / blockSize;

      const Vec2D blks(numberOfBlocks, 1u);
      const Vec2D thrs(blockSize, stride);
      const WorkDiv2D workDiv2D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(blks, thrs);
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc2D>(workDiv2D,
                                                      gpuPixelDoublets::fishbone(),
                                                      hh.view(),
                                                      alpaka::getPtrNative(device_theCells_),
                                                      alpaka::getPtrNative(device_nCells_),
                                                      alpaka::getPtrNative(device_isOuterHitOfCell_),
                                                      nhits,
                                                      true));
    }

    if (m_params.doStats_) {
      numberOfBlocks = (std::max(nhits, m_params.maxNumberOfDoublets_) + blockSize - 1) / blockSize;
      workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                                 Vec1D::all(blockSize));
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                      kernel_checkOverflows(),
                                                      tuples_d,
                                                      alpaka::getPtrNative(device_tupleMultiplicity_),
                                                      alpaka::getPtrNative(device_hitTuple_apc_),
                                                      alpaka::getPtrNative(device_theCells_),
                                                      alpaka::getPtrNative(device_nCells_),
                                                      alpaka::getPtrNative(device_theCellNeighbors_),
                                                      alpaka::getPtrNative(device_theCellTracks_),
                                                      alpaka::getPtrNative(device_isOuterHitOfCell_),
                                                      nhits,
                                                      m_params.maxNumberOfDoublets_,
                                                      alpaka::getPtrNative(counters_)));
    }

    // free space asap
    // device_isOuterHitOfCell_.reset();
  }

  void CAHitNtupletGeneratorKernels::buildDoublets(HitsOnCPU const &hh, Queue &queue) {
    auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
    std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

    ALPAKA_ASSERT_OFFLOAD(alpaka::getPtrNative(device_isOuterHitOfCell_));

    {
      int threadsPerBlock = 128;
      // at least one block!
      int blocks = (std::max(1U, nhits) + threadsPerBlock - 1) / threadsPerBlock;
      const WorkDiv1D workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(
          Vec1D::all(blocks), Vec1D::all(threadsPerBlock));
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                      gpuPixelDoublets::initDoublets(),
                                                      alpaka::getPtrNative(device_isOuterHitOfCell_),
                                                      nhits,
                                                      alpaka::getPtrNative(device_theCellNeighbors_),
                                                      alpaka::getPtrNative(device_theCellNeighborsContainer_),
                                                      alpaka::getPtrNative(device_theCellTracks_),
                                                      alpaka::getPtrNative(device_theCellTracksContainer_)));
    }

    if (0 == nhits)
      return;  // protect against empty events

    // FIXME avoid magic numbers
    auto nActualPairs = gpuPixelDoublets::nPairs;
    if (!m_params.includeJumpingForwardDoublets_)
      nActualPairs = 15;
    if (m_params.minHitsPerNtuplet_ > 3) {
      nActualPairs = 13;
    }

    ALPAKA_ASSERT_OFFLOAD(nActualPairs <= gpuPixelDoublets::nPairs);
    const uint32_t stride = 4;
    const uint32_t threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
    const uint32_t blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
    const Vec2D blks(blocks, 1u);
    const Vec2D thrs(threadsPerBlock, stride);
    const WorkDiv2D workDiv2D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(blks, thrs);
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc2D>(workDiv2D,
                                                    gpuPixelDoublets::getDoubletsFromHisto(),
                                                    alpaka::getPtrNative(device_theCells_),
                                                    alpaka::getPtrNative(device_nCells_),
                                                    alpaka::getPtrNative(device_theCellNeighbors_),
                                                    alpaka::getPtrNative(device_theCellTracks_),
                                                    hh.view(),
                                                    alpaka::getPtrNative(device_isOuterHitOfCell_),
                                                    nActualPairs,
                                                    m_params.idealConditions_,
                                                    m_params.doClusterCut_,
                                                    m_params.doZ0Cut_,
                                                    m_params.doPtCut_,
                                                    m_params.maxNumberOfDoublets_));
  }

  void CAHitNtupletGeneratorKernels::classifyTuples(HitsOnCPU const &hh, TkSoA *tracks_d, Queue &queue) {
    // these are pointer on GPU!
    auto const *tuples_d = &tracks_d->hitIndices;
    auto *quality_d = (Quality *)(&tracks_d->m_quality);

    const auto blockSize = 64;

    // classify tracks based on kinematics
    auto numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    WorkDiv1D workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                                         Vec1D::all(blockSize));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(
                        workDiv1D, kernel_classifyTracks(), tuples_d, tracks_d, m_params.cuts_, quality_d));

    if (m_params.lateFishbone_) {
      // apply fishbone cleaning to good tracks
      numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
      workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                                 Vec1D::all(blockSize));
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                      kernel_fishboneCleaner(),
                                                      alpaka::getPtrNative(device_theCells_),
                                                      alpaka::getPtrNative(device_nCells_),
                                                      quality_d));
    }

    // remove duplicates (tracks that share a doublet)
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                               Vec1D::all(blockSize));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                    kernel_fastDuplicateRemover(),
                                                    alpaka::getPtrNative(device_theCells_),
                                                    alpaka::getPtrNative(device_nCells_),
                                                    tuples_d,
                                                    tracks_d));

    if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
      // fill hit->track "map"
      numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
      workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                                 Vec1D::all(blockSize));
      alpaka::enqueue(
          queue,
          alpaka::createTaskKernel<Acc1D>(
              workDiv1D, kernel_countHitInTracks(), tuples_d, quality_d, alpaka::getPtrNative(device_hitToTuple_)));

      ::cms::alpakatools::launchFinalize(alpaka::getPtrNative(device_hitToTuple_), queue);

      workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                                 Vec1D::all(blockSize));
      alpaka::enqueue(
          queue,
          alpaka::createTaskKernel<Acc1D>(
              workDiv1D, kernel_fillHitInTracks(), tuples_d, quality_d, alpaka::getPtrNative(device_hitToTuple_)));
    }
    if (m_params.minHitsPerNtuplet_ < 4) {
      // remove duplicates (tracks that share a hit)
      numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
      workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                                 Vec1D::all(blockSize));
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                      kernel_tripletCleaner(),
                                                      hh.view(),
                                                      tuples_d,
                                                      tracks_d,
                                                      quality_d,
                                                      alpaka::getPtrNative(device_hitToTuple_)));
    }

    if (m_params.doStats_) {
      // counters (add flag???)
      numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
      workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                                 Vec1D::all(blockSize));
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                      kernel_doStatsForHitInTracks(),
                                                      alpaka::getPtrNative(device_hitToTuple_),
                                                      alpaka::getPtrNative(counters_)));

      numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
      workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(numberOfBlocks),
                                                                                 Vec1D::all(blockSize));
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(
                          workDiv1D, kernel_doStatsForTracks(), tuples_d, quality_d, alpaka::getPtrNative(counters_)));
    }

#ifdef DUMP_GPU_TK_TUPLES
    static std::atomic<int> iev(0);
    ++iev;
    workDiv1D = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(1u), Vec1D::all(32u));
    alpaka::enqueue(queue,
                    alpaka::createTaskKernel<Acc1D>(workDiv1D,
                                                    kernel_print_found_ntuplets(),
                                                    hh.view(),
                                                    tuples_d,
                                                    tracks_d,
                                                    quality_d,
                                                    alpaka::getPtrNative(device_hitToTuple_),
                                                    100,
                                                    iev));
#endif
  }

  void CAHitNtupletGeneratorKernels::printCounters(Queue &queue) {
    const WorkDiv1D workDiv1D =
        ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(Vec1D::all(1u), Vec1D::all(1u));
    alpaka::enqueue(
        queue, alpaka::createTaskKernel<Acc1D>(workDiv1D, kernel_printCounters(), alpaka::getPtrNative(counters_)));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
