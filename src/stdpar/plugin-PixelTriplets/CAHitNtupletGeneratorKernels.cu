#include <memory>

#include "CAHitNtupletGeneratorKernelsImpl.h"

void CAHitNtupletGeneratorKernelsGPU::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d) {
  auto blockSize = 128;
  auto numberOfBlocks = (HitContainer::capacity() + blockSize - 1) / blockSize;

  kernel_fillHitDetIndices<<<numberOfBlocks, blockSize, 0>>>(
      &tracks_d->hitIndices, hv, &tracks_d->detIndices);
  cudaCheck(cudaGetLastError());
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
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

  kernel_connect<<<blks, thrs, 0>>>(
      device_hitTuple_apc_,
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
  cudaCheck(cudaGetLastError());

  if (nhits > 1 && m_params.earlyFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    gpuPixelDoublets::fishbone<<<blks, thrs, 0>>>(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, false);
    cudaCheck(cudaGetLastError());
  }

  blockSize = 64;
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  kernel_find_ntuplets<<<numberOfBlocks, blockSize, 0>>>(hh.view(),
                                                                     device_theCells_.get(),
                                                                     device_nCells_,
                                                                     device_theCellTracks_.get(),
                                                                     tuples_d,
                                                                     device_hitTuple_apc_,
                                                                     quality_d,
                                                                     m_params.minHitsPerNtuplet_);
  cudaCheck(cudaGetLastError());

  if (m_params.doStats_)
    kernel_mark_used<<<numberOfBlocks, blockSize, 0>>>(hh.view(), device_theCells_.get(), device_nCells_);
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  blockSize = 128;
  numberOfBlocks = (HitContainer::totbins() + blockSize - 1) / blockSize;
  cms::cuda::finalizeBulk<<<numberOfBlocks, blockSize, 0>>>(device_hitTuple_apc_, tuples_d);

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  kernel_earlyDuplicateRemover<<<numberOfBlocks, blockSize, 0>>>(
      device_theCells_.get(), device_nCells_, tuples_d, quality_d);
  cudaCheck(cudaGetLastError());

  blockSize = 128;
  numberOfBlocks = (3 * CAConstants::maxTuples() / 4 + blockSize - 1) / blockSize;
  kernel_countMultiplicity<<<numberOfBlocks, blockSize, 0>>>(
      tuples_d, quality_d, device_tupleMultiplicity_.get());
  cms::cuda::launchFinalize(device_tupleMultiplicity_.get());
  kernel_fillMultiplicity<<<numberOfBlocks, blockSize, 0>>>(
      tuples_d, quality_d, device_tupleMultiplicity_.get());
  cudaCheck(cudaGetLastError());

  if (nhits > 1 && m_params.lateFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    gpuPixelDoublets::fishbone<<<blks, thrs, 0>>>(
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, true);
    cudaCheck(cudaGetLastError());
  }

  if (m_params.doStats_) {
    numberOfBlocks = (std::max(nhits, m_params.maxNumberOfDoublets_) + blockSize - 1) / blockSize;
    kernel_checkOverflows<<<numberOfBlocks, blockSize, 0>>>(tuples_d,
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
    cudaCheck(cudaGetLastError());
  }
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  // free space asap
  // device_isOuterHitOfCell_.reset();
}

void CAHitNtupletGeneratorKernelsGPU::buildDoublets(HitsOnCPU const &hh) {
  auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  // in principle we can use "nhits" to heuristically dimension the workspace...
  device_isOuterHitOfCell_ = std::make_unique<GPUCACell::OuterHitOfCell[]>(std::max(1U, nhits));
  assert(device_isOuterHitOfCell_.get());

  cellStorage_ = std::make_unique<unsigned char[]>(
      CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellNeighbors) +
          CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellTracks));
  device_theCellNeighborsContainer_ = (GPUCACell::CellNeighbors *)cellStorage_.get();
  device_theCellTracksContainer_ =
      (GPUCACell::CellTracks *)(cellStorage_.get() +
                                CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellNeighbors));

  {
    int threadsPerBlock = 128;
    // at least one block!
    int blocks = (std::max(1U, nhits) + threadsPerBlock - 1) / threadsPerBlock;
    gpuPixelDoublets::initDoublets<<<blocks, threadsPerBlock, 0>>>(device_isOuterHitOfCell_.get(),
                                                                           nhits,
                                                                           device_theCellNeighbors_.get(),
                                                                           device_theCellNeighborsContainer_,
                                                                           device_theCellTracks_.get(),
                                                                           device_theCellTracksContainer_);
    cudaCheck(cudaGetLastError());
  }

  device_theCells_ = std::make_unique<GPUCACell[]>(m_params.maxNumberOfDoublets_);

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
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
  int stride = 4;
  int threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
  int blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blks(1, blocks, 1);
  dim3 thrs(stride, threadsPerBlock, 1);
  gpuPixelDoublets::getDoubletsFromHisto<<<blks, thrs, 0>>>(device_theCells_.get(),
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
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
}

void CAHitNtupletGeneratorKernelsGPU::classifyTuples(HitsOnCPU const &hh, TkSoA *tracks_d) {
  // these are pointer on GPU!
  auto const *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  auto blockSize = 64;

  // classify tracks based on kinematics
  auto numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
  kernel_classifyTracks<<<numberOfBlocks, blockSize, 0>>>(tuples_d, tracks_d, m_params.cuts_, quality_d);
  cudaCheck(cudaGetLastError());

  if (m_params.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    kernel_fishboneCleaner<<<numberOfBlocks, blockSize, 0>>>(
        device_theCells_.get(), device_nCells_, quality_d);
    cudaCheck(cudaGetLastError());
  }

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  kernel_fastDuplicateRemover<<<numberOfBlocks, blockSize, 0>>>(
      device_theCells_.get(), device_nCells_, tuples_d, tracks_d);
  cudaCheck(cudaGetLastError());

  if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
    // fill hit->track "map"
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    kernel_countHitInTracks<<<numberOfBlocks, blockSize, 0>>>(
        tuples_d, quality_d, device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
    cms::cuda::launchFinalize(device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
    kernel_fillHitInTracks<<<numberOfBlocks, blockSize, 0>>>(tuples_d, quality_d, device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
  }
  if (m_params.minHitsPerNtuplet_ < 4) {
    // remove duplicates (tracks that share a hit)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    kernel_tripletCleaner<<<numberOfBlocks, blockSize, 0>>>(
        hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get());
    cudaCheck(cudaGetLastError());
  }

  if (m_params.doStats_) {
    // counters (add flag???)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    kernel_doStatsForHitInTracks<<<numberOfBlocks, blockSize, 0>>>(device_hitToTuple_.get(), counters_);
    cudaCheck(cudaGetLastError());
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    kernel_doStatsForTracks<<<numberOfBlocks, blockSize, 0>>>(tuples_d, quality_d, counters_);
    cudaCheck(cudaGetLastError());
  }
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

#ifdef DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  ++iev;
  kernel_print_found_ntuplets<<<1, 32, 0>>>(
      hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get(), 100, iev);
#endif
}

void CAHitNtupletGeneratorKernelsGPU::printCounters(Counters const *counters) {
  kernel_printCounters<<<1, 1>>>(counters);
}
