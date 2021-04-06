#include "hip/hip_runtime.h"
#include "CAHitNtupletGeneratorKernelsImpl.h"

template <>
void CAHitNtupletGeneratorKernelsGPU::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d, hipStream_t cudaStream) {
  auto blockSize = 128;
  auto numberOfBlocks = (HitContainer::capacity() + blockSize - 1) / blockSize;

  hipLaunchKernelGGL(kernel_fillHitDetIndices,
                     dim3(numberOfBlocks),
                     dim3(blockSize),
                     0,
                     cudaStream,
                     &tracks_d->hitIndices,
                     hv,
                     &tracks_d->detIndices);
  cudaCheck(hipGetLastError());
#ifdef GPU_DEBUG
  hipDeviceSynchronize();
  cudaCheck(hipGetLastError());
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d, hipStream_t cudaStream) {
  // these are pointer on GPU!
  auto *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  // zero tuples
  cms::hip::launchZero(tuples_d, cudaStream);

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

  hipLaunchKernelGGL(kernel_connect,
                     dim3(blks),
                     dim3(thrs),
                     0,
                     cudaStream,
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
  cudaCheck(hipGetLastError());

  if (nhits > 1 && m_params.earlyFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    hipLaunchKernelGGL(gpuPixelDoublets::fishbone,
                       dim3(blks),
                       dim3(thrs),
                       0,
                       cudaStream,
                       hh.view(),
                       device_theCells_.get(),
                       device_nCells_,
                       device_isOuterHitOfCell_.get(),
                       nhits,
                       false);
    cudaCheck(hipGetLastError());
  }

  blockSize = 64;
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(kernel_find_ntuplets,
                     dim3(numberOfBlocks),
                     dim3(blockSize),
                     0,
                     cudaStream,
                     hh.view(),
                     device_theCells_.get(),
                     device_nCells_,
                     device_theCellTracks_.get(),
                     tuples_d,
                     device_hitTuple_apc_,
                     quality_d,
                     m_params.minHitsPerNtuplet_);
  cudaCheck(hipGetLastError());

  if (m_params.doStats_)
    hipLaunchKernelGGL(kernel_mark_used,
                       dim3(numberOfBlocks),
                       dim3(blockSize),
                       0,
                       cudaStream,
                       hh.view(),
                       device_theCells_.get(),
                       device_nCells_);
  cudaCheck(hipGetLastError());

#ifdef GPU_DEBUG
  hipDeviceSynchronize();
  cudaCheck(hipGetLastError());
#endif

  blockSize = 128;
  numberOfBlocks = (HitContainer::totbins() + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(
      cms::hip::finalizeBulk, dim3(numberOfBlocks), dim3(blockSize), 0, cudaStream, device_hitTuple_apc_, tuples_d);

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(kernel_earlyDuplicateRemover,
                     dim3(numberOfBlocks),
                     dim3(blockSize),
                     0,
                     cudaStream,
                     device_theCells_.get(),
                     device_nCells_,
                     tuples_d,
                     quality_d);
  cudaCheck(hipGetLastError());

  blockSize = 128;
  numberOfBlocks = (3 * CAConstants::maxTuples() / 4 + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(kernel_countMultiplicity,
                     dim3(numberOfBlocks),
                     dim3(blockSize),
                     0,
                     cudaStream,
                     tuples_d,
                     quality_d,
                     device_tupleMultiplicity_.get());
  cms::hip::launchFinalize(device_tupleMultiplicity_.get(), cudaStream);
  hipLaunchKernelGGL(kernel_fillMultiplicity,
                     dim3(numberOfBlocks),
                     dim3(blockSize),
                     0,
                     cudaStream,
                     tuples_d,
                     quality_d,
                     device_tupleMultiplicity_.get());
  cudaCheck(hipGetLastError());

  if (nhits > 1 && m_params.lateFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    hipLaunchKernelGGL(gpuPixelDoublets::fishbone,
                       dim3(blks),
                       dim3(thrs),
                       0,
                       cudaStream,
                       hh.view(),
                       device_theCells_.get(),
                       device_nCells_,
                       device_isOuterHitOfCell_.get(),
                       nhits,
                       true);
    cudaCheck(hipGetLastError());
  }

  if (m_params.doStats_) {
    numberOfBlocks = (std::max(nhits, m_params.maxNumberOfDoublets_) + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(kernel_checkOverflows,
                       dim3(numberOfBlocks),
                       dim3(blockSize),
                       0,
                       cudaStream,
                       tuples_d,
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
    cudaCheck(hipGetLastError());
  }
#ifdef GPU_DEBUG
  hipDeviceSynchronize();
  cudaCheck(hipGetLastError());
#endif

  // free space asap
  // device_isOuterHitOfCell_.reset();
}

template <>
void CAHitNtupletGeneratorKernelsGPU::buildDoublets(HitsOnCPU const &hh, hipStream_t stream) {
  auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
  hipDeviceSynchronize();
  cudaCheck(hipGetLastError());
#endif

  // in principle we can use "nhits" to heuristically dimension the workspace...
  device_isOuterHitOfCell_ = cms::hip::make_device_unique<GPUCACell::OuterHitOfCell[]>(std::max(1U, nhits), stream);
  assert(device_isOuterHitOfCell_.get());

  cellStorage_ = cms::hip::make_device_unique<unsigned char[]>(
      CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellNeighbors) +
          CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellTracks),
      stream);
  device_theCellNeighborsContainer_ = (GPUCACell::CellNeighbors *)cellStorage_.get();
  device_theCellTracksContainer_ =
      (GPUCACell::CellTracks *)(cellStorage_.get() +
                                CAConstants::maxNumOfActiveDoublets() * sizeof(GPUCACell::CellNeighbors));

  {
    int threadsPerBlock = 128;
    // at least one block!
    int blocks = (std::max(1U, nhits) + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(gpuPixelDoublets::initDoublets,
                       dim3(blocks),
                       dim3(threadsPerBlock),
                       0,
                       stream,
                       device_isOuterHitOfCell_.get(),
                       nhits,
                       device_theCellNeighbors_.get(),
                       device_theCellNeighborsContainer_,
                       device_theCellTracks_.get(),
                       device_theCellTracksContainer_);
    cudaCheck(hipGetLastError());
  }

  device_theCells_ = cms::hip::make_device_unique<GPUCACell[]>(m_params.maxNumberOfDoublets_, stream);

#ifdef GPU_DEBUG
  hipDeviceSynchronize();
  cudaCheck(hipGetLastError());
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
  hipLaunchKernelGGL(gpuPixelDoublets::getDoubletsFromHisto,
                     dim3(blks),
                     dim3(thrs),
                     0,
                     stream,
                     device_theCells_.get(),
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
  cudaCheck(hipGetLastError());

#ifdef GPU_DEBUG
  hipDeviceSynchronize();
  cudaCheck(hipGetLastError());
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::classifyTuples(HitsOnCPU const &hh, TkSoA *tracks_d, hipStream_t cudaStream) {
  // these are pointer on GPU!
  auto const *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  auto blockSize = 64;

  // classify tracks based on kinematics
  auto numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(kernel_classifyTracks,
                     dim3(numberOfBlocks),
                     dim3(blockSize),
                     0,
                     cudaStream,
                     tuples_d,
                     tracks_d,
                     m_params.cuts_,
                     quality_d);
  cudaCheck(hipGetLastError());

  if (m_params.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(kernel_fishboneCleaner,
                       dim3(numberOfBlocks),
                       dim3(blockSize),
                       0,
                       cudaStream,
                       device_theCells_.get(),
                       device_nCells_,
                       quality_d);
    cudaCheck(hipGetLastError());
  }

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  hipLaunchKernelGGL(kernel_fastDuplicateRemover,
                     dim3(numberOfBlocks),
                     dim3(blockSize),
                     0,
                     cudaStream,
                     device_theCells_.get(),
                     device_nCells_,
                     tuples_d,
                     tracks_d);
  cudaCheck(hipGetLastError());

  if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
    // fill hit->track "map"
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(kernel_countHitInTracks,
                       dim3(numberOfBlocks),
                       dim3(blockSize),
                       0,
                       cudaStream,
                       tuples_d,
                       quality_d,
                       device_hitToTuple_.get());
    cudaCheck(hipGetLastError());
    cms::hip::launchFinalize(device_hitToTuple_.get(), cudaStream);
    cudaCheck(hipGetLastError());
    hipLaunchKernelGGL(kernel_fillHitInTracks,
                       dim3(numberOfBlocks),
                       dim3(blockSize),
                       0,
                       cudaStream,
                       tuples_d,
                       quality_d,
                       device_hitToTuple_.get());
    cudaCheck(hipGetLastError());
  }
  if (m_params.minHitsPerNtuplet_ < 4) {
    // remove duplicates (tracks that share a hit)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(kernel_tripletCleaner,
                       dim3(numberOfBlocks),
                       dim3(blockSize),
                       0,
                       cudaStream,
                       hh.view(),
                       tuples_d,
                       tracks_d,
                       quality_d,
                       device_hitToTuple_.get());
    cudaCheck(hipGetLastError());
  }

  if (m_params.doStats_) {
    // counters (add flag???)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(kernel_doStatsForHitInTracks,
                       dim3(numberOfBlocks),
                       dim3(blockSize),
                       0,
                       cudaStream,
                       device_hitToTuple_.get(),
                       counters_);
    cudaCheck(hipGetLastError());
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(
        kernel_doStatsForTracks, dim3(numberOfBlocks), dim3(blockSize), 0, cudaStream, tuples_d, quality_d, counters_);
    cudaCheck(hipGetLastError());
  }
#ifdef GPU_DEBUG
  hipDeviceSynchronize();
  cudaCheck(hipGetLastError());
#endif

#ifdef DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  ++iev;
  hipLaunchKernelGGL(kernel_print_found_ntuplets,
                     dim3(1),
                     dim3(32),
                     0,
                     cudaStream,
                     hh.view(),
                     tuples_d,
                     tracks_d,
                     quality_d,
                     device_hitToTuple_.get(),
                     100,
                     iev);
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::printCounters(Counters const *counters) {
  hipLaunchKernelGGL(kernel_printCounters, dim3(1), dim3(1), 0, 0, counters);
}
