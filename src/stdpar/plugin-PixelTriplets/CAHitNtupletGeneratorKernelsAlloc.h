#include <memory>

#include "CAHitNtupletGeneratorKernels.h"

#include "CUDACore/cudaCheck.h"

void CAHitNtupletGeneratorKernelsGPU::allocateOnGPU() {
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  device_theCellNeighbors_ = std::make_unique<CAConstants::CellNeighborsVector>();
  device_theCellTracks_ = std::make_unique<CAConstants::CellTracksVector>();

  device_hitToTuple_ = std::make_unique<HitToTuple>();

  device_tupleMultiplicity_ = std::make_unique<TupleMultiplicity>();

  device_storage_ = std::make_unique<cms::cuda::AtomicPairCounter::c_type[]>(3);

  device_hitTuple_apc_ = (cms::cuda::AtomicPairCounter*)device_storage_.get();
  device_hitToTuple_apc_ = (cms::cuda::AtomicPairCounter*)device_storage_.get() + 1;
  device_nCells_ = (uint32_t*)(device_storage_.get() + 2);

  *device_nCells_ = 0;
  cms::cuda::launchZero(device_tupleMultiplicity_.get());
  cms::cuda::launchZero(device_hitToTuple_.get());  // we may wish to keep it in the edm...
}
