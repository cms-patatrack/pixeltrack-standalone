#include "CAHitNtupletGeneratorKernels.h"

void CAHitNtupletGeneratorKernels::allocateOnGPU(sycl::queue stream) {
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  device_theCellNeighbors_ = cms::sycltools::make_device_unique<CAConstants::CellNeighborsVector>(stream);
  device_theCellTracks_ = cms::sycltools::make_device_unique<CAConstants::CellTracksVector>(stream);

  device_hitToTuple_ = cms::sycltools::make_device_unique<HitToTuple>(stream);

  device_tupleMultiplicity_ = cms::sycltools::make_device_unique<TupleMultiplicity>(stream);

  device_storage_ = cms::sycltools::make_device_unique<cms::sycltools::AtomicPairCounter::c_type[]>(3, stream);

  device_hitTuple_apc_ = (cms::sycltools::AtomicPairCounter*)device_storage_.get();
  device_hitToTuple_apc_ = (cms::sycltools::AtomicPairCounter*)device_storage_.get() + 1;
  device_nCells_ = (uint32_t*)(device_storage_.get() + 2);

  stream.memset(device_nCells_, 0x00, sizeof(uint32_t));

  cms::sycltools::launchZero(device_tupleMultiplicity_.get(), stream);
  cms::sycltools::launchZero(device_hitToTuple_.get(), stream);  // we may wish to keep it in the edm...
}
