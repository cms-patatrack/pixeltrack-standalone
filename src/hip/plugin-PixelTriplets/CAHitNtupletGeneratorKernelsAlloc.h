#include "CAHitNtupletGeneratorKernels.h"

#include "CUDACore/cudaCheck.h"

template <>
#ifdef __HIPCC__
void CAHitNtupletGeneratorKernelsGPU::allocateOnGPU(hipStream_t stream) {
#else
void CAHitNtupletGeneratorKernelsCPU::allocateOnGPU(hipStream_t stream) {
#endif
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  device_theCellNeighbors_ = Traits::template make_unique<CAConstants::CellNeighborsVector>(stream);
  device_theCellTracks_ = Traits::template make_unique<CAConstants::CellTracksVector>(stream);

  device_hitToTuple_ = Traits::template make_unique<HitToTuple>(stream);

  device_tupleMultiplicity_ = Traits::template make_unique<TupleMultiplicity>(stream);

  device_storage_ = Traits::template make_unique<cms::hip::AtomicPairCounter::c_type[]>(3, stream);

  device_hitTuple_apc_ = (cms::hip::AtomicPairCounter*)device_storage_.get();
  device_hitToTuple_apc_ = (cms::hip::AtomicPairCounter*)device_storage_.get() + 1;
  device_nCells_ = (uint32_t*)(device_storage_.get() + 2);

  if
#ifndef __HIPCC__
      constexpr
#endif
      (std::is_same<Traits, cms::hipcompat::GPUTraits>::value) {
    cudaCheck(hipMemsetAsync(device_nCells_, 0, sizeof(uint32_t), stream));
  } else {
    *device_nCells_ = 0;
  }
  cms::hip::launchZero(device_tupleMultiplicity_.get(), stream);
  cms::hip::launchZero(device_hitToTuple_.get(), stream);  // we may wish to keep it in the edm...
}
