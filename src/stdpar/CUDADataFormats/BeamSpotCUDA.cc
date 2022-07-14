#include "CUDADataFormats/BeamSpotCUDA.h"

#include "CUDACore/cudaCheck.h"
#include "CUDACore/managed_unique_ptr.h"
#include "CUDACore/ScopedSetDevice.h"

#ifndef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
void BeamSpotCUDA::memAdviseAndPrefetch(int device, cudaStream_t stream) {
  device_ = device;
#ifndef CUDAUVM_DISABLE_ADVISE
  cudaCheck(cudaMemAdvise(data_d_.get(), sizeof(BeamSpotPOD), cudaMemAdviseSetReadMostly, device));
#endif
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(data_d_.get(), sizeof(BeamSpotPOD), device, stream));
#endif
}
#endif  // CUDAUVM_DISABLE_MANAGED_BEAMSPOT

BeamSpotCUDA::~BeamSpotCUDA() {
#ifndef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
#ifndef CUDAUVM_DISABLE_ADVISE
  if (data_d_ and device_ >= 0) {
    // need to make sure a CUDA context is initialized for a thread
    cms::cuda::ScopedSetDevice setDevice(device_);
    cudaCheck(cudaMemAdvise(data_d_.get(), sizeof(BeamSpotPOD), cudaMemAdviseUnsetReadMostly, device_));
  }
#endif
#endif
}
