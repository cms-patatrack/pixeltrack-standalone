#include "CUDADataFormats/BeamSpotCUDA.h"

#include "CUDACore/cudaCheck.h"
#include "CUDACore/managed_unique_ptr.h"
#include "CUDACore/ScopedSetDevice.h"

#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
BeamSpotCUDA::BeamSpotCUDA(Data const* data_h, cudaStream_t stream) {
  data_d_ = cms::cuda::make_device_unique<Data>(stream);
  cudaCheck(cudaMemcpyAsync(data_d_.get(), data_h, sizeof(Data), cudaMemcpyHostToDevice, stream));
}
#else
BeamSpotCUDA::BeamSpotCUDA(Data const& data_h, int device, cudaStream_t stream) : device_(device) {
  data_d_ = cms::cuda::make_managed_unique<Data>(stream);
  *data_d_ = data_h;
#ifndef CUDAUVM_DISABLE_ADVISE
  cudaCheck(cudaMemAdvise(data_d_.get(), sizeof(Data), cudaMemAdviseSetReadMostly, device));
#endif
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(data_d_.get(), sizeof(Data), device, stream));
#endif
}
#endif  // CUDAUVM_DISABLE_MANAGED_BEAMSPOT

BeamSpotCUDA::~BeamSpotCUDA() {
#ifndef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
#ifndef CUDAUVM_DISABLE_ADVISE
  // need to make sure a CUDA context is initialized for a thread
  cms::cuda::ScopedSetDevice(0);
  cudaCheck(cudaMemAdvise(data_d_.get(), sizeof(Data), cudaMemAdviseUnsetReadMostly, device_));
#endif
#endif
}
