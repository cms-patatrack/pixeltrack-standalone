#include "CUDADataFormats/BeamSpotCUDA.h"

#include "CUDACore/cudaCheck.h"
#include "CUDACore/managed_unique_ptr.h"

BeamSpotCUDA::BeamSpotCUDA(Data const& data_h, int device, cudaStream_t stream) {
  data_d_ = cms::cuda::make_managed_unique<Data>(stream);
  *data_d_ = data_h;
#ifndef CUDAUVM_DISABLE_ADVISE
  cudaCheck(cudaMemAdvise(data_d_.get(), sizeof(Data), cudaMemAdviseSetReadMostly, device));
#endif
#ifndef CUDAUVM_DISABLE_PREFETCH
  cudaCheck(cudaMemPrefetchAsync(data_d_.get(), sizeof(Data), device, stream));
#endif
}
