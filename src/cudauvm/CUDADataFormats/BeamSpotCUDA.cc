#include "CUDADataFormats/BeamSpotCUDA.h"

#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"

BeamSpotCUDA::BeamSpotCUDA(Data const* data_h, cudaStream_t stream) {
  data_d_ = cms::cuda::make_device_unique<Data>(stream);
  cudaCheck(cudaMemcpyAsync(data_d_.get(), data_h, sizeof(Data), cudaMemcpyHostToDevice, stream));
}
