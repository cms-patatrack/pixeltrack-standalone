#ifndef Test2_gpuAlgo2_h
#define Test2_gpuAlgo2_h

#include <cuda_runtime.h>

#include "CUDACore/device_unique_ptr.h"

cms::cuda::device::unique_ptr<float[]> gpuAlgo2(cudaStream_t stream);

#endif
