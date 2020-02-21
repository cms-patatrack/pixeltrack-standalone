#ifndef Test1_gpuAlgo1_h
#define Test1_gpuAlgo1_h

#include <cuda_runtime.h>

#include "CUDACore/device_unique_ptr.h"

cms::cuda::device::unique_ptr<float[]> gpuAlgo1(cudaStream_t stream);

#endif
