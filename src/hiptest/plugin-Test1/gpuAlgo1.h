#ifndef Test1_gpuAlgo1_h
#define Test1_gpuAlgo1_h

#include <hip/hip_runtime.h>

#include "CUDACore/device_unique_ptr.h"

cms::cuda::device::unique_ptr<float[]> gpuAlgo1(hipStream_t stream);

#endif
