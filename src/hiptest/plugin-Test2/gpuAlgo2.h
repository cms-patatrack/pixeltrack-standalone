#ifndef Test2_gpuAlgo2_h
#define Test2_gpuAlgo2_h

#include <hip/hip_runtime.h>

#include "CUDACore/device_unique_ptr.h"

cms::hip::device::unique_ptr<float[]> gpuAlgo2(hipStream_t stream);

#endif
