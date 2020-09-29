#ifndef Test1_gpuAlgo1_h
#define Test1_gpuAlgo1_h

#include <CL/sycl.hpp>

#include "CUDACore/device_unique_ptr.h"

cms::cuda::device::unique_ptr<float[]> gpuAlgo1(sycl::queue stream);

#endif
