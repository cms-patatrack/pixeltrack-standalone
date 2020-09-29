#ifndef Test2_gpuAlgo2_h
#define Test2_gpuAlgo2_h

#include <CL/sycl.hpp>

#include "CUDACore/device_unique_ptr.h"

cms::cuda::device::unique_ptr<float[]> gpuAlgo2(sycl::queue stream);

#endif
