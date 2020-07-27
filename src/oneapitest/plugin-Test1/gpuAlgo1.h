#ifndef Test1_gpuAlgo1_h
#define Test1_gpuAlgo1_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SYCLCore/device_unique_ptr.h"

cms::sycl::device::unique_ptr<float[]> gpuAlgo1(::sycl::queue stream);

#endif
