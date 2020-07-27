#ifndef Test2_gpuAlgo2_h
#define Test2_gpuAlgo2_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SYCLCore/device_unique_ptr.h"

cms::sycl::device::unique_ptr<float[]> gpuAlgo2(::sycl::queue stream);

#endif
