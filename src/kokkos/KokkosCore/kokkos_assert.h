// The omission of #include guards is on purpose: it does make sense to #include
// this file multiple times, setting a different value of GPU_DEBUG beforehand.

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifndef GPU_DEBUG
// disable asserts
#ifndef NDEBUG
#define NDEBUG
#endif
#else
// enable asserts
#ifdef NDEBUG
#undef NDEBUG
#endif
#endif
#endif  // __CUDA_ARCH__

#include <cassert>
