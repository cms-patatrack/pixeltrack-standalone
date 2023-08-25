#ifndef AlpakaCore_math_h
#define AlpakaCore_math_h

#include <cmath>
#if defined(ALPAKA_ACC_SYCL_ENABLED)
#include <sycl/sycl.hpp>
#endif

namespace math {

  template <typename T>
  ALPAKA_FN_HOST_ACC T abs(T const& arg) {
#if (defined(ALPAKA_ACC_SYCL_ENABLED) && defined(__SYCL_DEVICE_ONLY__))
    return sycl::abs(arg);
#elif (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
    return ::abs(arg);
#else
    return std::abs(arg);
#endif
  }

  template <typename T>
  ALPAKA_FN_HOST_ACC T atan(T const& arg) {
#if (defined(ALPAKA_ACC_SYCL_ENABLED) && defined(__SYCL_DEVICE_ONLY__))
    return sycl::atan(arg);
#elif (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
    return ::atan(arg);
#else
    return std::atan(arg);
#endif
  }

  template <typename T1, typename T2>
  ALPAKA_FN_HOST_ACC auto atan2(T1 const& arg1, T2 const& arg2) {
    using TCommon = std::common_type_t<T1, T2>;
#if (defined(ALPAKA_ACC_SYCL_ENABLED) && defined(__SYCL_DEVICE_ONLY__))
    return sycl::atan2(static_cast<TCommon>(arg1), static_cast<TCommon>(arg2));
#elif (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
    return ::atan2(static_cast<TCommon>(arg1), static_cast<TCommon>(arg2));
#else
    return std::atan2(static_cast<TCommon>(arg1), static_cast<TCommon>(arg2));
#endif
  }

  template <typename T>
  ALPAKA_FN_HOST_ACC T cos(T const& arg) {
#if (defined(ALPAKA_ACC_SYCL_ENABLED) && defined(__SYCL_DEVICE_ONLY__))
    return sycl::cos(arg);
#elif (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
    return ::cos(arg);
#else
    return std::cos(arg);
#endif
  }

  template <typename T>
  ALPAKA_FN_HOST_ACC T sin(T const& arg) {
#if (defined(ALPAKA_ACC_SYCL_ENABLED) && defined(__SYCL_DEVICE_ONLY__))
    return sycl::sin(arg);
#elif (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
    return ::sin(arg);
#else
    return std::sin(arg);
#endif
  }

  template <typename T>
  ALPAKA_FN_HOST_ACC T log(T const& arg) {
#if (defined(ALPAKA_ACC_SYCL_ENABLED) && defined(__SYCL_DEVICE_ONLY__))
    return sycl::log(arg);
#elif (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
    return ::log(arg);
#else
    return std::log(arg);
#endif
  }

  template <typename T>
  ALPAKA_FN_HOST_ACC T sqrt(T const& arg) {
#if (defined(ALPAKA_ACC_SYCL_ENABLED) && defined(__SYCL_DEVICE_ONLY__))
    return sycl::sqrt(arg);
#elif (defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)) || \
    (defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__))
    return ::sqrt(arg);
#else
    return std::sqrt(arg);
#endif
  }

}  // namespace math

#endif
