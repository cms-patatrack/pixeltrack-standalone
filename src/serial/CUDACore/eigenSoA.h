#ifndef HeterogeneousCore_CUDAUtilities_interface_eigenSoA_h
#define HeterogeneousCore_CUDAUtilities_interface_eigenSoA_h

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <Eigen/Core>

#include "CUDACore/cudaCompat.h"

namespace eigenSoA {

  constexpr bool isPowerOf2(int32_t v) { return v && !(v & (v - 1)); }

  template <typename T, int S>
  class alignas(128) ScalarSoA {
  public:
    using Scalar = T;

    constexpr Scalar& operator()(int32_t i) { return data_[i]; }
    constexpr const Scalar operator()(int32_t i) const { return data_[i]; }
    constexpr Scalar& operator[](int32_t i) { return data_[i]; }
    constexpr const Scalar operator[](int32_t i) const { return data_[i]; }

    constexpr Scalar* data() { return data_; }
    constexpr Scalar const* data() const { return data_; }

  private:
    Scalar data_[S];
    static_assert(isPowerOf2(S), "SoA stride not a power of 2");
    static_assert(sizeof(data_) % 128 == 0, "SoA size not a multiple of 128");
  };

  template <typename M, int S>
  class alignas(128) MatrixSoA {
  public:
    using Scalar = typename M::Scalar;
    using Map = Eigen::Map<M, 0, Eigen::Stride<M::RowsAtCompileTime * S, S> >;
    using CMap = Eigen::Map<const M, 0, Eigen::Stride<M::RowsAtCompileTime * S, S> >;

    constexpr Map operator()(int32_t i) { return Map(data_ + i); }
    constexpr CMap operator()(int32_t i) const { return CMap(data_ + i); }
    constexpr Map operator[](int32_t i) { return Map(data_ + i); }
    constexpr CMap operator[](int32_t i) const { return CMap(data_ + i); }

  private:
    Scalar data_[S * M::RowsAtCompileTime * M::ColsAtCompileTime];
    static_assert(isPowerOf2(S), "SoA stride not a power of 2");
    static_assert(sizeof(data_) % 128 == 0, "SoA size not a multiple of 128");
  };

}  // namespace eigenSoA

#endif  // HeterogeneousCore_CUDAUtilities_interface_eigenSoA_h
