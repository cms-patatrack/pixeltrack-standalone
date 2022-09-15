#include <algorithm>
#include <execution>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <type_traits>

template <typename T>
struct RS {
  using type = std::uniform_int_distribution<T>;
  static auto ud() { return type(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()); }
  static constexpr T imax = std::numeric_limits<T>::max();
};

template <>
struct RS<float> {
  using T = float;
  using type = std::uniform_real_distribution<float>;
  static auto ud() { return type(-std::numeric_limits<T>::max() / 2, std::numeric_limits<T>::max() / 2); }
  //  static auto ud() { return type(0,std::numeric_limits<T>::max()/2);}
  static constexpr int imax = std::numeric_limits<int>::max();
};

// A templated unsigned integer type with N bytes
template <int N>
struct uintN;

template <>
struct uintN<8> {
  using type = uint8_t;
};

template <>
struct uintN<16> {
  using type = uint16_t;
};

template <>
struct uintN<32> {
  using type = uint32_t;
};

template <>
struct uintN<64> {
  using type = uint64_t;
};

template <int N>
using uintN_t = typename uintN<N>::type;

// A templated unsigned integer type with the same size as T
template <typename T>
using uintT_t = uintN_t<sizeof(T) * 8>;

// Keep only the `N` most significant bytes of `t`, and set the others to zero
template <int N, typename T, typename SFINAE = std::enable_if_t<N <= sizeof(T)>>
void truncate(T& t) {
  const int shift = 8 * (sizeof(T) - N);
  union {
    T t;
    uintT_t<T> u;
  } c;
  c.t = t;
  c.u = c.u >> shift << shift;
  t = c.t;
}

template <typename T, typename LL = long long>
void go() {
  std::mt19937 eng;
  //std::mt19937 eng2;
  auto rgen = RS<T>::ud();

  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;

  constexpr int blocks = 10;
  constexpr int blockSize = 256 * 32;
  constexpr int N = blockSize * blocks;
  auto v{std::make_unique<T[]>(N)};

  constexpr bool sgn = T(-1) < T(0);
  std::cout << "Will sort " << N << (sgn ? " signed" : " unsigned")
            << (std::numeric_limits<T>::is_integer ? " 'ints'" : " 'float'") << " of size " << sizeof(T) << std::endl;

  for (int i = 0; i < 50; ++i) {
    if (i == 49) {
      for (long long j = 0; j < N; j++)
        v[j] = 0;
    } else if (i > 30) {
      for (long long j = 0; j < N; j++)
        v[j] = rgen(eng);
    } else {
      uint64_t imax = (i < 15) ? uint64_t(RS<T>::imax) + 1LL : 255;
      for (uint64_t j = 0; j < N; j++) {
        v[j] = (j % imax);
        if (j % 2 && i % 2)
          v[j] = -v[j];
      }
    }

    std::shuffle(v.get(), v.get() + N, eng);

    delta -= (std::chrono::high_resolution_clock::now() - start);
    constexpr int MaxSize = 256 * 32;
    std::sort(std::execution::par_unseq, v.get(), v.get() + N);
    delta += (std::chrono::high_resolution_clock::now() - start);
    for (int j = 1; j < N; ++j)
      assert(v[j - 1] <= v[j]);
    if (32 == i) {
      std::cout << LL(v[0]) << ' ' << LL(v[1]) << ' ' << LL(v[2]) << std::endl;
      std::cout << LL(v[3]) << ' ' << LL(v[4]) << ' ' << LL(v[blockSize - 1000]) << std::endl;
      std::cout << LL(v[blockSize / 2 - 1]) << ' ' << LL(v[blockSize / 2]) << ' ' << LL(v[blockSize / 2 + 1])
                << std::endl;
    }
  }  // 50 times
  std::cout << "cuda computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count() / 50.
            << " ms" << std::endl;
}

int main() {
  go<int8_t>();
  go<int16_t>();
  go<int32_t>();
  go<int64_t>();
  go<float, double>();
  go<float, double>();

  // go<uint64_t>();

  return 0;
}
