#include <Eigen/Dense>

#include "CUDACore/eigenSoA.h"

template <int32_t S>
struct MySoA {
  // we can find a way to avoid this copy/paste???
  static constexpr int32_t stride() { return S; }

  eigenSoA::ScalarSoA<float, S> a;
  eigenSoA::ScalarSoA<float, S> b;
};

using V = MySoA<128>;

 void testBasicSoA(float* p) {
  using namespace eigenSoA;

  assert(!isPowerOf2(0));
  assert(isPowerOf2(1));
  assert(isPowerOf2(1024));
  assert(!isPowerOf2(1026));

  using M3 = Eigen::Matrix<float, 3, 3>;

   eigenSoA::MatrixSoA<M3, 64> m;

  int first = 0;
  if (0 == first)
    printf("before %f\n", p[0]);

  // a silly game...
  int n = 64;
  for (int i = first; i < n; i += 1) {
    m[i].setZero();
    m[i](0, 0) = p[i];
    m[i](1, 1) = p[i + 64];
    m[i](2, 2) = p[i + 64 * 2];
  }
    // not needed

  for (int i = first; i < n; i += 1)
    m[i] = m[i].inverse().eval();
  

  for (int i = first; i < n; i += 1) {
    p[i] = m[63 - i](0, 0);
    p[i + 64] = m[63 - i](1, 1);
    p[i + 64 * 2] = m[63 - i](2, 2);
  }

  if (0 == first)
    printf("after %f\n", p[0]);
}

#include <cassert>
#include <iostream>
#include <memory>
#include <random>

int main() {
  float p[1024];

  std::uniform_real_distribution<float> rgen(0.01, 0.99);
  std::mt19937 eng;

  for (auto& r : p)
    r = rgen(eng);
  for (int i = 0, n = 64 * 3; i < n; ++i)
    assert(p[i] > 0 && p[i] < 1.);

  std::cout << p[0] << std::endl;
  testBasicSoA(p);

  std::cout << p[0] << std::endl;

  for (int i = 0, n = 64 * 3; i < n; ++i)
    assert(p[i] > 1.);

  std::cout << "END" << std::endl;
  return 0;
}
