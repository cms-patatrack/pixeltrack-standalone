#include <iostream>

#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "../test_common.h"

using namespace Eigen;

using Matrix5d = Matrix<double, 5, 5>;

template <class C>
KOKKOS_INLINE_FUNCTION void printIt(C* m) {
#ifdef TEST_DEBUG
  printf("\nMatrix %dx%d\n", (int)m->rows(), (int)m->cols());
  for (u_int r = 0; r < m->rows(); ++r) {
    for (u_int c = 0; c < m->cols(); ++c) {
      printf("Matrix(%d,%d) = %f\n", r, c, (*m)(r, c));
    }
  }
#endif
}

KOKKOS_INLINE_FUNCTION void eigenValues(Matrix3d* m, Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType* ret) {
#if TEST_DEBUG
  printf("Matrix(0,0): %f\n", (*m)(0, 0));
  printf("Matrix(1,1): %f\n", (*m)(1, 1));
  printf("Matrix(2,2): %f\n", (*m)(2, 2));
#endif
  SelfAdjointEigenSolver<Matrix3d> es;
  es.computeDirect(*m);
  (*ret) = es.eigenvalues();
  return;
}

KOKKOS_INLINE_FUNCTION void kernel(
    Kokkos::View<Matrix3d, KokkosExecSpace> vm,
    Kokkos::View<Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType, KokkosExecSpace> vret) {
  eigenValues(vm.data(), vret.data());
}

KOKKOS_INLINE_FUNCTION void kernelInverse3x3(Kokkos::View<Matrix3d, KokkosExecSpace> vm,
                                             Kokkos::View<Matrix3d, KokkosExecSpace> vmret) {
  vmret() = vm().inverse();
}

KOKKOS_INLINE_FUNCTION void kernelInverse4x4(Kokkos::View<Matrix4d, KokkosExecSpace> vm,
                                             Kokkos::View<Matrix4d, KokkosExecSpace> vmret) {
  vmret() = vm().inverse();
}

KOKKOS_INLINE_FUNCTION void kernelInverse5x5(Kokkos::View<Matrix5d, KokkosExecSpace> vm,
                                             Kokkos::View<Matrix5d, KokkosExecSpace> vmret) {
  vmret() = vm().inverse();
}

template <typename M1, typename M2, typename M3>
KOKKOS_INLINE_FUNCTION void kernelMultiply(Kokkos::View<M1, KokkosExecSpace> d_j,
                                           Kokkos::View<M2, KokkosExecSpace> d_c,
                                           Kokkos::View<M3, KokkosExecSpace> d_result) {
//  Map<M3> res(result->data());
#if TEST_DEBUG
  printf("*** GPU IN ***\n");
#endif
  printIt(d_j.data());
  printIt(d_c.data());
  //  res.noalias() = (*J) * (*C);
  //  printIt(&res);
  d_result() = d_j() * d_c();
#if TEST_DEBUG
  printf("*** GPU OUT ***\n");
#endif
  return;
}

template <int row1, int col1, int row2, int col2>
void testMultiply() {
  std::cout << "TEST MULTIPLY" << std::endl;
  std::cout << "Product of type " << row1 << "x" << col1 << " * " << row2 << "x" << col2 << std::endl;

  Kokkos::View<Matrix<double, row1, col1>, KokkosExecSpace> d_j("d_j");
  Kokkos::View<Matrix<double, row2, col2>, KokkosExecSpace> d_c("d_c");
  Kokkos::View<Matrix<double, row1, col2>, KokkosExecSpace> d_multiply_result("d_multiply_result");

  auto h_j = Kokkos::create_mirror_view(d_j);
  auto h_c = Kokkos::create_mirror_view(d_c);
  auto h_multiply_result = Kokkos::create_mirror_view(d_multiply_result);

  fillMatrix(h_j());
  fillMatrix(h_c());
  h_multiply_result() = h_j() * h_c();
  auto multiply_result = h_multiply_result();

#if TEST_DEBUG
  std::cout << "Input J:" << std::endl;
  printIt(h_j.data());
  std::cout << "Input C:" << std::endl;
  printIt(h_c.data());
  std::cout << "Output:" << std::endl;
  printIt(&multiply_result);
#endif
  // GPU
  Kokkos::deep_copy(KokkosExecSpace(), d_j, h_j);
  Kokkos::deep_copy(KokkosExecSpace(), d_c, h_c);
  Kokkos::deep_copy(KokkosExecSpace(), d_multiply_result, h_multiply_result);

  auto policy = Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, 1);
  Kokkos::parallel_for(
      "kernelMultiply", policy, KOKKOS_LAMBDA(const int& i) { kernelMultiply(d_j, d_c, d_multiply_result); });
  KokkosExecSpace().fence();

  Kokkos::deep_copy(KokkosExecSpace(), h_multiply_result, d_multiply_result);
  printIt(h_multiply_result.data());
  assert(isEqualFuzzy(multiply_result, h_multiply_result()));
}

void testInverse3x3() {
  std::cout << "TEST INVERSE 3x3" << std::endl;

  Kokkos::View<Matrix3d, KokkosExecSpace> d_m("d_m");
  Kokkos::View<Matrix3d, KokkosExecSpace> d_mret("d_mret");

  auto h_m = Kokkos::create_mirror_view(d_m);
  auto h_mret = Kokkos::create_mirror_view(d_mret);

  fillMatrix(h_m());
  h_m() += h_m().transpose().eval();

  Matrix3d m_inv = h_m().inverse();

#if TEST_DEBUG
  std::cout << "Here is the matrix m:" << std::endl << h_m() << std::endl;
  std::cout << "Its inverse is:" << std::endl << m_inv << std::endl;
#endif
  Kokkos::deep_copy(KokkosExecSpace(), d_m, h_m);

  auto policy = Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, 1);
  Kokkos::parallel_for(
      "kernelInverse3x3", policy, KOKKOS_LAMBDA(const int& i) { kernelInverse3x3(d_m, d_mret); });
  Kokkos::deep_copy(KokkosExecSpace(), h_mret, d_mret);
  KokkosExecSpace().fence();

#if TEST_DEBUG
  std::cout << "Its GPU inverse is:" << std::endl << h_mret() << std::endl;
#endif
  assert(isEqualFuzzy(m_inv, h_mret()));
}

void testInverse4x4() {
  std::cout << "TEST INVERSE 4x4" << std::endl;

  Kokkos::View<Matrix4d, KokkosExecSpace> d_m("d_m");
  Kokkos::View<Matrix4d, KokkosExecSpace> d_mret("d_mret");

  auto h_m = Kokkos::create_mirror_view(d_m);
  auto h_mret = Kokkos::create_mirror_view(d_mret);

  fillMatrix(h_m());
  h_m() += h_m().transpose().eval();

  Matrix4d m_inv = h_m().inverse();

#if TEST_DEBUG
  std::cout << "Here is the matrix m:" << std::endl << h_m() << std::endl;
  std::cout << "Its inverse is:" << std::endl << m_inv << std::endl;
#endif
  Kokkos::deep_copy(KokkosExecSpace(), d_m, h_m);

  auto policy = Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, 1);
  Kokkos::parallel_for(
      "kernelInverse4x4", policy, KOKKOS_LAMBDA(const int& i) { kernelInverse4x4(d_m, d_mret); });
  Kokkos::deep_copy(KokkosExecSpace(), h_mret, d_mret);
  KokkosExecSpace().fence();
#if TEST_DEBUG
  std::cout << "Its GPU inverse is:" << std::endl << h_mret() << std::endl;
#endif
  assert(isEqualFuzzy(m_inv, h_mret()));
}

void testInverse5x5() {
  std::cout << "TEST INVERSE 5x5" << std::endl;

  Kokkos::View<Matrix5d, KokkosExecSpace> d_m("d_m");
  Kokkos::View<Matrix5d, KokkosExecSpace> d_mret("d_mret");

  auto h_m = Kokkos::create_mirror_view(d_m);
  auto h_mret = Kokkos::create_mirror_view(d_mret);

  fillMatrix(h_m());
  h_m() += h_m().transpose().eval();

  Matrix5d m_inv = h_m().inverse();

#if TEST_DEBUG
  std::cout << "Here is the matrix m:" << std::endl << h_m() << std::endl;
  std::cout << "Its inverse is:" << std::endl << m_inv << std::endl;
#endif
  Kokkos::deep_copy(KokkosExecSpace(), d_m, h_m);

  auto policy = Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, 1);
  Kokkos::parallel_for(
      "kernelInverse5x5", policy, KOKKOS_LAMBDA(const int& i) { kernelInverse5x5(d_m, d_mret); });
  Kokkos::deep_copy(KokkosExecSpace(), h_mret, d_mret);
  KokkosExecSpace().fence();
#if TEST_DEBUG
  std::cout << "Its GPU inverse is:" << std::endl << h_mret() << std::endl;
#endif
  assert(isEqualFuzzy(m_inv, h_mret()));
}

void testEigenvalues() {
  std::cout << "TEST EIGENVALUES" << std::endl;

  Kokkos::View<Matrix3d, KokkosExecSpace> d_m("d_m");
  Kokkos::View<Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType, KokkosExecSpace> d_ret("d_ret");

  auto h_m = Kokkos::create_mirror_view(d_m);
  auto h_ret = Kokkos::create_mirror_view(d_ret);

  fillMatrix(h_m());
  h_m() += h_m().transpose().eval();

  Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType* ret =
      new Eigen::SelfAdjointEigenSolver<Matrix3d>::RealVectorType;
  eigenValues(h_m.data(), ret);
#if TEST_DEBUG
  std::cout << "Generated Matrix M 3x3:\n" << h_m() << std::endl;
  std::cout << "The eigenvalues of M are:" << std::endl << (*ret) << std::endl;
  std::cout << "*************************\n\n" << std::endl;
#endif
  Kokkos::deep_copy(KokkosExecSpace(), d_m, h_m);

  auto policy = Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, 1);
  Kokkos::parallel_for(
      "kernel", policy, KOKKOS_LAMBDA(const int& i) { kernel(d_m, d_ret); });
  Kokkos::deep_copy(KokkosExecSpace(), h_m, d_m);
  Kokkos::deep_copy(KokkosExecSpace(), h_ret, d_ret);
  KokkosExecSpace().fence();

#if TEST_DEBUG
  std::cout << "GPU Generated Matrix M 3x3:\n" << (h_m()) << std::endl;
  std::cout << "GPU The eigenvalues of M are:" << std::endl << (h_ret()) << std::endl;
  std::cout << "*************************\n\n" << std::endl;
#endif
  assert(isEqualFuzzy(*ret, h_ret()));
}

int main(int argc, char* argv[]) {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  testEigenvalues();
  testInverse3x3();
  testInverse4x4();
  // disable testInverse5x5 since it will result in runtime error: what():  cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, instance.cuda_stream()) error( cudaErrorLaunchFailure): unspecified launch failure
  // testInverse5x5();

  testMultiply<1, 2, 2, 1>();
  testMultiply<1, 2, 2, 2>();
  testMultiply<1, 2, 2, 3>();
  testMultiply<1, 2, 2, 4>();
  testMultiply<1, 2, 2, 5>();
  testMultiply<2, 1, 1, 2>();
  testMultiply<2, 1, 1, 3>();
  testMultiply<2, 1, 1, 4>();
  testMultiply<2, 1, 1, 5>();
  testMultiply<2, 2, 2, 2>();
  testMultiply<2, 3, 3, 1>();
  testMultiply<2, 3, 3, 2>();
  testMultiply<2, 3, 3, 4>();
  testMultiply<2, 3, 3, 5>();
  testMultiply<3, 2, 2, 3>();
  testMultiply<2, 3, 3, 3>();  // DOES NOT COMPILE W/O PATCHING EIGEN
  testMultiply<3, 3, 3, 3>();
  testMultiply<8, 8, 8, 8>();
  testMultiply<3, 4, 4, 3>();
  testMultiply<2, 4, 4, 2>();
  testMultiply<3, 4, 4, 2>();  // DOES NOT COMPILE W/O PATCHING EIGEN

  return 0;
}
