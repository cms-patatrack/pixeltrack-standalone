#include <iostream>

#include "KokkosCore/kokkosConfigCommon.h"
#include "KokkosCore/kokkosConfig.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#ifdef USE_BL
#include "plugin-PixelTriplets/kokkos/BrokenLine.h"
#else
#include "plugin-PixelTriplets/kokkos/RiemannFit.h"
#endif

#include "../test_common.h"

using namespace Eigen;

namespace KOKKOS_NAMESPACE {
  namespace Rfit {
    constexpr uint32_t maxNumberOfTracks() { return 5 * 1024; }
    constexpr uint32_t stride() { return maxNumberOfTracks(); }
    // hits
    template <int N>
    using Matrix3xNd = Eigen::Matrix<double, 3, N>;
    template <int N>
    using Map3xNd = Eigen::Map<Matrix3xNd<N>, 0, Eigen::Stride<3 * stride(), stride()>>;
    // errors
    template <int N>
    using Matrix6xNf = Eigen::Matrix<float, 6, N>;
    template <int N>
    using Map6xNf = Eigen::Map<Matrix6xNf<N>, 0, Eigen::Stride<6 * stride(), stride()>>;
    // fast fit
    using Map4d = Eigen::Map<Vector4d, 0, Eigen::InnerStride<stride()>>;

  }  // namespace Rfit

  template <int N>
  KOKKOS_INLINE_FUNCTION void kernelPrintSizes(Kokkos::View<double*, KokkosExecSpace> vhits,
                                               Kokkos::View<float*, KokkosExecSpace> vhits_ge,
                                               const int& i) {
    double* __restrict__ phits = vhits.data();
    float* __restrict__ phits_ge = vhits_ge.data();

    Rfit::Map3xNd<N> hits(phits + i, 3, 4);
    Rfit::Map6xNf<N> hits_ge(phits_ge + i, 6, 4);
    if (i != 0)
      return;
    printf("GPU sizes %lu %lu %lu %lu %lu\n",
           sizeof(hits[i]),
           sizeof(hits_ge[i]),
           sizeof(Vector4d),
           sizeof(Rfit::line_fit),
           sizeof(Rfit::circle_fit));
  }  // namespace Rfit
}  // namespace KOKKOS_NAMESPACE

using namespace KOKKOS_NAMESPACE;

template <int N>
KOKKOS_INLINE_FUNCTION void kernelFastFit(Kokkos::View<double*, KokkosExecSpace> vhits,
                                          Kokkos::View<double*, KokkosExecSpace> vresults,
                                          const int& i) {
  double* __restrict__ phits = vhits.data();
  double* __restrict__ presults = vresults.data();

  Rfit::Map3xNd<N> hits(phits + i, 3, N);
  Rfit::Map4d result(presults + i, 4);
#ifdef USE_BL
  BrokenLine::BL_Fast_fit(hits, result);
#else
  Rfit::Fast_fit(hits, result);
#endif
}

#ifdef USE_BL

template <int N>
KOKKOS_INLINE_FUNCTION void kernelBrokenLineFit(Kokkos::View<double*, KokkosExecSpace> vhits,
                                                Kokkos::View<float*, KokkosExecSpace> vhits_ge,
                                                Kokkos::View<double*, KokkosExecSpace> vfast_fit_input,
                                                double B,
                                                Kokkos::View<Rfit::circle_fit*, KokkosExecSpace> vcircle_fit,
                                                Kokkos::View<Rfit::line_fit*, KokkosExecSpace> vline_fit,
                                                const int& i) {
  double* __restrict__ phits = vhits.data();
  float* __restrict__ phits_ge = vhits_ge.data();
  double* __restrict__ pfast_fit_input = vfast_fit_input.data();
  Rfit::circle_fit* __restrict__ circle_fit = vcircle_fit.data();
  Rfit::line_fit* __restrict__ line_fit = vline_fit.data();

  Rfit::Map3xNd<N> hits(phits + i, 3, N);
  Rfit::Map4d fast_fit_input(pfast_fit_input + i, 4);
  Rfit::Map6xNf<N> hits_ge(phits_ge + i, 6, N);

  BrokenLine::PreparedBrokenLineData<N> data;
  Rfit::Matrix3d Jacob;

  auto& line_fit_results = line_fit[i];
  auto& circle_fit_results = circle_fit[i];

  BrokenLine::prepareBrokenLineData(hits, fast_fit_input, B, data);
  BrokenLine::BL_Line_fit(hits_ge, fast_fit_input, B, data, line_fit_results);
  BrokenLine::BL_Circle_fit(hits, hits_ge, fast_fit_input, B, data, circle_fit_results);
  Jacob << 1., 0, 0, 0, 1., 0, 0, 0,
      -B / std::copysign(Rfit::sqr(circle_fit_results.par(2)), circle_fit_results.par(2));
  circle_fit_results.par(2) = B / std::abs(circle_fit_results.par(2));
  circle_fit_results.cov = Jacob * circle_fit_results.cov * Jacob.transpose();

#ifdef TEST_DEBUG
  if (0 == i) {
    printf("Circle param %f,%f,%f\n", circle_fit[i].par(0), circle_fit[i].par(1), circle_fit[i].par(2));
  }
#endif
}

#else

template <int N>
KOKKOS_INLINE_FUNCTION void kernelCircleFit(Kokkos::View<double*, KokkosExecSpace> vhits,
                                            Kokkos::View<float*, KokkosExecSpace> vhits_ge,
                                            Kokkos::View<double*, KokkosExecSpace> vfast_fit_input,
                                            double B,
                                            Kokkos::View<Rfit::circle_fit*, KokkosExecSpace> vcircle_fit,
                                            const int& i) {
  double* __restrict__ phits = vhits.data();
  float* __restrict__ phits_ge = vhits_ge.data();
  double* __restrict__ pfast_fit_input = vfast_fit_input.data();
  Rfit::circle_fit* __restrict__ pcircle_fit = vcircle_fit.data();

  Rfit::Map3xNd<N> hits(phits + i, 3, N);
  Rfit::Map4d fast_fit_input(pfast_fit_input + i, 4);
  Rfit::Map6xNf<N> hits_ge(phits_ge + i, 6, N);

  constexpr auto n = N;

  Rfit::VectorNd<N> rad = (hits.block(0, 0, 2, n).colwise().norm());
  Rfit::Matrix2Nd<N> hits_cov = MatrixXd::Zero(2 * n, 2 * n);
  Rfit::loadCovariance2D(hits_ge, hits_cov);

#ifdef TEST_DEBUG
  if (0 == i) {
    printf("hits %f, %f\n", hits.block(0, 0, 2, n)(0, 0), hits.block(0, 0, 2, n)(0, 1));
    printf("hits %f, %f\n", hits.block(0, 0, 2, n)(1, 0), hits.block(0, 0, 2, n)(1, 1));
    printf("fast_fit_input(0): %f\n", fast_fit_input(0));
    printf("fast_fit_input(1): %f\n", fast_fit_input(1));
    printf("fast_fit_input(2): %f\n", fast_fit_input(2));
    printf("fast_fit_input(3): %f\n", fast_fit_input(3));
    printf("rad(0,0): %f\n", rad(0, 0));
    printf("rad(1,1): %f\n", rad(1, 1));
    printf("rad(2,2): %f\n", rad(2, 2));
    printf("hits_cov(0,0): %f\n", (*hits_cov)(0, 0));
    printf("hits_cov(1,1): %f\n", (*hits_cov)(1, 1));
    printf("hits_cov(2,2): %f\n", (*hits_cov)(2, 2));
    printf("hits_cov(11,11): %f\n", (*hits_cov)(11, 11));
    printf("B: %f\n", B);
  }
#endif
  pcircle_fit[i] = Rfit::Circle_fit(hits.block(0, 0, 2, n), hits_cov, fast_fit_input, rad, B, true);
#ifdef TEST_DEBUG
  if (0 == i) {
    printf("Circle param %f,%f,%f\n", pcircle_fit[i].par(0), pcircle_fit[i].par(1), pcircle_fit[i].par(2));
  }
#endif
}

template <int N>
KOKKOS_INLINE_FUNCTION void kernelLineFit(Kokkos::View<double*, KokkosExecSpace> vhits,
                                          Kokkos::View<float*, KokkosExecSpace> vhits_ge,
                                          double B,
                                          Kokkos::View<Rfit::circle_fit*, KokkosExecSpace> vcircle_fit,
                                          Kokkos::View<double*, KokkosExecSpace> vfast_fit_input,
                                          Kokkos::View<Rfit::line_fit*, KokkosExecSpace> vline_fit,
                                          const int& i) {
  double* __restrict__ phits = vhits.data();
  float* __restrict__ phits_ge = vhits_ge.data();
  Rfit::circle_fit* __restrict__ circle_fit = vcircle_fit.data();
  double* __restrict__ pfast_fit_input = vfast_fit_input.data();
  Rfit::line_fit* __restrict__ line_fit = vline_fit.data();

  Rfit::Map3xNd<N> hits(phits + i, 3, N);
  Rfit::Map4d fast_fit_input(pfast_fit_input + i, 4);
  Rfit::Map6xNf<N> hits_ge(phits_ge + i, 6, N);
  line_fit[i] = Rfit::Line_fit(hits, hits_ge, circle_fit[i], fast_fit_input, B, true);
}
#endif

template <typename M3xN, typename M6xN>
KOKKOS_INLINE_FUNCTION void fillHitsAndHitsCov(M3xN& hits, M6xN& hits_ge) {
  constexpr uint32_t N = M3xN::ColsAtCompileTime;

  if (N == 5) {
    hits << 2.934787, 6.314229, 8.936963, 10.360559, 12.856387, 0.773211, 1.816356, 2.765734, 3.330824, 4.422212,
        -10.980247, -23.162731, -32.759060, -38.061260, -47.518867;
    hits_ge.col(0) << 1.424715e-07, -4.996975e-07, 1.752614e-06, 3.660689e-11, 1.644638e-09, 7.346080e-05;
    hits_ge.col(1) << 6.899177e-08, -1.873414e-07, 5.087101e-07, -2.078806e-10, -2.210498e-11, 4.346079e-06;
    hits_ge.col(2) << 1.406273e-06, 4.042467e-07, 6.391180e-07, -3.141497e-07, 6.513821e-08, 1.163863e-07;
    hits_ge.col(3) << 1.176358e-06, 2.154100e-07, 5.072816e-07, -8.161219e-08, 1.437878e-07, 5.951832e-08;
    hits_ge.col(4) << 2.852843e-05, 7.956492e-06, 3.117701e-06, -1.060541e-06, 8.777413e-09, 1.426417e-07;
    return;
  }

  if (N > 3)
    hits << 1.98645, 4.72598, 7.65632, 11.3151, 2.18002, 4.88864, 7.75845, 11.3134, 2.46338, 6.99838, 11.808, 17.793;
  else
    hits << 1.98645, 4.72598, 7.65632, 2.18002, 4.88864, 7.75845, 2.46338, 6.99838, 11.808;

  hits_ge.col(0)[0] = 7.14652e-06;
  hits_ge.col(1)[0] = 2.15789e-06;
  hits_ge.col(2)[0] = 1.63328e-06;
  if (N > 3)
    hits_ge.col(3)[0] = 6.27919e-06;
  hits_ge.col(0)[2] = 6.10348e-06;
  hits_ge.col(1)[2] = 2.08211e-06;
  hits_ge.col(2)[2] = 1.61672e-06;
  if (N > 3)
    hits_ge.col(3)[2] = 6.28081e-06;
  hits_ge.col(0)[5] = 5.184e-05;
  hits_ge.col(1)[5] = 1.444e-05;
  hits_ge.col(2)[5] = 6.25e-06;
  if (N > 3)
    hits_ge.col(3)[5] = 3.136e-05;
  hits_ge.col(0)[1] = -5.60077e-06;
  hits_ge.col(1)[1] = -1.11936e-06;
  hits_ge.col(2)[1] = -6.24945e-07;
  if (N > 3)
    hits_ge.col(3)[1] = -5.28e-06;
}

template <int N>
KOKKOS_INLINE_FUNCTION void kernelFillHitsAndHitsCov(Kokkos::View<double*, KokkosExecSpace> vhits,
                                                     Kokkos::View<float*, KokkosExecSpace> vhits_ge,
                                                     const int& i) {
  double* __restrict__ phits = vhits.data();
  float* __restrict__ phits_ge = vhits_ge.data();

  Rfit::Map3xNd<N> hits(phits + i, 3, N);
  Rfit::Map6xNf<N> hits_ge(phits_ge + i, 6, N);
  hits_ge = MatrixXf::Zero(6, N);
  fillHitsAndHitsCov(hits, hits_ge);
}

template <int N>
void testFit() {
  constexpr double B = 0.0113921;

  Kokkos::View<double*, KokkosExecSpace> d_hits("d_hits", Rfit::maxNumberOfTracks() * sizeof(Rfit::Matrix3xNd<N>));
  Kokkos::View<float*, KokkosExecSpace> d_hits_ge("d_hits_ge", Rfit::maxNumberOfTracks() * sizeof(Rfit::Matrix6xNf<N>));
  Kokkos::View<double*, KokkosExecSpace> d_fast_fit_results("d_fast_fit_results",
                                                            Rfit::maxNumberOfTracks() * sizeof(Vector4d));
  Kokkos::View<Rfit::line_fit*, KokkosExecSpace> d_line_fit_results("d_line_fit_results",
                                                                    Rfit::maxNumberOfTracks() * sizeof(Rfit::line_fit));
  Kokkos::View<Rfit::circle_fit*, KokkosExecSpace> d_circle_fit_results(
      "d_circle_fit_results", Rfit::maxNumberOfTracks() * sizeof(Rfit::circle_fit));

  Rfit::Matrix3xNd<N> hits;
  Rfit::Matrix6xNf<N> hits_ge = MatrixXf::Zero(6, N);

  double* fast_fit_resultsGPUret = new double[Rfit::maxNumberOfTracks() * sizeof(Vector4d)];
  Rfit::circle_fit* circle_fit_resultsGPUret = new Rfit::circle_fit();
  Rfit::line_fit* line_fit_resultsGPUret = new Rfit::line_fit();

  fillHitsAndHitsCov(hits, hits_ge);

  std::cout << "sizes " << N << ' ' << sizeof(hits) << ' ' << sizeof(hits_ge) << ' ' << sizeof(Vector4d) << ' '
            << sizeof(Rfit::line_fit) << ' ' << sizeof(Rfit::circle_fit) << std::endl;

  std::cout << "Generated hits:\n" << hits << std::endl;
  std::cout << "Generated cov:\n" << hits_ge << std::endl;

  // FAST_FIT_CPU
  Vector4d fast_fit_results;
#ifdef USE_BL
  BrokenLine::BL_Fast_fit(hits, fast_fit_results);
#else
  Rfit::Fast_fit(hits, fast_fit_results);
#endif
  std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]):\n" << fast_fit_results << std::endl;

  // cudaMemset d_fast_fit_results & d_line_fit_results to 0
  Kokkos::deep_copy(KokkosExecSpace(), d_fast_fit_results, 0);
  // Kokkos::deep_copy(KokkosExecSpace(), d_line_fit_results, 0) will result in compilation error:
  // no instance of overloaded function "Kokkos::deep_copy" matches the argument list. argument
  // types are: (KokkosExecSpace, Kokkos::View<kokkos_cuda::Rfit::line_fit *, KokkosExecSpace>, int).
  // Use for loop instead
  Kokkos::parallel_for(
      "init_line_fit_res",
      Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, Rfit::maxNumberOfTracks()),
      KOKKOS_LAMBDA(const int& i) {
        d_line_fit_results(i).par = Vector2d::Zero();
        d_line_fit_results(i).cov = Matrix2d::Zero();
        d_line_fit_results(i).chi2 = 0.;
      });

  // for timing purposes we fit 4096 tracks
  constexpr uint32_t Ntracks = 4096;

  auto policy = Kokkos::RangePolicy<KokkosExecSpace>(KokkosExecSpace(), 0, Ntracks);

  Kokkos::parallel_for(
      "kernelPrintSizes", policy, KOKKOS_LAMBDA(const int& i) { kernelPrintSizes<N>(d_hits, d_hits_ge, i); });
  Kokkos::parallel_for(
      "kernelFillHitsAndHitsCov", policy, KOKKOS_LAMBDA(const int& i) {
        kernelFillHitsAndHitsCov<N>(d_hits, d_hits_ge, i);
      });

  // FAST_FIT GPU
  Kokkos::parallel_for(
      "kernelFastFit", policy, KOKKOS_LAMBDA(const int& i) { kernelFastFit<N>(d_hits, d_fast_fit_results, i); });
  KokkosExecSpace().fence();

  auto h_fast_fit_results = Kokkos::create_mirror_view(d_fast_fit_results);
  Kokkos::deep_copy(KokkosExecSpace(), h_fast_fit_results, d_fast_fit_results);
  KokkosExecSpace().fence();

  auto* presults = h_fast_fit_results.data();
  Rfit::Map4d fast_fit(presults + 10, 4);
  std::cout << "Fitted values (FastFit, [X0, Y0, R, tan(theta)]): GPU\n" << fast_fit << std::endl;
  assert(isEqualFuzzy(fast_fit_results, fast_fit));

#ifdef USE_BL
  // CIRCLE AND LINE FIT CPU
  BrokenLine::PreparedBrokenLineData<N> data;
  BrokenLine::karimaki_circle_fit circle_fit_results;
  Rfit::line_fit line_fit_results;
  Rfit::Matrix3d Jacob;
  BrokenLine::prepareBrokenLineData(hits, fast_fit_results, B, data);
  BrokenLine::BL_Line_fit(hits_ge, fast_fit_results, B, data, line_fit_results);
  BrokenLine::BL_Circle_fit(hits, hits_ge, fast_fit_results, B, data, circle_fit_results);
  Jacob << 1., 0, 0, 0, 1., 0, 0, 0,
      -B / std::copysign(Rfit::sqr(circle_fit_results.par(2)), circle_fit_results.par(2));
  circle_fit_results.par(2) = B / std::abs(circle_fit_results.par(2));
  circle_fit_results.cov = Jacob * circle_fit_results.cov * Jacob.transpose();

  // fit on device
  Kokkos::parallel_for(
      "kernelBrokenLineFit", policy, KOKKOS_LAMBDA(const int& i) {
        kernelBrokenLineFit<N>(d_hits, d_hits_ge, d_fast_fit_results, B, d_circle_fit_results, d_line_fit_results, i);
      });
  KokkosExecSpace().fence();

#else
  // CIRCLE_FIT CPU
  Rfit::VectorNd<N> rad = (hits.block(0, 0, 2, N).colwise().norm());

  Rfit::Matrix2Nd<N> hits_cov = Rfit::Matrix2Nd<N>::Zero();
  Rfit::loadCovariance2D(hits_ge, hits_cov);
  Rfit::circle_fit circle_fit_results =
      Rfit::Circle_fit(hits.block(0, 0, 2, N), hits_cov, fast_fit_results, rad, B, true);

  // CIRCLE_FIT GPU
  Kokkos::parallel_for(
      "kernelCircleFit", policy, KOKKOS_LAMBDA(const int& i) {
        kernelCircleFit<N>(d_hits, d_hits_ge, d_fast_fit_results, B, d_circle_fit_results, i);
      });
  KokkosExecSpace().fence();

  // LINE_FIT CPU
  Rfit::line_fit line_fit_results = Rfit::Line_fit(hits, hits_ge, circle_fit_results, fast_fit_results, B, true);

  Kokkos::parallel_for(
      "kernelLineFit", policy, KOKKOS_LAMBDA(const int& i) {
        kernelLineFit<N>(d_hits, d_hits_ge, B, d_circle_fit_results, d_fast_fit_results, d_line_fit_results, i);
      });
  KokkosExecSpace().fence();

#endif

  std::cout << "Fitted values (CircleFit):\n" << circle_fit_results.par << std::endl;

  auto h_circle_fit_results = Kokkos::create_mirror_view(d_circle_fit_results);
  Kokkos::deep_copy(KokkosExecSpace(), h_circle_fit_results, d_circle_fit_results);
  auto* p_circle_fit_res = h_circle_fit_results.data();

  std::cout << "Fitted values (CircleFit) GPU:\n" << p_circle_fit_res->par << std::endl;
  assert(isEqualFuzzy(circle_fit_results.par, p_circle_fit_res->par));

  std::cout << "Fitted values (LineFit):\n" << line_fit_results.par << std::endl;

  // LINE_FIT GPU
  auto h_line_fit_results = Kokkos::create_mirror_view(d_line_fit_results);
  Kokkos::deep_copy(KokkosExecSpace(), h_line_fit_results, d_line_fit_results);
  auto* p_line_fit_res = h_line_fit_results.data();
  std::cout << "Fitted values (LineFit) GPU:\n" << p_line_fit_res->par << std::endl;
  assert(isEqualFuzzy(line_fit_results.par, p_line_fit_res->par, N == 5 ? 1e-4 : 1e-6));  // requires fma on CPU

  std::cout << "Fitted cov (CircleFit) CPU:\n" << circle_fit_results.cov << std::endl;
  std::cout << "Fitted cov (LineFit): CPU\n" << line_fit_results.cov << std::endl;
  std::cout << "Fitted cov (CircleFit) GPU:\n" << p_circle_fit_res->cov << std::endl;
  std::cout << "Fitted cov (LineFit): GPU\n" << p_line_fit_res->cov << std::endl;
}

int main(int argc, char* argv[]) {
  kokkos_common::InitializeScopeGuard kokkosGuard({KokkosBackend<KokkosExecSpace>::value});
  testFit<4>();
  testFit<3>();
  testFit<5>();

  std::cout << "TEST FIT, NO ERRORS" << std::endl;

  return 0;
}
