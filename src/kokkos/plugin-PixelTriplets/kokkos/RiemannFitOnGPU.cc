#include "RiemannFitOnGPU.h"

namespace KOKKOS_NAMESPACE {
  void HelixFitOnGPU::launchRiemannKernels(HitsView const *hv,
                                           uint32_t nhits,
                                           uint32_t maxNumberOfTuples,
                                           KokkosExecSpace const &execSpace) {
    //  Fit internals
    Kokkos::View<double *, KokkosExecSpace> hitsGPU("hitsGPU",
                                                    maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>));
    Kokkos::View<float *, KokkosExecSpace> hits_geGPU("hits_geGPU",
                                                      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f));
    Kokkos::View<double *, KokkosExecSpace> fast_fit_resultsGPU("fast_fit_resultsGPU",
                                                                maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d));
    Kokkos::View<Rfit::circle_fit *, KokkosExecSpace> circle_fit_resultsGPU("circle_fit_resultsGPU",
                                                                            maxNumberOfConcurrentFits_);

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // triplets
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
          KOKKOS_LAMBDA(size_t i) {
            kernelFastFit<3>(tuples_d, tupleMultiplicity_d, 3, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, offset, i);
          });

      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
          KOKKOS_LAMBDA(size_t i) {
            kernelCircleFit<3>(tupleMultiplicity_d,
                               3,
                               bField_,
                               hitsGPU,
                               hits_geGPU,
                               fast_fit_resultsGPU,
                               circle_fit_resultsGPU,
                               offset,
                               i);
          });
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
          KOKKOS_LAMBDA(size_t i) {
            kernelLineFit<3>(tupleMultiplicity_d,
                             3,
                             bField_,
                             outputSoa_d,
                             hitsGPU,
                             hits_geGPU,
                             fast_fit_resultsGPU,
                             circle_fit_resultsGPU,
                             offset,
                             i);
          });

      // quads
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
          KOKKOS_LAMBDA(size_t i) {
            kernelFastFit<4>(tuples_d, tupleMultiplicity_d, 4, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, offset, i);
          });

      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
          KOKKOS_LAMBDA(size_t i) {
            kernelCircleFit<4>(tupleMultiplicity_d,
                               4,
                               bField_,
                               hitsGPU,
                               hits_geGPU,
                               fast_fit_resultsGPU,
                               circle_fit_resultsGPU,
                               offset,
                               i);
          });
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
          KOKKOS_LAMBDA(size_t i) {
            kernelLineFit<4>(tupleMultiplicity_d,
                             4,
                             bField_,
                             outputSoa_d,
                             hitsGPU,
                             hits_geGPU,
                             fast_fit_resultsGPU,
                             circle_fit_resultsGPU,
                             offset,
                             i);
          });

      if (fit5as4_) {
        // penta
        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
            KOKKOS_LAMBDA(size_t i) {
              kernelFastFit<4>(
                  tuples_d, tupleMultiplicity_d, 5, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, offset, i);
            });

        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
            KOKKOS_LAMBDA(size_t i) {
              kernelCircleFit<4>(tupleMultiplicity_d,
                                 5,
                                 bField_,
                                 hitsGPU,
                                 hits_geGPU,
                                 fast_fit_resultsGPU,
                                 circle_fit_resultsGPU,
                                 offset,
                                 i);
            });
        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
            KOKKOS_LAMBDA(size_t i) {
              kernelLineFit<4>(tupleMultiplicity_d,
                               5,
                               bField_,
                               outputSoa_d,
                               hitsGPU,
                               hits_geGPU,
                               fast_fit_resultsGPU,
                               circle_fit_resultsGPU,
                               offset,
                               i);
            });
      } else {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
            KOKKOS_LAMBDA(size_t i) {
              kernelFastFit<5>(
                  tuples_d, tupleMultiplicity_d, 5, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, offset, i);
            });

        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
            KOKKOS_LAMBDA(size_t i) {
              kernelCircleFit<5>(tupleMultiplicity_d,
                                 5,
                                 bField_,
                                 hitsGPU,
                                 hits_geGPU,
                                 fast_fit_resultsGPU,
                                 circle_fit_resultsGPU,
                                 offset,
                                 i);
            });
        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
            KOKKOS_LAMBDA(size_t i) {
              kernelLineFit<5>(tupleMultiplicity_d,
                               5,
                               bField_,
                               outputSoa_d,
                               hitsGPU,
                               hits_geGPU,
                               fast_fit_resultsGPU,
                               circle_fit_resultsGPU,
                               offset,
                               i);
            });
      }
    }
  }
}  // namespace KOKKOS_NAMESPACE
