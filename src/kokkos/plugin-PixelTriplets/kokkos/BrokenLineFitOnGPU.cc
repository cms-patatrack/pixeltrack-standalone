#include "BrokenLineFitOnGPU.h"

namespace KOKKOS_NAMESPACE {
  void HelixFitOnGPU::launchBrokenLineKernels(HitsView const* hv,
                                              uint32_t hitsInFit,
                                              uint32_t maxNumberOfTuples,
                                              KokkosExecSpace const& execSpace) {
    //  Fit internals
    Kokkos::View<double*, KokkosExecSpace> hitsGPU("hitsGPU", maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>));
    Kokkos::View<float*, KokkosExecSpace> hits_geGPU("hits_geGPU",
                                                     maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f));
    Kokkos::View<double*, KokkosExecSpace> fast_fit_resultsGPU("fast_fit_resultsGPU",
                                                               maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d));

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // fit triplets
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
          KOKKOS_LAMBDA(size_t i) {
            kernelBLFastFit<3>(
                tuples_d, tupleMultiplicity_d, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 3, offset, i);
          });

      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
          KOKKOS_LAMBDA(size_t i) {
            kernelBLFit<3>(
                tupleMultiplicity_d, bField_, outputSoa_d, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 3, offset, i);
          });

      // fit quads
      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
          KOKKOS_LAMBDA(size_t i) {
            kernelBLFastFit<4>(
                tuples_d, tupleMultiplicity_d, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 4, offset, i);
          });

      Kokkos::parallel_for(
          Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
          KOKKOS_LAMBDA(size_t i) {
            kernelBLFit<4>(
                tupleMultiplicity_d, bField_, outputSoa_d, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 4, offset, i);
          });

      if (fit5as4_) {
        // fit penta (only first 4)
        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
            KOKKOS_LAMBDA(size_t i) {
              kernelBLFastFit<4>(
                  tuples_d, tupleMultiplicity_d, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 5, offset, i);
            });

        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
            KOKKOS_LAMBDA(size_t i) {
              kernelBLFit<4>(
                  tupleMultiplicity_d, bField_, outputSoa_d, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 5, offset, i);
            });
      } else {
        // fit penta (all 5)
        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
            KOKKOS_LAMBDA(size_t i) {
              kernelBLFastFit<5>(
                  tuples_d, tupleMultiplicity_d, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 5, offset, i);
            });

        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits()),
            KOKKOS_LAMBDA(size_t i) {
              kernelBLFit<5>(
                  tupleMultiplicity_d, bField_, outputSoa_d, hitsGPU, hits_geGPU, fast_fit_resultsGPU, 5, offset, i);
            });
      }
    }  // loop on concurrent fits
  }
}  // namespace KOKKOS_NAMESPACE
