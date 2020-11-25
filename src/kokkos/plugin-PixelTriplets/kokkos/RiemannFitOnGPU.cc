#include "RiemannFitOnGPU.h"

#include "KokkosCore/hintLightWeight.h"

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

    // avoid capturing this by the lambdas
    auto const bField = bField_;
    auto tuples = tuples_d;
    auto tupleMultiplicity = tupleMultiplicity_d;
    auto outputSoa = outputSoa_d;

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // triplets
      Kokkos::parallel_for(
          "kernelFastFit_3",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
          KOKKOS_LAMBDA(size_t i) {
            kernelFastFit<3>(tuples, tupleMultiplicity, 3, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, offset, i);
          });

      Kokkos::parallel_for(
          "kernelCircleFit_3",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
          KOKKOS_LAMBDA(size_t i) {
            kernelCircleFit<3>(tupleMultiplicity,
                               3,
                               bField,
                               hitsGPU,
                               hits_geGPU,
                               fast_fit_resultsGPU,
                               circle_fit_resultsGPU,
                               offset,
                               i);
          });
      Kokkos::parallel_for(
          "kernelLineFit_3",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
          KOKKOS_LAMBDA(size_t i) {
            kernelLineFit<3>(tupleMultiplicity,
                             3,
                             bField,
                             outputSoa,
                             hitsGPU,
                             hits_geGPU,
                             fast_fit_resultsGPU,
                             circle_fit_resultsGPU,
                             offset,
                             i);
          });

      // quads
      Kokkos::parallel_for(
          "kernelFastFit_4",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
          KOKKOS_LAMBDA(size_t i) {
            kernelFastFit<4>(tuples, tupleMultiplicity, 4, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, offset, i);
          });

      Kokkos::parallel_for(
          "kernelCircleFit_4",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
          KOKKOS_LAMBDA(size_t i) {
            kernelCircleFit<4>(tupleMultiplicity,
                               4,
                               bField,
                               hitsGPU,
                               hits_geGPU,
                               fast_fit_resultsGPU,
                               circle_fit_resultsGPU,
                               offset,
                               i);
          });
      Kokkos::parallel_for(
          "kernelLineFit_4",
          hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
          KOKKOS_LAMBDA(size_t i) {
            kernelLineFit<4>(tupleMultiplicity,
                             4,
                             bField,
                             outputSoa,
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
            "kernelFastFit_4",
            hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
            KOKKOS_LAMBDA(size_t i) {
              kernelFastFit<4>(tuples, tupleMultiplicity, 5, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, offset, i);
            });

        Kokkos::parallel_for(
            "kernelCircleFit_4",
            hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
            KOKKOS_LAMBDA(size_t i) {
              kernelCircleFit<4>(tupleMultiplicity,
                                 5,
                                 bField,
                                 hitsGPU,
                                 hits_geGPU,
                                 fast_fit_resultsGPU,
                                 circle_fit_resultsGPU,
                                 offset,
                                 i);
            });
        Kokkos::parallel_for(
            "kernelLineFit_4",
            hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
            KOKKOS_LAMBDA(size_t i) {
              kernelLineFit<4>(tupleMultiplicity,
                               5,
                               bField,
                               outputSoa,
                               hitsGPU,
                               hits_geGPU,
                               fast_fit_resultsGPU,
                               circle_fit_resultsGPU,
                               offset,
                               i);
            });
      } else {
        Kokkos::parallel_for(
            "kernelFastFit_5",
            hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
            KOKKOS_LAMBDA(size_t i) {
              kernelFastFit<5>(tuples, tupleMultiplicity, 5, hv, hitsGPU, hits_geGPU, fast_fit_resultsGPU, offset, i);
            });

        Kokkos::parallel_for(
            "kernelCircleFit_5",
            hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
            KOKKOS_LAMBDA(size_t i) {
              kernelCircleFit<5>(tupleMultiplicity,
                                 5,
                                 bField,
                                 hitsGPU,
                                 hits_geGPU,
                                 fast_fit_resultsGPU,
                                 circle_fit_resultsGPU,
                                 offset,
                                 i);
            });
        Kokkos::parallel_for(
            "kernelLineFit_5",
            hintLightWeight(Kokkos::RangePolicy<KokkosExecSpace>(execSpace, 0, Rfit::maxNumberOfConcurrentFits())),
            KOKKOS_LAMBDA(size_t i) {
              kernelLineFit<5>(tupleMultiplicity,
                               5,
                               bField,
                               outputSoa,
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
