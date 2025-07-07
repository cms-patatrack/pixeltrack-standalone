#include "HelixFitOnGPU.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void HelixFitOnGPU::allocateOnGPU(Tuples const *tuples,
                                    TupleMultiplicity const *tupleMultiplicity,
                                    OutputSoA *helix_fit_results) {
    tuples_d = tuples;
    tupleMultiplicity_d = tupleMultiplicity;
    outputSoa_d = helix_fit_results;

    ALPAKA_ASSERT_ACC(tuples_d);
    ALPAKA_ASSERT_ACC(tupleMultiplicity_d);
    ALPAKA_ASSERT_ACC(outputSoa_d);
  }

  void HelixFitOnGPU::deallocateOnGPU() {}

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
