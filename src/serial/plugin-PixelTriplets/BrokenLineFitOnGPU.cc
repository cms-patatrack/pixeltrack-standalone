#include "BrokenLineFitOnGPU.h"

void HelixFitOnGPU::launchBrokenLineKernelsOnCPU(HitsView const* hv, uint32_t hitsInFit, uint32_t maxNumberOfTuples) {
  assert(tuples_d);

  // the GPU kernels process the n-tuplets in chunks of maxNumberOfConcurrentFits_, up to maxNumberOfTuples
  uint32_t const maxTuples =
      (maxNumberOfTuples + maxNumberOfConcurrentFits_ - 1) / maxNumberOfConcurrentFits_ * maxNumberOfConcurrentFits_;

  // fit triplets
  kernelBLFastFitAndFit<3>(tuples_d, tupleMultiplicity_d, hv, bField_, outputSoa_d, 3, maxTuples);

  // fit quads
  kernelBLFastFitAndFit<4>(tuples_d, tupleMultiplicity_d, hv, bField_, outputSoa_d, 4, maxTuples);

  if (fit5as4_) {
    // fit penta (only first 4)
    kernelBLFastFitAndFit<4>(tuples_d, tupleMultiplicity_d, hv, bField_, outputSoa_d, 5, maxTuples);
  } else {
    // fit penta (all 5)
    kernelBLFastFitAndFit<5>(tuples_d, tupleMultiplicity_d, hv, bField_, outputSoa_d, 5, maxTuples);
  }
}
