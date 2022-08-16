#include <memory>

#include "BrokenLineFitOnGPU.h"

void HelixFitOnGPU::launchBrokenLineKernels(HitsView const *hv,
                                            uint32_t hitsInFit,
                                            uint32_t maxNumberOfTuples) {
  assert(tuples_d);

  auto blockSize = 64;
  auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

  //  Fit internals
  auto hitsGPU_ = std::make_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double));
  auto hits_geGPU_ = std::make_unique<float[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float));
  auto fast_fit_resultsGPU_ = std::make_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double));

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    kernelBLFastFit<3><<<numberOfBlocks, blockSize, 0>>>(
        tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 3, offset);
    cudaCheck(cudaGetLastError());

    kernelBLFit<3><<<numberOfBlocks, blockSize, 0>>>(tupleMultiplicity_d,
                                                             bField_,
                                                             outputSoa_d,
                                                             hitsGPU_.get(),
                                                             hits_geGPU_.get(),
                                                             fast_fit_resultsGPU_.get(),
                                                             3,
                                                             offset);
    cudaCheck(cudaGetLastError());

    // fit quads
    kernelBLFastFit<4><<<numberOfBlocks / 4, blockSize, 0>>>(
        tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 4, offset);
    cudaCheck(cudaGetLastError());

    kernelBLFit<4><<<numberOfBlocks / 4, blockSize, 0>>>(tupleMultiplicity_d,
                                                                 bField_,
                                                                 outputSoa_d,
                                                                 hitsGPU_.get(),
                                                                 hits_geGPU_.get(),
                                                                 fast_fit_resultsGPU_.get(),
                                                                 4,
                                                                 offset);
    cudaCheck(cudaGetLastError());

    if (fit5as4_) {
      // fit penta (only first 4)
      kernelBLFastFit<4><<<numberOfBlocks / 4, blockSize, 0>>>(
          tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);
      cudaCheck(cudaGetLastError());

      kernelBLFit<4><<<numberOfBlocks / 4, blockSize, 0>>>(tupleMultiplicity_d,
                                                                   bField_,
                                                                   outputSoa_d,
                                                                   hitsGPU_.get(),
                                                                   hits_geGPU_.get(),
                                                                   fast_fit_resultsGPU_.get(),
                                                                   5,
                                                                   offset);
      cudaCheck(cudaGetLastError());
    } else {
      // fit penta (all 5)
      kernelBLFastFit<5><<<numberOfBlocks / 4, blockSize, 0>>>(
          tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);
      cudaCheck(cudaGetLastError());

      kernelBLFit<5><<<numberOfBlocks / 4, blockSize, 0>>>(tupleMultiplicity_d,
                                                                   bField_,
                                                                   outputSoa_d,
                                                                   hitsGPU_.get(),
                                                                   hits_geGPU_.get(),
                                                                   fast_fit_resultsGPU_.get(),
                                                                   5,
                                                                   offset);
      cudaCheck(cudaGetLastError());
    }

  }  // loop on concurrent fits
}
