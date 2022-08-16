#include <memory>

#include "RiemannFitOnGPU.h"

#ifndef DISABLE_RFIT

void HelixFitOnGPU::launchRiemannKernels(HitsView const *hv,
                                         uint32_t nhits,
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
  auto circle_fit_resultsGPU_holder =
      std::make_unique<char[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit));
  Rfit::circle_fit *circle_fit_resultsGPU_ = (Rfit::circle_fit *)(circle_fit_resultsGPU_holder.get());

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // triplets
    kernelFastFit<3><<<numberOfBlocks, blockSize, 0>>>(
        tuples_d, tupleMultiplicity_d, 3, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);
    cudaCheck(cudaGetLastError());

    kernelCircleFit<3><<<numberOfBlocks, blockSize, 0>>>(tupleMultiplicity_d,
                                                                 3,
                                                                 bField_,
                                                                 hitsGPU_.get(),
                                                                 hits_geGPU_.get(),
                                                                 fast_fit_resultsGPU_.get(),
                                                                 circle_fit_resultsGPU_,
                                                                 offset);
    cudaCheck(cudaGetLastError());

    kernelLineFit<3><<<numberOfBlocks, blockSize, 0>>>(tupleMultiplicity_d,
                                                               3,
                                                               bField_,
                                                               outputSoa_d,
                                                               hitsGPU_.get(),
                                                               hits_geGPU_.get(),
                                                               fast_fit_resultsGPU_.get(),
                                                               circle_fit_resultsGPU_,
                                                               offset);
    cudaCheck(cudaGetLastError());

    // quads
    kernelFastFit<4><<<numberOfBlocks / 4, blockSize, 0>>>(
        tuples_d, tupleMultiplicity_d, 4, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);
    cudaCheck(cudaGetLastError());

    kernelCircleFit<4><<<numberOfBlocks / 4, blockSize, 0>>>(tupleMultiplicity_d,
                                                                     4,
                                                                     bField_,
                                                                     hitsGPU_.get(),
                                                                     hits_geGPU_.get(),
                                                                     fast_fit_resultsGPU_.get(),
                                                                     circle_fit_resultsGPU_,
                                                                     offset);
    cudaCheck(cudaGetLastError());

    kernelLineFit<4><<<numberOfBlocks / 4, blockSize, 0>>>(tupleMultiplicity_d,
                                                                   4,
                                                                   bField_,
                                                                   outputSoa_d,
                                                                   hitsGPU_.get(),
                                                                   hits_geGPU_.get(),
                                                                   fast_fit_resultsGPU_.get(),
                                                                   circle_fit_resultsGPU_,
                                                                   offset);
    cudaCheck(cudaGetLastError());

    if (fit5as4_) {
      // penta
      kernelFastFit<4><<<numberOfBlocks / 4, blockSize, 0>>>(
          tuples_d, tupleMultiplicity_d, 5, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);
      cudaCheck(cudaGetLastError());

      kernelCircleFit<4><<<numberOfBlocks / 4, blockSize, 0>>>(tupleMultiplicity_d,
                                                                       5,
                                                                       bField_,
                                                                       hitsGPU_.get(),
                                                                       hits_geGPU_.get(),
                                                                       fast_fit_resultsGPU_.get(),
                                                                       circle_fit_resultsGPU_,
                                                                       offset);
      cudaCheck(cudaGetLastError());

      kernelLineFit<4><<<numberOfBlocks / 4, blockSize, 0>>>(tupleMultiplicity_d,
                                                                     5,
                                                                     bField_,
                                                                     outputSoa_d,
                                                                     hitsGPU_.get(),
                                                                     hits_geGPU_.get(),
                                                                     fast_fit_resultsGPU_.get(),
                                                                     circle_fit_resultsGPU_,
                                                                     offset);
      cudaCheck(cudaGetLastError());
    } else {
      // penta all 5
      kernelFastFit<5><<<numberOfBlocks / 4, blockSize, 0>>>(
          tuples_d, tupleMultiplicity_d, 5, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);
      cudaCheck(cudaGetLastError());

      kernelCircleFit<5><<<numberOfBlocks / 4, blockSize, 0>>>(tupleMultiplicity_d,
                                                                       5,
                                                                       bField_,
                                                                       hitsGPU_.get(),
                                                                       hits_geGPU_.get(),
                                                                       fast_fit_resultsGPU_.get(),
                                                                       circle_fit_resultsGPU_,
                                                                       offset);
      cudaCheck(cudaGetLastError());

      kernelLineFit<5><<<numberOfBlocks / 4, blockSize, 0>>>(tupleMultiplicity_d,
                                                                     5,
                                                                     bField_,
                                                                     outputSoa_d,
                                                                     hitsGPU_.get(),
                                                                     hits_geGPU_.get(),
                                                                     fast_fit_resultsGPU_.get(),
                                                                     circle_fit_resultsGPU_,
                                                                     offset);
      cudaCheck(cudaGetLastError());
    }
  }
}

#else
void HelixFitOnGPU::launchRiemannKernels(HitsView const *hv,
                                         uint32_t nhits,
                                         uint32_t maxNumberOfTuples) {}
#endif