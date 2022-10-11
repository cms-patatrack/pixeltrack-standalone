#include "BrokenLineFitOnGPU.h"
#include "CUDACore/device_unique_ptr.h"

void HelixFitOnGPU::launchBrokenLineKernels(HitsView const *hv,
                                            uint32_t hitsInFit,
                                            uint32_t maxNumberOfTuples,
                                            cudaStream_t stream) {
  assert(tuples_d);

  //  Fit internals
  auto hitsGPU_ = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double), stream);
  auto hits_geGPU_ = cms::cuda::make_device_unique<float[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float), stream);
  auto fast_fit_resultsGPU_ = cms::cuda::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double), stream);

  cms::cuda::ExecutionConfiguration exec;
  auto blockSize_ff3 = exec.configFromFile("kernelFastFit3");
  auto numberOfBlocks_ff3 = (maxNumberOfConcurrentFits_ + blockSize_ff3 - 1) / blockSize_ff3;

  auto blockSize_blf3 = exec.configFromFile("kernelLineFit3");
  auto numberOfBlocks_blf3 = (maxNumberOfConcurrentFits_ + blockSize_blf3 - 1) / blockSize_blf3;

  auto blockSize_ff4 = exec.configFromFile("kernelFastFit4");
  auto numberOfBlocks_ff4 = (maxNumberOfConcurrentFits_ + blockSize_ff4 - 1) / blockSize_ff4;

  auto blockSize_blf4 = exec.configFromFile("kernelLineFit4");
  auto numberOfBlocks_blf4 = (maxNumberOfConcurrentFits_ + blockSize_blf4 - 1) / blockSize_blf4;

  auto blockSize_ff5 = exec.configFromFile("kernelFastFit5");
  auto numberOfBlocks_ff5 = (maxNumberOfConcurrentFits_ + blockSize_ff5 - 1) / blockSize_ff5;

  auto blockSize_blf5 = exec.configFromFile("kernelLineFit5");
  auto numberOfBlocks_blf5 = (maxNumberOfConcurrentFits_ + blockSize_blf5 - 1) / blockSize_blf5;
  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    kernelBLFastFit<3><<<numberOfBlocks_ff3, blockSize_ff3, 0, stream>>>(
        tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 3, offset);
    cudaCheck(cudaGetLastError());

    kernelBLFit<3><<<numberOfBlocks_blf3, blockSize_blf3, 0, stream>>>(tupleMultiplicity_d,
                                                             bField_,
                                                             outputSoa_d,
                                                             hitsGPU_.get(),
                                                             hits_geGPU_.get(),
                                                             fast_fit_resultsGPU_.get(),
                                                             3,
                                                             offset);
    cudaCheck(cudaGetLastError());

    // fit quads
    kernelBLFastFit<4><<<numberOfBlocks_ff4 / 4, blockSize_ff4, 0, stream>>>(
        tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 4, offset);
    cudaCheck(cudaGetLastError());

    kernelBLFit<4><<<numberOfBlocks_blf4 / 4, blockSize_blf4, 0, stream>>>(tupleMultiplicity_d,
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
      kernelBLFastFit<4><<<numberOfBlocks_ff4 / 4, blockSize_ff4, 0, stream>>>(
          tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);
      cudaCheck(cudaGetLastError());

      kernelBLFit<4><<<numberOfBlocks_blf4 / 4, blockSize_blf4, 0, stream>>>(tupleMultiplicity_d,
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
      kernelBLFastFit<5><<<numberOfBlocks_ff5 / 4, blockSize_ff5, 0, stream>>>(
          tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);
      cudaCheck(cudaGetLastError());

      kernelBLFit<5><<<numberOfBlocks_blf5 / 4, blockSize_blf5, 0, stream>>>(tupleMultiplicity_d,
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
