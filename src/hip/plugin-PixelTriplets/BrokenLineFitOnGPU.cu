#include "hip/hip_runtime.h"
#include "BrokenLineFitOnGPU.h"
#include "CUDACore/device_unique_ptr.h"

void HelixFitOnGPU::launchBrokenLineKernels(HitsView const *hv,
                                            uint32_t hitsInFit,
                                            uint32_t maxNumberOfTuples,
                                            hipStream_t stream) {
  assert(tuples_d);

  auto blockSize = 64;
  auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

  //  Fit internals
  auto hitsGPU_ = cms::hip::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double), stream);
  auto hits_geGPU_ = cms::hip::make_device_unique<float[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float), stream);
  auto fast_fit_resultsGPU_ = cms::hip::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double), stream);

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    kernelBLFastFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(
        tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 3, offset);
    cudaCheck(hipGetLastError());

    kernelBLFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(tupleMultiplicity_d,
                                                             bField_,
                                                             outputSoa_d,
                                                             hitsGPU_.get(),
                                                             hits_geGPU_.get(),
                                                             fast_fit_resultsGPU_.get(),
                                                             3,
                                                             offset);
    cudaCheck(hipGetLastError());

    // fit quads
    kernelBLFastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(
        tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 4, offset);
    cudaCheck(hipGetLastError());

    kernelBLFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_d,
                                                                 bField_,
                                                                 outputSoa_d,
                                                                 hitsGPU_.get(),
                                                                 hits_geGPU_.get(),
                                                                 fast_fit_resultsGPU_.get(),
                                                                 4,
                                                                 offset);
    cudaCheck(hipGetLastError());

    if (fit5as4_) {
      // fit penta (only first 4)
      kernelBLFastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(
          tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);
      cudaCheck(hipGetLastError());

      kernelBLFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_d,
                                                                   bField_,
                                                                   outputSoa_d,
                                                                   hitsGPU_.get(),
                                                                   hits_geGPU_.get(),
                                                                   fast_fit_resultsGPU_.get(),
                                                                   5,
                                                                   offset);
      cudaCheck(hipGetLastError());
    } else {
      // fit penta (all 5)
      kernelBLFastFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(
          tuples_d, tupleMultiplicity_d, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), 5, offset);
      cudaCheck(hipGetLastError());

      kernelBLFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_d,
                                                                   bField_,
                                                                   outputSoa_d,
                                                                   hitsGPU_.get(),
                                                                   hits_geGPU_.get(),
                                                                   fast_fit_resultsGPU_.get(),
                                                                   5,
                                                                   offset);
      cudaCheck(hipGetLastError());
    }

  }  // loop on concurrent fits
}
