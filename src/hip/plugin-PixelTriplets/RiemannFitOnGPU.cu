#include "hip/hip_runtime.h"
#include "RiemannFitOnGPU.h"
#include "CUDACore/device_unique_ptr.h"

void HelixFitOnGPU::launchRiemannKernels(HitsView const *hv,
                                         uint32_t nhits,
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
  auto circle_fit_resultsGPU_holder =
      cms::hip::make_device_unique<char[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit), stream);
  Rfit::circle_fit *circle_fit_resultsGPU_ = (Rfit::circle_fit *)(circle_fit_resultsGPU_holder.get());

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // triplets
    kernelFastFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(
        tuples_d, tupleMultiplicity_d, 3, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);
    cudaCheck(hipGetLastError());

    kernelCircleFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(tupleMultiplicity_d,
                                                                 3,
                                                                 bField_,
                                                                 hitsGPU_.get(),
                                                                 hits_geGPU_.get(),
                                                                 fast_fit_resultsGPU_.get(),
                                                                 circle_fit_resultsGPU_,
                                                                 offset);
    cudaCheck(hipGetLastError());

    kernelLineFit<3><<<numberOfBlocks, blockSize, 0, stream>>>(tupleMultiplicity_d,
                                                               3,
                                                               bField_,
                                                               outputSoa_d,
                                                               hitsGPU_.get(),
                                                               hits_geGPU_.get(),
                                                               fast_fit_resultsGPU_.get(),
                                                               circle_fit_resultsGPU_,
                                                               offset);
    cudaCheck(hipGetLastError());

    // quads
    kernelFastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(
        tuples_d, tupleMultiplicity_d, 4, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);
    cudaCheck(hipGetLastError());

    kernelCircleFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_d,
                                                                     4,
                                                                     bField_,
                                                                     hitsGPU_.get(),
                                                                     hits_geGPU_.get(),
                                                                     fast_fit_resultsGPU_.get(),
                                                                     circle_fit_resultsGPU_,
                                                                     offset);
    cudaCheck(hipGetLastError());

    kernelLineFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_d,
                                                                   4,
                                                                   bField_,
                                                                   outputSoa_d,
                                                                   hitsGPU_.get(),
                                                                   hits_geGPU_.get(),
                                                                   fast_fit_resultsGPU_.get(),
                                                                   circle_fit_resultsGPU_,
                                                                   offset);
    cudaCheck(hipGetLastError());

    if (fit5as4_) {
      // penta
      kernelFastFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(
          tuples_d, tupleMultiplicity_d, 5, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);
      cudaCheck(hipGetLastError());

      kernelCircleFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_d,
                                                                       5,
                                                                       bField_,
                                                                       hitsGPU_.get(),
                                                                       hits_geGPU_.get(),
                                                                       fast_fit_resultsGPU_.get(),
                                                                       circle_fit_resultsGPU_,
                                                                       offset);
      cudaCheck(hipGetLastError());

      kernelLineFit<4><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_d,
                                                                     5,
                                                                     bField_,
                                                                     outputSoa_d,
                                                                     hitsGPU_.get(),
                                                                     hits_geGPU_.get(),
                                                                     fast_fit_resultsGPU_.get(),
                                                                     circle_fit_resultsGPU_,
                                                                     offset);
      cudaCheck(hipGetLastError());
    } else {
      // penta all 5
      kernelFastFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(
          tuples_d, tupleMultiplicity_d, 5, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);
      cudaCheck(hipGetLastError());

      kernelCircleFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_d,
                                                                       5,
                                                                       bField_,
                                                                       hitsGPU_.get(),
                                                                       hits_geGPU_.get(),
                                                                       fast_fit_resultsGPU_.get(),
                                                                       circle_fit_resultsGPU_,
                                                                       offset);
      cudaCheck(hipGetLastError());

      kernelLineFit<5><<<numberOfBlocks / 4, blockSize, 0, stream>>>(tupleMultiplicity_d,
                                                                     5,
                                                                     bField_,
                                                                     outputSoa_d,
                                                                     hitsGPU_.get(),
                                                                     hits_geGPU_.get(),
                                                                     fast_fit_resultsGPU_.get(),
                                                                     circle_fit_resultsGPU_,
                                                                     offset);
      cudaCheck(hipGetLastError());
    }
  }
}
