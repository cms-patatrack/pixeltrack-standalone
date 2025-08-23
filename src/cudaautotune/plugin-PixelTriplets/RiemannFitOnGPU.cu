#include "RiemannFitOnGPU.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/ExecutionConfiguration.h"

void HelixFitOnGPU::launchRiemannKernels(HitsView const *hv,
                                         uint32_t nhits,
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
  auto circle_fit_resultsGPU_holder =
      cms::cuda::make_device_unique<char[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit), stream);
  Rfit::circle_fit *circle_fit_resultsGPU_ = (Rfit::circle_fit *)(circle_fit_resultsGPU_holder.get());

  cms::cuda::ExecutionConfiguration exec;
  auto blockSize_ff3 = exec.configFromFile("kernelFastFit3");
  auto numberOfBlocks_ff3 = (maxNumberOfConcurrentFits_ + blockSize_ff3 - 1) / blockSize_ff3;

  auto blockSize_clf3 = exec.configFromFile("kernelCircleFit3");
  auto numberOfBlocks_clf3 = (maxNumberOfConcurrentFits_ + blockSize_clf3 - 1) / blockSize_clf3;

  auto blockSize_blf3 = exec.configFromFile("kernelLineFit3");
  auto numberOfBlocks_blf3 = (maxNumberOfConcurrentFits_ + blockSize_blf3 - 1) / blockSize_blf3;

  auto blockSize_ff4 = exec.configFromFile("kernelFastFit4");
  auto numberOfBlocks_ff4 = (maxNumberOfConcurrentFits_ + blockSize_ff4 - 1) / blockSize_ff4;

  auto blockSize_clf4 = exec.configFromFile("kernelCircleFit4");
  auto numberOfBlocks_clf4 = (maxNumberOfConcurrentFits_ + blockSize_clf4 - 1) / blockSize_clf4;

  auto blockSize_blf4 = exec.configFromFile("kernelLineFit4");
  auto numberOfBlocks_blf4 = (maxNumberOfConcurrentFits_ + blockSize_blf4 - 1) / blockSize_blf4;

  auto blockSize_ff5 = exec.configFromFile("kernelFastFit5");
  auto numberOfBlocks_ff5 = (maxNumberOfConcurrentFits_ + blockSize_ff5 - 1) / blockSize_ff5;

  auto blockSize_clf5 = exec.configFromFile("kernelCircleFit5");
  auto numberOfBlocks_clf5 = (maxNumberOfConcurrentFits_ + blockSize_clf5 - 1) / blockSize_clf5;

  auto blockSize_blf5 = exec.configFromFile("kernelLineFit5");
  auto numberOfBlocks_blf5 = (maxNumberOfConcurrentFits_ + blockSize_blf5 - 1) / blockSize_blf5;

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // triplets
    kernelFastFit<3><<<numberOfBlocks_ff3, blockSize_ff3, 0, stream>>>(
        tuples_d, tupleMultiplicity_d, 3, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);
    cudaCheck(cudaGetLastError());

    kernelCircleFit<3><<<numberOfBlocks_clf3, blockSize_clf3, 0, stream>>>(tupleMultiplicity_d,
                                                                 3,
                                                                 bField_,
                                                                 hitsGPU_.get(),
                                                                 hits_geGPU_.get(),
                                                                 fast_fit_resultsGPU_.get(),
                                                                 circle_fit_resultsGPU_,
                                                                 offset);
    cudaCheck(cudaGetLastError());

    kernelLineFit<3><<<numberOfBlocks_blf3, blockSize_blf3, 0, stream>>>(tupleMultiplicity_d,
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
    kernelFastFit<4><<<numberOfBlocks_ff4 / 4, blockSize_ff4, 0, stream>>>(
        tuples_d, tupleMultiplicity_d, 4, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);
    cudaCheck(cudaGetLastError());

    kernelCircleFit<4><<<numberOfBlocks_clf4 / 4, blockSize_clf4, 0, stream>>>(tupleMultiplicity_d,
                                                                     4,
                                                                     bField_,
                                                                     hitsGPU_.get(),
                                                                     hits_geGPU_.get(),
                                                                     fast_fit_resultsGPU_.get(),
                                                                     circle_fit_resultsGPU_,
                                                                     offset);
    cudaCheck(cudaGetLastError());

    kernelLineFit<4><<<numberOfBlocks_blf4 / 4, blockSize_blf4, 0, stream>>>(tupleMultiplicity_d,
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
      kernelFastFit<4><<<numberOfBlocks_ff4 / 4, blockSize_ff4, 0, stream>>>(
          tuples_d, tupleMultiplicity_d, 5, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);
      cudaCheck(cudaGetLastError());

      kernelCircleFit<4><<<numberOfBlocks_clf4 / 4, blockSize_clf4, 0, stream>>>(tupleMultiplicity_d,
                                                                       5,
                                                                       bField_,
                                                                       hitsGPU_.get(),
                                                                       hits_geGPU_.get(),
                                                                       fast_fit_resultsGPU_.get(),
                                                                       circle_fit_resultsGPU_,
                                                                       offset);
      cudaCheck(cudaGetLastError());

      kernelLineFit<4><<<numberOfBlocks_blf4 / 4, blockSize_blf4, 0, stream>>>(tupleMultiplicity_d,
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
      kernelFastFit<5><<<numberOfBlocks_ff5 / 4, blockSize_ff5, 0, stream>>>(
          tuples_d, tupleMultiplicity_d, 5, hv, hitsGPU_.get(), hits_geGPU_.get(), fast_fit_resultsGPU_.get(), offset);
      cudaCheck(cudaGetLastError());

      kernelCircleFit<5><<<numberOfBlocks_clf5 / 4, blockSize_clf5, 0, stream>>>(tupleMultiplicity_d,
                                                                       5,
                                                                       bField_,
                                                                       hitsGPU_.get(),
                                                                       hits_geGPU_.get(),
                                                                       fast_fit_resultsGPU_.get(),
                                                                       circle_fit_resultsGPU_,
                                                                       offset);
      cudaCheck(cudaGetLastError());

      kernelLineFit<5><<<numberOfBlocks_blf5 / 4, blockSize_blf5, 0, stream>>>(tupleMultiplicity_d,
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
