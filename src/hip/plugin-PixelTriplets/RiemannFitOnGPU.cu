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
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelFastFit<3>),
                       dim3(numberOfBlocks),
                       dim3(blockSize),
                       0,
                       stream,
                       tuples_d,
                       tupleMultiplicity_d,
                       3,
                       hv,
                       hitsGPU_.get(),
                       hits_geGPU_.get(),
                       fast_fit_resultsGPU_.get(),
                       offset);
    cudaCheck(hipGetLastError());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelCircleFit<3>),
                       dim3(numberOfBlocks),
                       dim3(blockSize),
                       0,
                       stream,
                       tupleMultiplicity_d,
                       3,
                       bField_,
                       hitsGPU_.get(),
                       hits_geGPU_.get(),
                       fast_fit_resultsGPU_.get(),
                       circle_fit_resultsGPU_,
                       offset);
    cudaCheck(hipGetLastError());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelLineFit<3>),
                       dim3(numberOfBlocks),
                       dim3(blockSize),
                       0,
                       stream,
                       tupleMultiplicity_d,
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
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelFastFit<4>),
                       dim3(numberOfBlocks / 4),
                       dim3(blockSize),
                       0,
                       stream,
                       tuples_d,
                       tupleMultiplicity_d,
                       4,
                       hv,
                       hitsGPU_.get(),
                       hits_geGPU_.get(),
                       fast_fit_resultsGPU_.get(),
                       offset);
    cudaCheck(hipGetLastError());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelCircleFit<4>),
                       dim3(numberOfBlocks / 4),
                       dim3(blockSize),
                       0,
                       stream,
                       tupleMultiplicity_d,
                       4,
                       bField_,
                       hitsGPU_.get(),
                       hits_geGPU_.get(),
                       fast_fit_resultsGPU_.get(),
                       circle_fit_resultsGPU_,
                       offset);
    cudaCheck(hipGetLastError());

    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelLineFit<4>),
                       dim3(numberOfBlocks / 4),
                       dim3(blockSize),
                       0,
                       stream,
                       tupleMultiplicity_d,
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
      hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelFastFit<4>),
                         dim3(numberOfBlocks / 4),
                         dim3(blockSize),
                         0,
                         stream,
                         tuples_d,
                         tupleMultiplicity_d,
                         5,
                         hv,
                         hitsGPU_.get(),
                         hits_geGPU_.get(),
                         fast_fit_resultsGPU_.get(),
                         offset);
      cudaCheck(hipGetLastError());

      hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelCircleFit<4>),
                         dim3(numberOfBlocks / 4),
                         dim3(blockSize),
                         0,
                         stream,
                         tupleMultiplicity_d,
                         5,
                         bField_,
                         hitsGPU_.get(),
                         hits_geGPU_.get(),
                         fast_fit_resultsGPU_.get(),
                         circle_fit_resultsGPU_,
                         offset);
      cudaCheck(hipGetLastError());

      hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelLineFit<4>),
                         dim3(numberOfBlocks / 4),
                         dim3(blockSize),
                         0,
                         stream,
                         tupleMultiplicity_d,
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
      hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelFastFit<5>),
                         dim3(numberOfBlocks / 4),
                         dim3(blockSize),
                         0,
                         stream,
                         tuples_d,
                         tupleMultiplicity_d,
                         5,
                         hv,
                         hitsGPU_.get(),
                         hits_geGPU_.get(),
                         fast_fit_resultsGPU_.get(),
                         offset);
      cudaCheck(hipGetLastError());

      hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelCircleFit<5>),
                         dim3(numberOfBlocks / 4),
                         dim3(blockSize),
                         0,
                         stream,
                         tupleMultiplicity_d,
                         5,
                         bField_,
                         hitsGPU_.get(),
                         hits_geGPU_.get(),
                         fast_fit_resultsGPU_.get(),
                         circle_fit_resultsGPU_,
                         offset);
      cudaCheck(hipGetLastError());

      hipLaunchKernelGGL(HIP_KERNEL_NAME(kernelLineFit<5>),
                         dim3(numberOfBlocks / 4),
                         dim3(blockSize),
                         0,
                         stream,
                         tupleMultiplicity_d,
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
