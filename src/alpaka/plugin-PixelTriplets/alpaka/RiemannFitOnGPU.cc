#include "AlpakaCore/config.h"
#include "AlpakaCore/memory.h"
#include "AlpakaCore/workdivision.h"

#include "RiemannFitOnGPU.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void HelixFitOnGPU::launchRiemannKernels(HitsView const* hv,
                                           uint32_t nhits,
                                           uint32_t maxNumberOfTuples,
                                           Queue& queue) {
    ALPAKA_ASSERT_ACC(tuples_d);

    const auto blockSize = 64;
    const auto numberOfBlocks = cms::alpakatools::divide_up_by(maxNumberOfConcurrentFits_, blockSize);
    const auto workDivTriplets = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    const auto workDivQuadsPenta = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks / 4, blockSize);

    //  Fit internals
    auto hitsGPU_ = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double));
    auto hits_geGPU_ = cms::alpakatools::make_device_buffer<float[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float));
    auto fast_fit_resultsGPU_ = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double));

    auto circle_fit_resultsGPU_ =
        cms::alpakatools::make_device_buffer<Rfit::circle_fit[]>(queue, maxNumberOfConcurrentFits_);

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // triplets
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivTriplets,
                                                      kernelFastFit<3>(),
                                                      tuples_d,
                                                      tupleMultiplicity_d,
                                                      3,
                                                      hv,
                                                      hitsGPU_.data(),
                                                      hits_geGPU_.data(),
                                                      fast_fit_resultsGPU_.data(),
                                                      offset));

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivTriplets,
                                                      kernelCircleFit<3>(),
                                                      tupleMultiplicity_d,
                                                      3,
                                                      bField_,
                                                      hitsGPU_.data(),
                                                      hits_geGPU_.data(),
                                                      fast_fit_resultsGPU_.data(),
                                                      circle_fit_resultsGPU_.data(),
                                                      offset));

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivTriplets,
                                                      kernelLineFit<3>(),
                                                      tupleMultiplicity_d,
                                                      3,
                                                      bField_,
                                                      outputSoa_d,
                                                      hitsGPU_.data(),
                                                      hits_geGPU_.data(),
                                                      fast_fit_resultsGPU_.data(),
                                                      circle_fit_resultsGPU_.data(),
                                                      offset));

      // quads
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                      kernelFastFit<4>(),
                                                      tuples_d,
                                                      tupleMultiplicity_d,
                                                      4,
                                                      hv,
                                                      hitsGPU_.data(),
                                                      hits_geGPU_.data(),
                                                      fast_fit_resultsGPU_.data(),
                                                      offset));

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                      kernelCircleFit<4>(),
                                                      tupleMultiplicity_d,
                                                      4,
                                                      bField_,
                                                      hitsGPU_.data(),
                                                      hits_geGPU_.data(),
                                                      fast_fit_resultsGPU_.data(),
                                                      circle_fit_resultsGPU_.data(),
                                                      offset));

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                      kernelLineFit<4>(),
                                                      tupleMultiplicity_d,
                                                      4,
                                                      bField_,
                                                      outputSoa_d,
                                                      hitsGPU_.data(),
                                                      hits_geGPU_.data(),
                                                      fast_fit_resultsGPU_.data(),
                                                      circle_fit_resultsGPU_.data(),
                                                      offset));

      if (fit5as4_) {
        // penta
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                        kernelFastFit<4>(),
                                                        tuples_d,
                                                        tupleMultiplicity_d,
                                                        5,
                                                        hv,
                                                        hitsGPU_.data(),
                                                        hits_geGPU_.data(),
                                                        fast_fit_resultsGPU_.data(),
                                                        offset));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                        kernelCircleFit<4>(),
                                                        tupleMultiplicity_d,
                                                        5,
                                                        bField_,
                                                        hitsGPU_.data(),
                                                        hits_geGPU_.data(),
                                                        fast_fit_resultsGPU_.data(),
                                                        circle_fit_resultsGPU_.data(),
                                                        offset));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                        kernelLineFit<4>(),
                                                        tupleMultiplicity_d,
                                                        5,
                                                        bField_,
                                                        outputSoa_d,
                                                        hitsGPU_.data(),
                                                        hits_geGPU_.data(),
                                                        fast_fit_resultsGPU_.data(),
                                                        circle_fit_resultsGPU_.data(),
                                                        offset));
      } else {
        // penta all 5
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                        kernelFastFit<5>(),
                                                        tuples_d,
                                                        tupleMultiplicity_d,
                                                        5,
                                                        hv,
                                                        hitsGPU_.data(),
                                                        hits_geGPU_.data(),
                                                        fast_fit_resultsGPU_.data(),
                                                        offset));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                        kernelCircleFit<5>(),
                                                        tupleMultiplicity_d,
                                                        5,
                                                        bField_,
                                                        hitsGPU_.data(),
                                                        hits_geGPU_.data(),
                                                        fast_fit_resultsGPU_.data(),
                                                        circle_fit_resultsGPU_.data(),
                                                        offset));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                        kernelLineFit<5>(),
                                                        tupleMultiplicity_d,
                                                        5,
                                                        bField_,
                                                        outputSoa_d,
                                                        hitsGPU_.data(),
                                                        hits_geGPU_.data(),
                                                        fast_fit_resultsGPU_.data(),
                                                        circle_fit_resultsGPU_.data(),
                                                        offset));
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
