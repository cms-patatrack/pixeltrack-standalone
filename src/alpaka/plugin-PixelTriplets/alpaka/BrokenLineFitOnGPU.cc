#include "AlpakaCore/config.h"
#include "AlpakaCore/memory.h"
#include "AlpakaCore/workdivision.h"

#include "BrokenLineFitOnGPU.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void HelixFitOnGPU::launchBrokenLineKernels(HitsView const* hv,
                                              uint32_t hitsInFit,
                                              uint32_t maxNumberOfTuples,
                                              Queue& queue) {
    ALPAKA_ASSERT_ACC(tuples_d);

    const auto blockSize = 64;
    const auto numberOfBlocks = cms::alpakatools::divide_up_by(maxNumberOfConcurrentFits_, blockSize);
    const WorkDiv1D workDivTriplets = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks, blockSize);
    const WorkDiv1D workDivQuadsPenta = cms::alpakatools::make_workdiv<Acc1D>(numberOfBlocks / 4, blockSize);

    //  Fit internals
    auto hitsGPU_ = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double));
    auto hits_geGPU_ = cms::alpakatools::make_device_buffer<float[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float));
    auto fast_fit_resultsGPU_ = cms::alpakatools::make_device_buffer<double[]>(
        queue, maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double));

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // fit triplets
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivTriplets,
                                                      kernelBLFastFit<3>(),
                                                      tuples_d,
                                                      tupleMultiplicity_d,
                                                      hv,
                                                      hitsGPU_.data(),
                                                      hits_geGPU_.data(),
                                                      fast_fit_resultsGPU_.data(),
                                                      3,
                                                      offset));

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivTriplets,
                                                      kernelBLFit<3>(),
                                                      tupleMultiplicity_d,
                                                      bField_,
                                                      outputSoa_d,
                                                      hitsGPU_.data(),
                                                      hits_geGPU_.data(),
                                                      fast_fit_resultsGPU_.data(),
                                                      3,
                                                      offset));

      // fit quads
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                      kernelBLFastFit<4>(),
                                                      tuples_d,
                                                      tupleMultiplicity_d,
                                                      hv,
                                                      hitsGPU_.data(),
                                                      hits_geGPU_.data(),
                                                      fast_fit_resultsGPU_.data(),
                                                      4,
                                                      offset));

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                      kernelBLFit<4>(),
                                                      tupleMultiplicity_d,
                                                      bField_,
                                                      outputSoa_d,
                                                      hitsGPU_.data(),
                                                      hits_geGPU_.data(),
                                                      fast_fit_resultsGPU_.data(),
                                                      4,
                                                      offset));

      if (fit5as4_) {
        // fit penta (only first 4)
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                        kernelBLFastFit<4>(),
                                                        tuples_d,
                                                        tupleMultiplicity_d,
                                                        hv,
                                                        hitsGPU_.data(),
                                                        hits_geGPU_.data(),
                                                        fast_fit_resultsGPU_.data(),
                                                        5,
                                                        offset));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                        kernelBLFit<4>(),
                                                        tupleMultiplicity_d,
                                                        bField_,
                                                        outputSoa_d,
                                                        hitsGPU_.data(),
                                                        hits_geGPU_.data(),
                                                        fast_fit_resultsGPU_.data(),
                                                        5,
                                                        offset));
      } else {
        // fit penta (all 5)
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                        kernelBLFastFit<5>(),
                                                        tuples_d,
                                                        tupleMultiplicity_d,
                                                        hv,
                                                        hitsGPU_.data(),
                                                        hits_geGPU_.data(),
                                                        fast_fit_resultsGPU_.data(),
                                                        5,
                                                        offset));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
                                                        kernelBLFit<5>(),
                                                        tupleMultiplicity_d,
                                                        bField_,
                                                        outputSoa_d,
                                                        hitsGPU_.data(),
                                                        hits_geGPU_.data(),
                                                        fast_fit_resultsGPU_.data(),
                                                        5,
                                                        offset));
      }

    }  // loop on concurrent fits
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
