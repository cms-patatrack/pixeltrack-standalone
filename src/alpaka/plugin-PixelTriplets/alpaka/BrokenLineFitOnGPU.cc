#include "BrokenLineFitOnGPU.h"

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void HelixFitOnGPU::launchBrokenLineKernels(HitsView const* hv,
                                              uint32_t hitsInFit,
                                              uint32_t maxNumberOfTuples,
                                              Queue& queue) {
    assert(tuples_d);

    const auto blockSize = 64;
    const auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;
    const WorkDiv1 workDivTriplets = cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks), Vec1::all(blockSize));
    const WorkDiv1 workDivQuadsPenta =
        cms::alpakatools::make_workdiv(Vec1::all(numberOfBlocks / 4), Vec1::all(blockSize));

    //  Fit internals
    auto hitsGPU_ = cms::alpakatools::allocDeviceBuf<double>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) /
                                                             sizeof(double));

    auto hits_geGPU_ =
        cms::alpakatools::allocDeviceBuf<float>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float));

    auto fast_fit_resultsGPU_ =
        cms::alpakatools::allocDeviceBuf<double>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double));

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // fit triplets
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(workDivTriplets,
                                                     kernelBLFastFit<3>(),
                                                     tuples_d,
                                                     tupleMultiplicity_d,
                                                     hv,
                                                     alpaka::getPtrNative(hitsGPU_),
                                                     alpaka::getPtrNative(hits_geGPU_),
                                                     alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                     3,
                                                     offset));

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(workDivTriplets,
                                                     kernelBLFit<3>(),
                                                     tupleMultiplicity_d,
                                                     bField_,
                                                     outputSoa_d,
                                                     alpaka::getPtrNative(hitsGPU_),
                                                     alpaka::getPtrNative(hits_geGPU_),
                                                     alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                     3,
                                                     offset));

      // fit quads
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                     kernelBLFastFit<4>(),
                                                     tuples_d,
                                                     tupleMultiplicity_d,
                                                     hv,
                                                     alpaka::getPtrNative(hitsGPU_),
                                                     alpaka::getPtrNative(hits_geGPU_),
                                                     alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                     4,
                                                     offset));

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                     kernelBLFit<4>(),
                                                     tupleMultiplicity_d,
                                                     bField_,
                                                     outputSoa_d,
                                                     alpaka::getPtrNative(hitsGPU_),
                                                     alpaka::getPtrNative(hits_geGPU_),
                                                     alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                     4,
                                                     offset));

      if (fit5as4_) {
        // fit penta (only first 4)
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                       kernelBLFastFit<4>(),
                                                       tuples_d,
                                                       tupleMultiplicity_d,
                                                       hv,
                                                       alpaka::getPtrNative(hitsGPU_),
                                                       alpaka::getPtrNative(hits_geGPU_),
                                                       alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                       5,
                                                       offset));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                       kernelBLFit<4>(),
                                                       tupleMultiplicity_d,
                                                       bField_,
                                                       outputSoa_d,
                                                       alpaka::getPtrNative(hitsGPU_),
                                                       alpaka::getPtrNative(hits_geGPU_),
                                                       alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                       5,
                                                       offset));
        alpaka::wait(queue);
      } else {
        // fit penta (all 5)
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                       kernelBLFastFit<5>(),
                                                       tuples_d,
                                                       tupleMultiplicity_d,
                                                       hv,
                                                       alpaka::getPtrNative(hitsGPU_),
                                                       alpaka::getPtrNative(hits_geGPU_),
                                                       alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                       5,
                                                       offset));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                       kernelBLFit<5>(),
                                                       tupleMultiplicity_d,
                                                       bField_,
                                                       outputSoa_d,
                                                       alpaka::getPtrNative(hitsGPU_),
                                                       alpaka::getPtrNative(hits_geGPU_),
                                                       alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                       5,
                                                       offset));
        alpaka::wait(queue);
      }

    }  // loop on concurrent fits
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
