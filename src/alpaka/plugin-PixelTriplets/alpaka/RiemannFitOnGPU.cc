#include "RiemannFitOnGPU.h"

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void HelixFitOnGPU::launchRiemannKernels(HitsView const* hv,
                                           uint32_t nhits,
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

    //auto circle_fit_resultsGPU_holder =
    //cms::cuda::make_device_unique<char[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit), stream);
    //Rfit::circle_fit *circle_fit_resultsGPU_ = (Rfit::circle_fit *)(circle_fit_resultsGPU_holder.get());
    //auto circle_fit_resultsGPU_holder = cms::alpakatools::allocDeviceBuf<char>(maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit));
    auto circle_fit_resultsGPU_ = cms::alpakatools::allocDeviceBuf<Rfit::circle_fit>(maxNumberOfConcurrentFits_);

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // triplets
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(workDivTriplets,
                                                     kernelFastFit<3>(),
                                                     tuples_d,
                                                     tupleMultiplicity_d,
                                                     3,
                                                     hv,
                                                     alpaka::getPtrNative(hitsGPU_),
                                                     alpaka::getPtrNative(hits_geGPU_),
                                                     alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                     offset));

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(workDivTriplets,
                                                     kernelCircleFit<3>(),
                                                     tupleMultiplicity_d,
                                                     3,
                                                     bField_,
                                                     alpaka::getPtrNative(hitsGPU_),
                                                     alpaka::getPtrNative(hits_geGPU_),
                                                     alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                     alpaka::getPtrNative(circle_fit_resultsGPU_),
                                                     offset));

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(workDivTriplets,
                                                     kernelLineFit<3>(),
                                                     tupleMultiplicity_d,
                                                     3,
                                                     bField_,
                                                     outputSoa_d,
                                                     alpaka::getPtrNative(hitsGPU_),
                                                     alpaka::getPtrNative(hits_geGPU_),
                                                     alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                     alpaka::getPtrNative(circle_fit_resultsGPU_),
                                                     offset));

      // quads
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                     kernelFastFit<4>(),
                                                     tuples_d,
                                                     tupleMultiplicity_d,
                                                     4,
                                                     hv,
                                                     alpaka::getPtrNative(hitsGPU_),
                                                     alpaka::getPtrNative(hits_geGPU_),
                                                     alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                     offset));

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                     kernelCircleFit<4>(),
                                                     tupleMultiplicity_d,
                                                     4,
                                                     bField_,
                                                     alpaka::getPtrNative(hitsGPU_),
                                                     alpaka::getPtrNative(hits_geGPU_),
                                                     alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                     alpaka::getPtrNative(circle_fit_resultsGPU_),
                                                     offset));

      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                     kernelLineFit<4>(),
                                                     tupleMultiplicity_d,
                                                     4,
                                                     bField_,
                                                     outputSoa_d,
                                                     alpaka::getPtrNative(hitsGPU_),
                                                     alpaka::getPtrNative(hits_geGPU_),
                                                     alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                     alpaka::getPtrNative(circle_fit_resultsGPU_),
                                                     offset));

      if (fit5as4_) {
        // penta
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                       kernelFastFit<4>(),
                                                       tuples_d,
                                                       tupleMultiplicity_d,
                                                       5,
                                                       hv,
                                                       alpaka::getPtrNative(hitsGPU_),
                                                       alpaka::getPtrNative(hits_geGPU_),
                                                       alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                       offset));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                       kernelCircleFit<4>(),
                                                       tupleMultiplicity_d,
                                                       5,
                                                       bField_,
                                                       alpaka::getPtrNative(hitsGPU_),
                                                       alpaka::getPtrNative(hits_geGPU_),
                                                       alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                       alpaka::getPtrNative(circle_fit_resultsGPU_),
                                                       offset));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                       kernelLineFit<4>(),
                                                       tupleMultiplicity_d,
                                                       5,
                                                       bField_,
                                                       outputSoa_d,
                                                       alpaka::getPtrNative(hitsGPU_),
                                                       alpaka::getPtrNative(hits_geGPU_),
                                                       alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                       alpaka::getPtrNative(circle_fit_resultsGPU_),
                                                       offset));
        alpaka::wait(queue);
      } else {
        // penta all 5
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                       kernelFastFit<5>(),
                                                       tuples_d,
                                                       tupleMultiplicity_d,
                                                       5,
                                                       hv,
                                                       alpaka::getPtrNative(hitsGPU_),
                                                       alpaka::getPtrNative(hits_geGPU_),
                                                       alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                       offset));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                       kernelCircleFit<5>(),
                                                       tupleMultiplicity_d,
                                                       5,
                                                       bField_,
                                                       alpaka::getPtrNative(hitsGPU_),
                                                       alpaka::getPtrNative(hits_geGPU_),
                                                       alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                       alpaka::getPtrNative(circle_fit_resultsGPU_),
                                                       offset));

        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1>(workDivQuadsPenta,
                                                       kernelLineFit<5>(),
                                                       tupleMultiplicity_d,
                                                       5,
                                                       bField_,
                                                       outputSoa_d,
                                                       alpaka::getPtrNative(hitsGPU_),
                                                       alpaka::getPtrNative(hits_geGPU_),
                                                       alpaka::getPtrNative(fast_fit_resultsGPU_),
                                                       alpaka::getPtrNative(circle_fit_resultsGPU_),
                                                       offset));
        alpaka::wait(queue);
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
