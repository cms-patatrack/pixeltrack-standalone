#include "RiemannFitOnGPU.h"

#include "AlpakaCore/alpakaCommon.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void HelixFitOnGPU::launchRiemannKernels(HitsView const* hv,
                                           uint32_t nhits,
                                           uint32_t maxNumberOfTuples,
                                           Queue& queue) {
    ALPAKA_ASSERT_OFFLOAD(tuples_d);

    const auto blockSize = 64;
    const auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;
    const WorkDiv1D workDivTriplets = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(
        Vec1D::all(numberOfBlocks), Vec1D::all(blockSize));
    const WorkDiv1D workDivQuadsPenta = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::make_workdiv(
        Vec1D::all(numberOfBlocks / 4), Vec1D::all(blockSize));

    //  Fit internals
    auto hitsGPU_ = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::allocDeviceBuf<double>(
        maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double));

    auto hits_geGPU_ = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::allocDeviceBuf<float>(
        maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float));

    auto fast_fit_resultsGPU_ = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::allocDeviceBuf<double>(
        maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double));

    //auto circle_fit_resultsGPU_holder =
    //cms::cuda::make_device_unique<char[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit), stream);
    //Rfit::circle_fit *circle_fit_resultsGPU_ = (Rfit::circle_fit *)(circle_fit_resultsGPU_holder.get());
    //auto circle_fit_resultsGPU_holder = ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::allocDeviceBuf<char>(maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit));
    auto circle_fit_resultsGPU_ =
        ::cms::alpakatools::ALPAKA_ACCELERATOR_NAMESPACE::allocDeviceBuf<Rfit::circle_fit>(maxNumberOfConcurrentFits_);

    for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
      // triplets
      alpaka::enqueue(queue,
                      alpaka::createTaskKernel<Acc1D>(workDivTriplets,
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
                      alpaka::createTaskKernel<Acc1D>(workDivTriplets,
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
                      alpaka::createTaskKernel<Acc1D>(workDivTriplets,
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
                      alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
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
                      alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
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
                      alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
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
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
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
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
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
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
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
      } else {
        // penta all 5
        alpaka::enqueue(queue,
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
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
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
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
                        alpaka::createTaskKernel<Acc1D>(workDivQuadsPenta,
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
      }
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
