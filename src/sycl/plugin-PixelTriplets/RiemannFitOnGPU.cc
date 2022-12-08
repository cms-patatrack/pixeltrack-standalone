#include "RiemannFitOnGPU.h"

void HelixFitOnGPU::launchRiemannKernels(HitsView const *hv,
                                         uint32_t nhits,
                                         uint32_t maxNumberOfTuples,
                                         sycl::queue stream) {
  assert(tuples_d);

  auto blockSize = 64;
  auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

  //  Fit internals
  auto hitsGPU_ = cms::sycltools::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double), stream);
  auto hits_geGPU_ = cms::sycltools::make_device_unique<float[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float), stream);
  auto fast_fit_resultsGPU_ = cms::sycltools::make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double), stream);
  auto circle_fit_resultsGPU_holder =
      cms::sycltools::make_device_unique<char[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit), stream);

  Rfit::circle_fit *circle_fit_resultsGPU_ = (Rfit::circle_fit *)(circle_fit_resultsGPU_holder.get());

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // triplets
    stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel = tuples_d;
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
      auto hv_kernel = hv;
      auto hitsGPU_kernel = hitsGPU_.get();
      auto hits_geGPU_kernel = hits_geGPU_.get();
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for<class FastFit_3_3_Kernel>(sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
                                                 [=](sycl::nd_item<1> item) {
                                                   kernelFastFit<3>(tuples_d_kernel,
                                                                    tupleMultiplicity_d_kernel,
                                                                    3,
                                                                    hv_kernel,
                                                                    hitsGPU_kernel,
                                                                    hits_geGPU_kernel,
                                                                    fast_fit_resultsGPU_kernel,
                                                                    offset,
                                                                    item);
                                                 });
    });

    stream.submit([&](sycl::handler &cgh) {
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
      auto bField_kernel = bField_;
      auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
      auto hitsGPU_kernel = hitsGPU_.get();
      auto hits_geGPU_kernel = hits_geGPU_.get();
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for<class CircleFit_3_3_Kernel>(sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
                                                   [=](sycl::nd_item<1> item) {
                                                     kernelCircleFit<3>(tupleMultiplicity_d_kernel,
                                                                        3,
                                                                        bField_kernel,
                                                                        hitsGPU_kernel,
                                                                        hits_geGPU_kernel,
                                                                        fast_fit_resultsGPU_kernel,
                                                                        circle_fit_resultsGPU_kernel,
                                                                        offset,
                                                                        item);
                                                   });
    });

    stream.submit([&](sycl::handler &cgh) {
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
      auto bField_kernel = bField_;
      auto outputSoa_d_kernel = outputSoa_d;
      auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
      auto hitsGPU_kernel = hitsGPU_.get();
      auto hits_geGPU_kernel = hits_geGPU_.get();
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for<class LineFit_3_3_Kernel>(sycl::nd_range<1>(numberOfBlocks * blockSize, blockSize),
                                                 [=](sycl::nd_item<1> item) {
                                                   kernelLineFit<3>(tupleMultiplicity_d_kernel,
                                                                    3,
                                                                    bField_kernel,
                                                                    outputSoa_d_kernel,
                                                                    hitsGPU_kernel,
                                                                    hits_geGPU_kernel,
                                                                    fast_fit_resultsGPU_kernel,
                                                                    circle_fit_resultsGPU_kernel,
                                                                    offset,
                                                                    item);
                                                 });
    });

    // quads
    stream.submit([&](sycl::handler &cgh) {
      auto tuples_d_kernel = tuples_d;
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
      auto hv_kernel = hv;
      auto hitsGPU_kernel = hitsGPU_.get();
      auto hits_geGPU_kernel = hits_geGPU_.get();
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for<class FastFit_4_4_Kernel>(sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
                                                 [=](sycl::nd_item<1> item) {
                                                   kernelFastFit<4>(tuples_d_kernel,
                                                                    tupleMultiplicity_d_kernel,
                                                                    4,
                                                                    hv_kernel,
                                                                    hitsGPU_kernel,
                                                                    hits_geGPU_kernel,
                                                                    fast_fit_resultsGPU_kernel,
                                                                    offset,
                                                                    item);
                                                 });
    });

    stream.submit([&](sycl::handler &cgh) {
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
      auto bField_kernel = bField_;
      auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
      auto hitsGPU_kernel = hitsGPU_.get();
      auto hits_geGPU_kernel = hits_geGPU_.get();
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for<class CircleFit_4_4_Kernel>(sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
                                                   [=](sycl::nd_item<1> item) {
                                                     kernelCircleFit<4>(tupleMultiplicity_d_kernel,  //<4>
                                                                        4,
                                                                        bField_kernel,
                                                                        hitsGPU_kernel,
                                                                        hits_geGPU_kernel,
                                                                        fast_fit_resultsGPU_kernel,
                                                                        circle_fit_resultsGPU_kernel,
                                                                        offset,
                                                                        item);
                                                   });
    });

    stream.submit([&](sycl::handler &cgh) {
      auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
      auto bField_kernel = bField_;
      auto outputSoa_d_kernel = outputSoa_d;
      auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
      auto hitsGPU_kernel = hitsGPU_.get();
      auto hits_geGPU_kernel = hits_geGPU_.get();
      auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
      cgh.parallel_for<class LineFit_4_4_Kernel>(sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
                                                 [=](sycl::nd_item<1> item) {
                                                   kernelLineFit<4>(tupleMultiplicity_d_kernel,
                                                                    4,
                                                                    bField_kernel,
                                                                    outputSoa_d_kernel,
                                                                    hitsGPU_kernel,
                                                                    hits_geGPU_kernel,
                                                                    fast_fit_resultsGPU_kernel,
                                                                    circle_fit_resultsGPU_kernel,
                                                                    offset,
                                                                    item);
                                                 });
    });

    if (fit5as4_) {
      // penta
      stream.submit([&](sycl::handler &cgh) {
        auto tuples_d_kernel = tuples_d;
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
        auto hv_kernel = hv;
        auto hitsGPU_kernel = hitsGPU_.get();
        auto hits_geGPU_kernel = hits_geGPU_.get();
        auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
        cgh.parallel_for<class FastFit_5_4_Kernel>(sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
                                                   [=](sycl::nd_item<1> item) {
                                                     kernelFastFit<4>(tuples_d_kernel,
                                                                      tupleMultiplicity_d_kernel,
                                                                      5,
                                                                      hv_kernel,
                                                                      hitsGPU_kernel,
                                                                      hits_geGPU_kernel,
                                                                      fast_fit_resultsGPU_kernel,
                                                                      offset,
                                                                      item);
                                                   });
      });

      stream.submit([&](sycl::handler &cgh) {
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
        auto bField_kernel = bField_;
        auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
        auto hitsGPU_kernel = hitsGPU_.get();
        auto hits_geGPU_kernel = hits_geGPU_.get();
        auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
        cgh.parallel_for<class CircleFit_5_4_Kernel>(sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
                                                     [=](sycl::nd_item<1> item) {
                                                       kernelCircleFit<4>(tupleMultiplicity_d_kernel,  //<4>
                                                                          5,
                                                                          bField_kernel,
                                                                          hitsGPU_kernel,
                                                                          hits_geGPU_kernel,
                                                                          fast_fit_resultsGPU_kernel,
                                                                          circle_fit_resultsGPU_kernel,
                                                                          offset,
                                                                          item);
                                                     });
      });

      stream.submit([&](sycl::handler &cgh) {
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
        auto bField_kernel = bField_;
        auto outputSoa_d_kernel = outputSoa_d;
        auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
        auto hitsGPU_kernel = hitsGPU_.get();
        auto hits_geGPU_kernel = hits_geGPU_.get();
        auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
        cgh.parallel_for<class LineFit_5_4_Kernel>(sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
                                                   [=](sycl::nd_item<1> item) {
                                                     kernelLineFit<4>(tupleMultiplicity_d_kernel,
                                                                      5,
                                                                      bField_kernel,
                                                                      outputSoa_d_kernel,
                                                                      hitsGPU_kernel,
                                                                      hits_geGPU_kernel,
                                                                      fast_fit_resultsGPU_kernel,
                                                                      circle_fit_resultsGPU_kernel,
                                                                      offset,
                                                                      item);
                                                   });
      });

    } else {
      // penta all 5
      stream.submit([&](sycl::handler &cgh) {
        auto tuples_d_kernel = tuples_d;
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
        auto hv_kernel = hv;
        auto hitsGPU_kernel = hitsGPU_.get();
        auto hits_geGPU_kernel = hits_geGPU_.get();
        auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
        cgh.parallel_for<class FastFit_5_5_Kernel>(sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
                                                   [=](sycl::nd_item<1> item) {
                                                     kernelFastFit<5>(tuples_d_kernel,
                                                                      tupleMultiplicity_d_kernel,
                                                                      5,
                                                                      hv_kernel,
                                                                      hitsGPU_kernel,
                                                                      hits_geGPU_kernel,
                                                                      fast_fit_resultsGPU_kernel,
                                                                      offset,
                                                                      item);
                                                   });
      });

      stream.submit([&](sycl::handler &cgh) {
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
        auto bField_kernel = bField_;
        auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
        auto hitsGPU_kernel = hitsGPU_.get();
        auto hits_geGPU_kernel = hits_geGPU_.get();
        auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
        cgh.parallel_for<class CircleFit_5_5_Kernel>(sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
                                                     [=](sycl::nd_item<1> item) {
                                                       kernelCircleFit<5>(tupleMultiplicity_d_kernel,  //<5>
                                                                          5,
                                                                          bField_kernel,
                                                                          hitsGPU_kernel,
                                                                          hits_geGPU_kernel,
                                                                          fast_fit_resultsGPU_kernel,
                                                                          circle_fit_resultsGPU_kernel,
                                                                          offset,
                                                                          item);
                                                     });
      });

      stream.submit([&](sycl::handler &cgh) {
        auto tupleMultiplicity_d_kernel = tupleMultiplicity_d;
        auto bField_kernel = bField_;
        auto outputSoa_d_kernel = outputSoa_d;
        auto circle_fit_resultsGPU_kernel = circle_fit_resultsGPU_;
        auto hitsGPU_kernel = hitsGPU_.get();
        auto hits_geGPU_kernel = hits_geGPU_.get();
        auto fast_fit_resultsGPU_kernel = fast_fit_resultsGPU_.get();
        cgh.parallel_for<class LineFit_5_5_Kernel>(sycl::nd_range<1>(numberOfBlocks / 4 * blockSize, blockSize),
                                                   [=](sycl::nd_item<1> item) {
                                                     kernelLineFit<5>(tupleMultiplicity_d_kernel,
                                                                      5,
                                                                      bField_kernel,
                                                                      outputSoa_d_kernel,
                                                                      hitsGPU_kernel,
                                                                      hits_geGPU_kernel,
                                                                      fast_fit_resultsGPU_kernel,
                                                                      circle_fit_resultsGPU_kernel,
                                                                      offset,
                                                                      item);
                                                   });
      });
    }
  }
}
