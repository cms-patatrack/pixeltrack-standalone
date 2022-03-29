#include "hip/hip_runtime.h"
#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"

namespace testTrackingRecHit2D {

  __global__ void fill(TrackingRecHit2DSOAView* phits) {
    assert(phits);
    auto& hits = *phits;
    assert(hits.nHits() == 200);

    int i = threadIdx.x;
    if (i > 200)
      return;
  }

  __global__ void verify(TrackingRecHit2DSOAView const* phits) {
    assert(phits);
    auto const& hits = *phits;
    assert(hits.nHits() == 200);

    int i = threadIdx.x;
    if (i > 200)
      return;
  }

  void runKernels(TrackingRecHit2DSOAView* hits) {
    assert(hits);
    fill<<<1, 1024, 0, 0>>>(hits);
    verify<<<1, 1024, 0, 0>>>(hits);
  }

}  // namespace testTrackingRecHit2D

namespace testTrackingRecHit2D {

  void runKernels(TrackingRecHit2DSOAView* hits);

}

int main() {
  hipStream_t stream;
  cudaCheck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  // inner scope to deallocate memory before destroying the stream
  {
    auto nHits = 200;
    TrackingRecHit2DCUDA tkhit(nHits, nullptr, nullptr, stream);

    testTrackingRecHit2D::runKernels(tkhit.view());
  }

  cudaCheck(hipStreamDestroy(stream));

  return 0;
}
