#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"

namespace testTrackingRecHit2D {

  __global__ void fill(TrackingRecHit2DSOAView* phits) {
    assert(phits);
    assert(phits->nHits() == 200);

    int i = threadIdx.x;
    if (i > 200)
      return;

    // FIXME do something ?
  }

  __global__ void verify(TrackingRecHit2DSOAView const* phits) {
    assert(phits);
    assert(phits->nHits() == 200);

    int i = threadIdx.x;
    if (i > 200)
      return;

    // FIXME do something ?
  }

  void runKernels(TrackingRecHit2DSOAView* hits) {
    assert(hits);
    fill<<<1, 1024>>>(hits);
    verify<<<1, 1024>>>(hits);
  }

}  // namespace testTrackingRecHit2D

int main() {
  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // inner scope to deallocate memory before destroying the stream
  {
    auto nHits = 200;
    TrackingRecHit2DCUDA tkhit(nHits, nullptr, nullptr, stream);

    testTrackingRecHit2D::runKernels(tkhit.view());
  }

  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}
