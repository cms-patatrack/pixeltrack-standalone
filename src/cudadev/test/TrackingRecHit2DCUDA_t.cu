#include "CUDADataFormats/TrackingRecHit2DHeterogeneous.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/cudaCheck.h"

namespace testTrackingRecHit2D {

  __global__ void fill(TrackingRecHit2DSOAStore* phits) {
    assert(phits);
    [[maybe_unused]] auto& hits = *phits;
    assert(hits.nHits() == 200);

    int i = threadIdx.x;
    if (i > 200)
      return;
  }

  __global__ void verify(TrackingRecHit2DSOAStore const* phits) {
    assert(phits);
    [[maybe_unused]] auto const& hits = *phits;
    assert(hits.nHits() == 200);

    int i = threadIdx.x;
    if (i > 200)
      return;
  }

  void runKernels(TrackingRecHit2DSOAStore* hits) {
    assert(hits);
    fill<<<1, 1024>>>(hits);
    verify<<<1, 1024>>>(hits);
  }

}  // namespace testTrackingRecHit2D

namespace testTrackingRecHit2D {

  void runKernels(TrackingRecHit2DSOAStore* hits);

}

int main() {
  cudaStream_t stream;
  cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // inner scope to deallocate memory before destroying the stream
  {
    auto nHits = 200;
    TrackingRecHit2DCUDA tkhit(nHits, nullptr, nullptr, stream);

    testTrackingRecHit2D::runKernels(tkhit.store());
  }

  cudaCheck(cudaStreamDestroy(stream));

  return 0;
}