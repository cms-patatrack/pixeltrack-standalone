#include "CUDACore/FwkContextBase.h"
#include "CUDACore/StreamCache.h"
#include "CUDACore/cudaCheck.h"

#include "chooseDevice.h"

namespace cms::cuda::impl {
  FwkContextBase::FwkContextBase(edm::StreamID streamID) : FwkContextBase(chooseDevice(streamID)) {}

  FwkContextBase::FwkContextBase(int device) : currentDevice_(device) { cudaCheck(cudaSetDevice(currentDevice_)); }

  FwkContextBase::FwkContextBase(int device, SharedStreamPtr stream)
    : currentDevice_(device), stream_(std::make_shared<impl::StreamSharingHelper>(std::move(stream))) {
    cudaCheck(cudaSetDevice(currentDevice_));
  }

  void FwkContextBase::initialize() { stream_ = std::make_shared<impl::StreamSharingHelper>(getStreamCache().get()); }

  void FwkContextBase::initialize(const ProductBase& data) {
    SharedStreamPtr stream;
    if (data.mayReuseStream()) {
      stream = data.streamPtr();
    } else {
      stream = getStreamCache().get();
    }
    stream_ = std::make_shared<impl::StreamSharingHelper>(std::move(stream));
  }
}
