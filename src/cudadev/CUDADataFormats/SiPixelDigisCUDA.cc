#include "CUDADataFormats/SiPixelDigisCUDA.h"

#include "CUDACore/copyAsync.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream)
    : data_d(cms::cuda::make_device_unique<std::byte[]>(
        DeviceOnlyStore::computeDataSize(maxFedWords) + 
        HostDeviceStore::computeDataSize(maxFedWords),
      stream)),
      deviceOnlyStore_d(data_d.get(), maxFedWords),
      hostDeviceStore_d(deviceOnlyStore_d.soaMetadata().nextByte(), maxFedWords),
      deviceFullView_(deviceOnlyStore_d, hostDeviceStore_d),
      devicePixelView_(deviceFullView_)
{}

SiPixelDigisCUDA::SiPixelDigisCUDA()
    : data_d(),deviceOnlyStore_d(), hostDeviceStore_d(), deviceFullView_(), devicePixelView_()
{}

SiPixelDigisCUDA::HostStoreAndBuffer::HostStoreAndBuffer()
     : data_h(), hostStore_(nullptr, 0)
{}

SiPixelDigisCUDA::HostStoreAndBuffer::HostStoreAndBuffer(size_t maxFedWords, cudaStream_t stream)
     : data_h(cms::cuda::make_host_unique<std::byte[]>(SiPixelDigisCUDA::HostDeviceStore::computeDataSize(maxFedWords), stream)),
       hostStore_(data_h.get(), maxFedWords)
{}

void SiPixelDigisCUDA::HostStoreAndBuffer::reset() {
  hostStore_.~HostDeviceStore();
  new(&hostStore_) HostDeviceStore(nullptr, 0);
  data_h.reset();
}

cms::cuda::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::adcToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(nDigis(), stream);
  // TODO: this is downgraded from cms::cuda::copyAsync as we copy data from within a block but not the full block.
  cudaCheck(cudaMemcpyAsync(ret.get(), deviceFullView_.adc(), nDigis() * sizeof(decltype(ret[0])), cudaMemcpyDeviceToHost, stream));
  return ret;
}

SiPixelDigisCUDA::HostStoreAndBuffer SiPixelDigisCUDA::dataToHostAsync(cudaStream_t stream) const {
  // Allocate the needed space only and build the compact data in place in host memory (from the larger device memory).
  // Due to the compaction with the 2D copy, we need to know the precise geometry, and hence operate on the store (as opposed
  // to the view, which is unaware of the column pitches.
  HostStoreAndBuffer ret(nDigis(), stream);
  cudaCheck(cudaMemcpyAsync(ret.hostStore_.adc(), hostDeviceStore_d.adc(), nDigis_h * sizeof(decltype(*deviceFullView_.adc())),
          cudaMemcpyDeviceToHost, stream));
  // Copy the other columns, realigning the data in shorter arrays. clus is the first but all 3 columns (clus, pdigis, rawIdArr) have
  // the same geometry.
  cudaCheck(cudaMemcpy2DAsync(ret.hostStore_.clus(), ret.hostStore_.soaMetadata().clusPitch(),
          hostDeviceStore_d.clus(), hostDeviceStore_d.soaMetadata().clusPitch(),
          3 /* rows */,
          nDigis() * sizeof(decltype (*ret.hostStore_.clus())), cudaMemcpyDeviceToHost, stream));
  return ret;
}