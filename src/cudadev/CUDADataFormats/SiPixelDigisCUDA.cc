#include "CUDADataFormats/SiPixelDigisCUDA.h"

#include "CUDACore/copyAsync.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream)
    : data_d(cms::cuda::make_device_unique<std::byte[]>(
        DeviceOnlyLayout::computeDataSize(maxFedWords) + 
        HostDeviceLayout::computeDataSize(maxFedWords),
      stream)),
      deviceOnlyLayout_d(data_d.get(), maxFedWords),
      hostDeviceLayout_d(deviceOnlyLayout_d.soaMetadata().nextByte(), maxFedWords),
      deviceFullView_(deviceOnlyLayout_d, hostDeviceLayout_d),
      devicePixelConstView_(deviceFullView_)
{}

SiPixelDigisCUDA::SiPixelDigisCUDA()
    : data_d(),deviceOnlyLayout_d(), hostDeviceLayout_d(), deviceFullView_(), devicePixelConstView_()
{}

SiPixelDigisCUDA::HostStore::HostStore()
     : data_h(), hostLayout_(nullptr, 0), hostView_(hostLayout_)
{}

SiPixelDigisCUDA::HostStore::HostStore(size_t maxFedWords, cudaStream_t stream)
     : data_h(cms::cuda::make_host_unique<std::byte[]>(SiPixelDigisCUDA::HostDeviceLayout::computeDataSize(maxFedWords), stream)),
       hostLayout_(data_h.get(), maxFedWords),
       hostView_(hostLayout_)
{}

void SiPixelDigisCUDA::HostStore::reset() {
  hostLayout_ = HostDeviceLayout();
  hostView_ = HostDeviceView(hostLayout_);
  data_h.reset();
}

cms::cuda::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::adcToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(nDigis(), stream);
  // TODO: this is downgraded from cms::cuda::copyAsync as we copy data from within a block but not the full block.
  cudaCheck(cudaMemcpyAsync(ret.get(), deviceFullView_.adc(), nDigis() * sizeof(decltype(ret[0])), cudaMemcpyDeviceToHost, stream));
  return ret;
}

SiPixelDigisCUDA::HostStore SiPixelDigisCUDA::dataToHostAsync(cudaStream_t stream) const {
  // Allocate the needed space only and build the compact data in place in host memory (from the larger device memory).
  // Due to the compaction with the 2D copy, we need to know the precise geometry, and hence operate on the store (as opposed
  // to the view, which is unaware of the column pitches.
  HostStore ret(nDigis(), stream);
  auto rhlsm = ret.hostLayout_.soaMetadata();
  auto hdlsm_d = hostDeviceLayout_d.soaMetadata();
  cudaCheck(cudaMemcpyAsync(rhlsm.addressOf_adc(), hdlsm_d.addressOf_adc(), nDigis_h * sizeof(*rhlsm.addressOf_adc()),
          cudaMemcpyDeviceToHost, stream));
  // Copy the other columns, realigning the data in shorter arrays. clus is the first but all 3 columns (clus, pdigis, rawIdArr) have
  // the same geometry.
  cudaCheck(cudaMemcpy2DAsync(rhlsm.addressOf_clus(), rhlsm.clusPitch(),
          hdlsm_d.addressOf_clus(), hdlsm_d.clusPitch(),
          3 /* rows */,
          nDigis() * sizeof(decltype (*ret.hostView_.clus())), cudaMemcpyDeviceToHost, stream));
  return ret;
}