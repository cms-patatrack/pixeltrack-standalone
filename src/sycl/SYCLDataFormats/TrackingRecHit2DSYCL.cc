#include "SYCLDataFormats/TrackingRecHit2DSYCL.h"
#include "SYCLCore/device_unique_ptr.h"
#include "SYCLCore/host_unique_ptr.h"

cms::sycltools::host::unique_ptr<float[]> TrackingRecHit2DSYCL::localCoordToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<float[]>(4 * nHits(), stream);
  stream.memcpy(ret.get(), m_store32.get(), 4 * nHits() * sizeof(float));
  return ret;
}

cms::sycltools::host::unique_ptr<uint32_t[]> TrackingRecHit2DSYCL::hitsModuleStartToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<uint32_t[]>(2001, stream);
  stream.memcpy(ret.get(), m_hitsModuleStart, 4 * 2001 * sizeof(uint32_t));
  return ret;
}

cms::sycltools::host::unique_ptr<float[]> TrackingRecHit2DSYCL::globalCoordToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<float[]>(4 * nHits(), stream);
  stream.memcpy(ret.get(), m_store32.get() + 4 * nHits(), 4 * nHits() * sizeof(float));
  return ret;
}

cms::sycltools::host::unique_ptr<int32_t[]> TrackingRecHit2DSYCL::chargeToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<int32_t[]>(nHits(), stream);
  stream.memcpy(ret.get(), m_store32.get() + 8 * nHits(), nHits() * sizeof(int32_t));
  return ret;
}

cms::sycltools::host::unique_ptr<int16_t[]> TrackingRecHit2DSYCL::sizeToHostAsync(sycl::queue stream) const {
  auto ret = cms::sycltools::make_host_unique<int16_t[]>(2 * nHits(), stream);
  stream.memcpy(ret.get(), m_store16.get() + 2 * nHits(), 2 * nHits() * sizeof(int16_t));
  return ret;
}
