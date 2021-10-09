#include "CondFormats/SiPixelFedIds.h"
#include "CondFormats/SiPixelROCsStatusAndMapping.h"
#include "CondFormats/SiPixelROCsStatusAndMappingWrapper.h"
#include "CUDACore/ESContext.h"
#include "CUDACore/HostAllocator.h"
#include "CUDACore/copyAsync.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include <fstream>
#include <memory>

class SiPixelROCsStatusAndMappingWrapperESProducer : public edm::ESProducer {
public:
  explicit SiPixelROCsStatusAndMappingWrapperESProducer(std::filesystem::path const& datadir) : data_(datadir) {}
  void produce(edm::EventSetup& eventSetup);

private:
  std::filesystem::path data_;
};

void SiPixelROCsStatusAndMappingWrapperESProducer::produce(edm::EventSetup& eventSetup) {
  {
    std::ifstream in(data_ / "fedIds.bin", std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    unsigned int nfeds;
    in.read(reinterpret_cast<char*>(&nfeds), sizeof(unsigned));
    std::vector<unsigned int> fedIds(nfeds);
    in.read(reinterpret_cast<char*>(fedIds.data()), sizeof(unsigned int) * nfeds);
    eventSetup.put(std::make_unique<SiPixelFedIds>(std::move(fedIds)));
  }
  {
    std::ifstream in(data_ / "cablingMap.bin", std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    SiPixelROCsStatusAndMapping obj;
    in.read(reinterpret_cast<char*>(&obj), sizeof(SiPixelROCsStatusAndMapping));
    unsigned int modToUnpDefSize;
    in.read(reinterpret_cast<char*>(&modToUnpDefSize), sizeof(unsigned int));
    std::vector<unsigned char> modToUnpDefault(modToUnpDefSize);
    in.read(reinterpret_cast<char*>(modToUnpDefault.data()), modToUnpDefSize);

    eventSetup.put(cms::cuda::runForHost([&](cms::cuda::HostAllocatorContext& ctx) {
                     auto cablingMapHost = cms::cuda::make_host_unique_uninitialized<SiPixelROCsStatusAndMapping>(ctx);
                     *cablingMapHost = obj;
                     std::vector<unsigned char, cms::cuda::HostAllocator<unsigned char>> modToUnp(
                         modToUnpDefault.size());
                     std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
                     return std::tuple(std::move(cablingMapHost), std::move(modToUnp));
                   }).forEachDevice([&](auto const& tpl, cms::cuda::ESContext& ctx) {
      auto const& [cablingMapHost, modToUnp] = tpl;
      auto cablingMapDevice = cms::cuda::make_device_unique_uninitialized<SiPixelROCsStatusAndMapping>(ctx);
      auto modToUnpDevice = cms::cuda::make_device_unique<unsigned char[]>(pixelgpudetails::MAX_SIZE_BYTE_BOOL, ctx);
      cms::cuda::copyAsync(cablingMapDevice, cablingMapHost, ctx.stream());
      cudaCheck(cudaMemcpyAsync(modToUnpDevice.get(),
                                modToUnp.data(),
                                modToUnp.size() * sizeof(unsigned char),
                                cudaMemcpyDefault,
                                ctx.stream()));
      return SiPixelROCsStatusAndMappingWrapper(std::move(cablingMapDevice), std::move(modToUnpDevice));
    }));
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelROCsStatusAndMappingWrapperESProducer);
