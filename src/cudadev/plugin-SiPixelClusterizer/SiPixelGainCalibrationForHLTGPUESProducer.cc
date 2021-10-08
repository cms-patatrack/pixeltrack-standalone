#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
#include "CUDACore/ESContext.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include <fstream>
#include <iostream>
#include <memory>

class SiPixelGainCalibrationForHLTGPUESProducer : public edm::ESProducer {
public:
  explicit SiPixelGainCalibrationForHLTGPUESProducer(std::filesystem::path const& datadir) : data_(datadir) {}
  void produce(edm::EventSetup& eventSetup);

private:
  std::filesystem::path data_;
};

void SiPixelGainCalibrationForHLTGPUESProducer::produce(edm::EventSetup& eventSetup) {
  std::ifstream in(data_ / "gain.bin", std::ios::binary);
  in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
  SiPixelGainForHLTonGPU gain;
  in.read(reinterpret_cast<char*>(&gain), sizeof(SiPixelGainForHLTonGPU));
  unsigned int nbytes;
  in.read(reinterpret_cast<char*>(&nbytes), sizeof(unsigned int));
  std::vector<char> gainData(nbytes);
  in.read(gainData.data(), nbytes);

  eventSetup.put(cms::cuda::runForHost([&](cms::cuda::HostAllocatorContext& ctx) {
                   auto gainForHLTonHost = cms::cuda::make_host_unique_uninitialized<SiPixelGainForHLTonGPU>(ctx);
                   *gainForHLTonHost = gain;
                   return gainForHLTonHost;
                 }).forEachDevice([&](auto const& gainForHLTonHost, cms::cuda::ESContext& ctx) {
    auto gainForHLTonGPU = cms::cuda::make_device_unique_uninitialized<SiPixelGainForHLTonGPU>(ctx);
    auto gainDataOnGPU = cms::cuda::make_device_unique<char[]>(gainData.size(), ctx);
    cudaCheck(cudaMemcpyAsync(gainDataOnGPU.get(), gainData.data(), gainData.size(), cudaMemcpyDefault, ctx.stream()));
    cudaCheck(cudaMemcpyAsync(
        gainForHLTonGPU.get(), gainForHLTonHost.get(), sizeof(SiPixelGainForHLTonGPU), cudaMemcpyDefault, ctx.stream()));
    auto ptr = gainDataOnGPU.get();
    cudaCheck(cudaMemcpyAsync(&(gainForHLTonGPU->v_pedestals_),
                              &ptr,
                              sizeof(SiPixelGainForHLTonGPU_DecodingStructure*),
                              cudaMemcpyDefault,
                              ctx.stream()));
    return SiPixelGainCalibrationForHLTGPU(std::move(gainForHLTonGPU), std::move(gainDataOnGPU));
  }));
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelGainCalibrationForHLTGPUESProducer);
