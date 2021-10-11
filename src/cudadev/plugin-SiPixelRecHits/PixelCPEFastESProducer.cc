#include "CondFormats/PixelCPEFast.h"
#include "CUDACore/ESContext.h"
#include "CUDACore/copyAsync.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"

#include <fstream>
#include <iostream>
#include <memory>

class PixelCPEFastESProducer : public edm::ESProducer {
public:
  explicit PixelCPEFastESProducer(std::filesystem::path const& datadir) : data_(datadir) {}
  void produce(edm::EventSetup& eventSetup);

private:
  std::filesystem::path data_;
};

void PixelCPEFastESProducer::produce(edm::EventSetup& eventSetup) {
  struct CPUProduct {
    std::vector<pixelCPEforGPU::DetParams> detParamsGPU;
    pixelCPEforGPU::CommonParams commonParamsGPU;
    pixelCPEforGPU::LayerGeometry layerGeometry;
    pixelCPEforGPU::AverageGeometry averageGeometry;
  };
  auto cpuProd = std::make_unique<CPUProduct>();

  std::ifstream in(data_ / "cpefast.bin", std::ios::binary);
  in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
  in.read(reinterpret_cast<char*>(&cpuProd->commonParamsGPU), sizeof(pixelCPEforGPU::CommonParams));
  unsigned int ndetParams;
  in.read(reinterpret_cast<char*>(&ndetParams), sizeof(unsigned int));
  cpuProd->detParamsGPU.resize(ndetParams);
  in.read(reinterpret_cast<char*>(cpuProd->detParamsGPU.data()), ndetParams * sizeof(pixelCPEforGPU::DetParams));
  in.read(reinterpret_cast<char*>(&cpuProd->averageGeometry), sizeof(pixelCPEforGPU::AverageGeometry));
  in.read(reinterpret_cast<char*>(&cpuProd->layerGeometry), sizeof(pixelCPEforGPU::LayerGeometry));

  eventSetup.put(cms::cuda::runForHost([&](cms::cuda::HostAllocatorContext& ctx) {
                   return std::move(cpuProd);
                 }).forEachDevice([](auto const& cpuProd, cms::cuda::ESContext& ctx) {
    auto commonParams = cms::cuda::make_device_unique<pixelCPEforGPU::CommonParams>(ctx);
    auto detParams =
        cms::cuda::make_device_unique_uninitialized<pixelCPEforGPU::DetParams[]>(cpuProd->detParamsGPU.size(), ctx);
    auto averageGeometry = cms::cuda::make_device_unique<pixelCPEforGPU::AverageGeometry>(ctx);
    auto layerGeometry = cms::cuda::make_device_unique<pixelCPEforGPU::LayerGeometry>(ctx);
    auto paramsOnGPU_d = cms::cuda::make_device_unique<pixelCPEforGPU::ParamsOnGPU>(ctx);

    auto paramsOnGPU_h = cms::cuda::make_host_unique<pixelCPEforGPU::ParamsOnGPU>(ctx);
    paramsOnGPU_h->m_commonParams = commonParams.get();
    paramsOnGPU_h->m_detParams = detParams.get();
    paramsOnGPU_h->m_layerGeometry = layerGeometry.get();
    paramsOnGPU_h->m_averageGeometry = averageGeometry.get();

    cms::cuda::copyAsync(paramsOnGPU_d, paramsOnGPU_h, ctx.stream());
    cudaCheck(cudaMemcpyAsync(commonParams.get(),
                              &cpuProd->commonParamsGPU,
                              sizeof(pixelCPEforGPU::CommonParams),
                              cudaMemcpyDefault,
                              ctx.stream()));
    cudaCheck(cudaMemcpyAsync(averageGeometry.get(),
                              &cpuProd->averageGeometry,
                              sizeof(pixelCPEforGPU::AverageGeometry),
                              cudaMemcpyDefault,
                              ctx.stream()));
    cudaCheck(cudaMemcpyAsync(layerGeometry.get(),
                              &cpuProd->layerGeometry,
                              sizeof(pixelCPEforGPU::LayerGeometry),
                              cudaMemcpyDefault,
                              ctx.stream()));
    cudaCheck(cudaMemcpyAsync(detParams.get(),
                              cpuProd->detParamsGPU.data(),
                              cpuProd->detParamsGPU.size() * sizeof(pixelCPEforGPU::DetParams),
                              cudaMemcpyDefault,
                              ctx.stream()));
    return PixelCPEFast(std::move(paramsOnGPU_d),
                        std::move(commonParams),
                        std::move(detParams),
                        std::move(layerGeometry),
                        std::move(averageGeometry));
    ;
  }));
}

DEFINE_FWK_EVENTSETUP_MODULE(PixelCPEFastESProducer);
