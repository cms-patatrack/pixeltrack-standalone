#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelGainForHLTonGPU.h"
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
  eventSetup.put(std::make_unique<SiPixelGainCalibrationForHLTGPU>(gain, std::move(gainData)));
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelGainCalibrationForHLTGPUESProducer);
