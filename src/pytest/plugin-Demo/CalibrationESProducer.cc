// Publishes the calibration in the EventSetup.
//
// The conditions a Python module reads by type with no token, as pyserial's
// modules read the beam spot and the pixel CPE parameters.  IntESProducer
// covers the other EventSetup case, a fundamental type, which has no class to
// reflect and crosses through nanobind's own caster.

#include <memory>

#include "DataFormats/Products.h"
#include "Framework/ESPluginFactory.h"
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"

class CalibrationESProducer : public edm::ESProducer {
public:
  CalibrationESProducer(std::filesystem::path const& datadir) {}

private:
  void produce(edm::EventSetup& eventSetup) { eventSetup.put(std::make_unique<pytest::Calibration>()); }
};

DEFINE_FWK_EVENTSETUP_MODULE(CalibrationESProducer);
