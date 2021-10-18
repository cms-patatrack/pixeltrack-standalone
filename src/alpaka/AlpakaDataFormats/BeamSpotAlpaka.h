#ifndef AlpakaDataFormats_BeamSpot_interface_BeamSpotAlpaka_h
#define AlpakaDataFormats_BeamSpot_interface_BeamSpotAlpaka_h

#include "AlpakaCore/device_unique_ptr.h"
#include "DataFormats/BeamSpotPOD.h"

#include <cstring>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BeamSpotAlpaka {
  public:
    BeamSpotAlpaka() = default;

    BeamSpotAlpaka(BeamSpotPOD const* data, Queue& queue)
        : data_d{cms::alpakatools::make_device_unique<BeamSpotPOD>(1u)} {
      auto data_h{cms::alpakatools::createHostView<const BeamSpotPOD>(data, 1u)};
      auto data_d_view{cms::alpakatools::createDeviceView<BeamSpotPOD>(data_d.get(), 1u)};

      alpaka::memcpy(queue, data_d_view, data_h, 1u);
      alpaka::wait(queue);
    }

    const BeamSpotPOD* data() const { return data_d.get(); }

  private:
    cms::alpakatools::device::unique_ptr<BeamSpotPOD> data_d;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
