#ifndef AlpakaDataFormats_BeamSpot_interface_BeamSpotAlpaka_h
#define AlpakaDataFormats_BeamSpot_interface_BeamSpotAlpaka_h

#include "AlpakaCore/alpakaCommon.h"
#include "DataFormats/BeamSpotPOD.h"

#include <cstring>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BeamSpotAlpaka {
  public:
    BeamSpotAlpaka() = default;

    BeamSpotAlpaka(BeamSpotPOD const* data, Queue& queue) : data_d{cms::alpakatools::allocDeviceBuf<BeamSpotPOD>(1u)} {
      auto data_h{cms::alpakatools::createHostView<const BeamSpotPOD>(data, 1u)};

      alpaka::memcpy(queue, data_d, data_h, 1u);
      alpaka::wait(queue);
    }

    const BeamSpotPOD* data() const { return alpaka::getPtrNative(data_d); }

  private:
    AlpakaDeviceBuf<BeamSpotPOD> data_d;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
