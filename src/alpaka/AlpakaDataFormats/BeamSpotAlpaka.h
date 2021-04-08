#ifndef AlpakaDataFormats_BeamSpot_interface_BeamSpotAlpaka_h
#define AlpakaDataFormats_BeamSpot_interface_BeamSpotAlpaka_h

#include "AlpakaCore/alpakaCommon.h"
#include "DataFormats/BeamSpotPOD.h"

#include <cstring>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BeamSpotAlpaka {
  public:
    BeamSpotAlpaka() = default;
  
  BeamSpotAlpaka(BeamSpotPOD const* data, Queue& queue)
    : data_d{cms::alpakatools::allocDeviceBuf<BeamSpotPOD>(device)} 
    {      
      auto data_h{cms::alpakatools::createHostView<const BeamSpotPOD>(host, data)};

      cms::alpakatools::memcpy(queue, data_d, data_h);
    }

    const BeamSpotPOD* data() const { return alpaka::getPtrNative(data_d); }

  private:
    AlpakaDeviceBuf<BeamSpotPOD> data_d;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
