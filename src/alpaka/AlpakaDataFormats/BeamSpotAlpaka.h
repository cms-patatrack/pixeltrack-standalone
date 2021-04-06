#ifndef AlpakaDataFormats_BeamSpot_interface_BeamSpotAlpaka_h
#define AlpakaDataFormats_BeamSpot_interface_BeamSpotAlpaka_h

#include "AlpakaCore/alpakaConfig.h"
#include "DataFormats/BeamSpotPOD.h"

#include <cstring>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BeamSpotAlpaka {
  public:
    BeamSpotAlpaka() = default;
  
  BeamSpotAlpaka(BeamSpotPOD const* data, AlpakaExecSpace& space)
    : data_d{alpaka::allocBuf<BeamSpotPOD, Idx>(space.device, 1u)} 
    {      
      ViewHost<const BeamSpotPOD> data_h(data, space.host, 1u);

      alpaka::memcpy(space.queue, data_d, data_h, 1u);
    }

    BeamSpotPOD const* data() const { return alpaka::getPtrNative(data_d); }

  private:
    AlpakaAccBuf1<BeamSpotPOD> data_d;
  };

}

#endif
