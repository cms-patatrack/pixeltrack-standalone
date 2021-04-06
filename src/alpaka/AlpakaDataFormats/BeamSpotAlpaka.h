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
    : data_d{alpaka::allocBuf<BeamSpotPOD, Idx>(space.device, Vec1::all(1))} 
    {      
      ViewHost<const BeamSpotPOD> data_h(data, space.host, Vec1::all(1));

      alpaka::memcpy(space.queue, data_d, data_h, Vec1::all(1));
    }

    BeamSpotPOD const* data() const { return alpaka::getPtrNative(data_d); }

  private:
    ALPAKA_ACCELERATOR_NAMESPACE::AlpakaAccBuf1<BeamSpotPOD> data_d;
  };

}

#endif
