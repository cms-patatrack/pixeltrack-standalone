#ifndef AlpakaDataFormats_BeamSpot_interface_BeamSpotAlpaka_h
#define AlpakaDataFormats_BeamSpot_interface_BeamSpotAlpaka_h

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaCommon.h"
#include "DataFormats/BeamSpotPOD.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BeamSpotAlpaka {
  public:
    // default constructor, required by ::cms::alpakatools::Product<Queue, BeamSpotAlpaka>
    BeamSpotAlpaka() = default;

    // constructor that allocates cached device memory on the given queue
    BeamSpotAlpaka(Queue const& queue) : data_d_{::cms::alpakatools::allocDeviceBuf<BeamSpotPOD>(queue, 1u)} {}

    // movable, non-copiable
    BeamSpotAlpaka(BeamSpotAlpaka const&) = delete;
    BeamSpotAlpaka(BeamSpotAlpaka&&) = default;
    BeamSpotAlpaka& operator=(BeamSpotAlpaka const&) = delete;
    BeamSpotAlpaka& operator=(BeamSpotAlpaka&&) = default;

    BeamSpotPOD* data() { return alpaka::getPtrNative(data_d_); }
    BeamSpotPOD const* data() const { return alpaka::getPtrNative(data_d_); }

    AlpakaDeviceBuf<BeamSpotPOD>& buf() { return data_d_; }
    AlpakaDeviceBuf<BeamSpotPOD> const& buf() const { return data_d_; }

  private:
    AlpakaDeviceBuf<BeamSpotPOD> data_d_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // AlpakaDataFormats_BeamSpot_interface_BeamSpotAlpaka_h
