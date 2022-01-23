#ifndef AlpakaDataFormats_BeamSpotAlpaka_h
#define AlpakaDataFormats_BeamSpotAlpaka_h

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"
#include "DataFormats/BeamSpotPOD.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BeamSpotAlpaka {
  public:
    // default constructor, required by cms::alpakatools::Product<Queue, BeamSpotAlpaka>
    BeamSpotAlpaka() = default;

    // constructor that allocates cached device memory on the given queue
    BeamSpotAlpaka(Queue const& queue) : data_d_{cms::alpakatools::make_device_buffer<BeamSpotPOD>(queue)} {}

    // movable, non-copiable
    BeamSpotAlpaka(BeamSpotAlpaka const&) = delete;
    BeamSpotAlpaka(BeamSpotAlpaka&&) = default;
    BeamSpotAlpaka& operator=(BeamSpotAlpaka const&) = delete;
    BeamSpotAlpaka& operator=(BeamSpotAlpaka&&) = default;

    BeamSpotPOD* data() { return data_d_.data(); }
    BeamSpotPOD const* data() const { return data_d_.data(); }

    cms::alpakatools::device_buffer<Device, BeamSpotPOD>& buf() { return data_d_; }
    cms::alpakatools::device_buffer<Device, BeamSpotPOD> const& buf() const { return data_d_; }

  private:
    cms::alpakatools::device_buffer<Device, BeamSpotPOD> data_d_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // AlpakaDataFormats_BeamSpotAlpaka_h
