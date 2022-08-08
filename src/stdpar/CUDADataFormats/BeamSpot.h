#ifndef CUDADataFormats_BeamSpot_interface_BeamSpot_h
#define CUDADataFormats_BeamSpot_interface_BeamSpot_h

#include <memory>

#include "DataFormats/BeamSpotPOD.h"

class BeamSpot {
public:
  // default constructor, required by cms::cuda::Product<BeamSpot>
  BeamSpot() { data_d_ = std::make_unique<BeamSpotPOD>(); }

  // movable, non-copiable
  BeamSpot(BeamSpot const&) = delete;
  BeamSpot(BeamSpot&&) = default;
  BeamSpot& operator=(BeamSpot const&) = delete;
  BeamSpot& operator=(BeamSpot&&) = default;

  BeamSpotPOD* data() { return data_d_.get(); }
  BeamSpotPOD const* data() const { return data_d_.get(); }

  std::unique_ptr<BeamSpotPOD>& ptr() { return data_d_; }
  std::unique_ptr<BeamSpotPOD> const& ptr() const { return data_d_; }

private:
  std::unique_ptr<BeamSpotPOD> data_d_;
};

#endif
