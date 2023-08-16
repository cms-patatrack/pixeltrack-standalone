#ifndef SYCLDataFormats_BeamSpot_interface_BeamSpotSYCL_h
#define SYCLDataFormats_BeamSpot_interface_BeamSpotSYCL_h

#include <sycl/sycl.hpp>

#include "DataFormats/BeamSpotPOD.h"
#include "SYCLCore/device_unique_ptr.h"

class BeamSpotSYCL {
public:
  // default constructor, required by cms::sycltools::Product<BeamSpotSYCL>
  BeamSpotSYCL() = default;

  // constructor that allocates cached device memory on the given SYCL queue
  BeamSpotSYCL(sycl::queue stream) { data_d_ = cms::sycltools::make_device_unique<BeamSpotPOD>(stream); }

  // movable, non-copiable
  BeamSpotSYCL(BeamSpotSYCL const&) = delete;
  BeamSpotSYCL(BeamSpotSYCL&&) = default;
  BeamSpotSYCL& operator=(BeamSpotSYCL const&) = delete;
  BeamSpotSYCL& operator=(BeamSpotSYCL&&) = default;

  BeamSpotPOD* data() { return data_d_.get(); }
  BeamSpotPOD const* data() const { return data_d_.get(); }

  cms::sycltools::device::unique_ptr<BeamSpotPOD>& ptr() { return data_d_; }
  cms::sycltools::device::unique_ptr<BeamSpotPOD> const& ptr() const { return data_d_; }

private:
  cms::sycltools::device::unique_ptr<BeamSpotPOD> data_d_;
};

#endif  // SYCLDataFormats_BeamSpot_interface_BeamSpotSYCL_h
