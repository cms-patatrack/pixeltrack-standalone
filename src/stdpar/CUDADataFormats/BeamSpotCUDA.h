#ifndef CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h
#define CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h

#include <cuda_runtime.h>

#include "DataFormats/BeamSpotPOD.h"
#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
#include "CUDACore/device_unique_ptr.h"
#else
#include "CUDACore/managed_unique_ptr.h"
#endif

class BeamSpotCUDA {
public:
  // default constructor, required by cms::cuda::Product<BeamSpotCUDA>
  BeamSpotCUDA() = default;

  // constructor that allocates cached device memory on the given CUDA stream
#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
  BeamSpotCUDA(cudaStream_t stream) { data_d_ = cms::cuda::make_device_unique<BeamSpotPOD>(stream); }
#else
  BeamSpotCUDA(cudaStream_t stream) { data_d_ = cms::cuda::make_managed_unique<BeamSpotPOD>(stream); }
#endif

  ~BeamSpotCUDA();

  // movable, non-copiable
  BeamSpotCUDA(BeamSpotCUDA const&) = delete;
  BeamSpotCUDA(BeamSpotCUDA&&) = default;
  BeamSpotCUDA& operator=(BeamSpotCUDA const&) = delete;
  BeamSpotCUDA& operator=(BeamSpotCUDA&&) = default;

  BeamSpotPOD* data() { return data_d_.get(); }
  BeamSpotPOD const* data() const { return data_d_.get(); }

#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
  cms::cuda::device::unique_ptr<BeamSpotPOD>& ptr() { return data_d_; }
  cms::cuda::device::unique_ptr<BeamSpotPOD> const& ptr() const { return data_d_; }
#else
  cms::cuda::managed::unique_ptr<BeamSpotPOD>& ptr() { return data_d_; }
  cms::cuda::managed::unique_ptr<BeamSpotPOD> const& ptr() const { return data_d_; }

  void memAdviseAndPrefetch(int device, cudaStream_t stream);
#endif

private:
#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
  cms::cuda::device::unique_ptr<BeamSpotPOD> data_d_;
#else
  cms::cuda::managed::unique_ptr<BeamSpotPOD> data_d_;
  int device_ = -1;
#endif
};

#endif
