#ifndef CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h
#define CUDADataFormats_BeamSpot_interface_BeamSpotCUDA_h

#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
#include "CUDACore/device_unique_ptr.h"
#else
#include "CUDACore/managed_unique_ptr.h"
#endif

#include <cuda_runtime.h>

#include <thread>

class BeamSpotCUDA {
public:
  // alignas(128) doesn't really make sense as there is only one
  // beamspot per event?
  struct Data {
    float x, y, z;  // position
    // TODO: add covariance matrix

    float sigmaZ;
    float beamWidthX, beamWidthY;
    float dxdz, dydz;
    float emittanceX, emittanceY;
    float betaStar;
  };

  BeamSpotCUDA() = default;
#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
  BeamSpotCUDA(Data const* data_h, cudaStream_t stream);
#else
  BeamSpotCUDA(Data const& data_h, int device, cudaStream_t stream);
#endif
  ~BeamSpotCUDA();

  Data const* data() const { return data_d_.get(); }

private:
#ifdef CUDAUVM_DISABLE_MANAGED_BEAMSPOT
  cms::cuda::device::unique_ptr<Data> data_d_;
#else
  cms::cuda::managed::unique_ptr<Data> data_d_;
  int device_;
#endif
};

#endif
