#ifndef DataFormats_BeamSpotPOD_h
#define DataFormats_BeamSpotPOD_h

struct BeamSpotPOD {
  float x, y, z;  // position
  // TODO: add covariance matrix

  float sigmaZ;
  float beamWidthX, beamWidthY;
  float dxdz, dydz;
  float emittanceX, emittanceY;
  float betaStar;
};

#endif
