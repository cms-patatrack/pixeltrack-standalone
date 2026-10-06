#ifndef DataFormats_Products_h
#define DataFormats_Products_h

#include "DataFormats/BeamSpotPOD.h"
#include "DataFormats/DigiClusterCount.h"
#include "DataFormats/TrackCount.h"
#include "DataFormats/VertexCount.h"
#include "DataFormats/FEDRawDataCollection.h"
#include "CondFormats/SiPixelFedIds.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"
#include "CondFormats/SiPixelGainCalibrationForHLTGPU.h"
#include "CUDADataFormats/SiPixelDigisSoA.h"
#include "CUDADataFormats/SiPixelClustersSoA.h"
#include "CUDADataFormats/TrackingRecHit2DCUDA.h"
#include "CondFormats/PixelCPEFast.h"
#include "CUDADataFormats/PixelTrackHeterogeneous.h"
#include "CUDADataFormats/ZVertexHeterogeneous.h"

// The header rootcling parses to build the dictionaries for this backend's
// products.  Which of the types declared here are products is said once, in
// REFLECT_PRODUCTS in the Makefile: that list drives the dictionary, the class
// bindings, the get/emplace dispatch and the name -> type_index table alike.

#endif
