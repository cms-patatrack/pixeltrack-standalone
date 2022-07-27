#include <iostream>
#include <fstream>

#include "Geometry/phase1PixelTopology.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/StreamCache.h"
#include "CondFormats/PixelCPEFast.h"

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEFast::PixelCPEFast(std::string const &path) {
  unsigned int ndetParams;

  m_params = new pixelCPEforGPU::ParamsOnGPU;
  m_commonParams = new pixelCPEforGPU::CommonParams;
  m_layerGeometry = new pixelCPEforGPU::LayerGeometry;
  m_averageGeometry = new pixelCPEforGPU::AverageGeometry;
  {
    std::ifstream in(path, std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    in.read(reinterpret_cast<char *>(m_commonParams), sizeof(pixelCPEforGPU::CommonParams));
    in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
    m_detParams = new pixelCPEforGPU::DetParams[ndetParams];
    in.read(reinterpret_cast<char *>(m_detParams), ndetParams * sizeof(pixelCPEforGPU::DetParams));
    in.read(reinterpret_cast<char *>(m_averageGeometry), sizeof(pixelCPEforGPU::AverageGeometry));
    in.read(reinterpret_cast<char *>(m_layerGeometry), sizeof(pixelCPEforGPU::LayerGeometry));
  }

  m_params->m_commonParams = m_commonParams;
  m_params->m_detParams = m_detParams;
  m_params->m_layerGeometry = m_layerGeometry;
  m_params->m_averageGeometry = m_averageGeometry;
}

PixelCPEFast::~PixelCPEFast() {
  delete m_params;
  delete m_commonParams;
  delete[] m_detParams;
  delete m_layerGeometry;
  delete m_averageGeometry;
}
