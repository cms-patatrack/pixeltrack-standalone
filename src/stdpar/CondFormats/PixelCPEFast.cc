#include <iostream>
#include <fstream>
#include <memory>
#include <cstddef>

#include "Geometry/phase1PixelTopology.h"
#include "CUDACore/cudaCheck.h"
#include "CUDACore/deviceCount.h"
#include "CUDACore/ScopedSetDevice.h"
#include "CUDACore/StreamCache.h"
#include "CondFormats/PixelCPEFast.h"

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEFast::PixelCPEFast(std::string const &path)
    : m_params{std::make_shared<pixelCPEforGPU::ParamsOnGPU>()},
      m_commonParams{std::make_shared<pixelCPEforGPU::CommonParams>()},
      m_layerGeometry{std::make_shared<pixelCPEforGPU::LayerGeometry>()},
      m_averageGeometry{std::make_shared<pixelCPEforGPU::AverageGeometry>()} {
  unsigned int ndetParams;
  {
    std::ifstream in(path, std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    in.read(reinterpret_cast<char *>(m_commonParams.get()), sizeof(pixelCPEforGPU::CommonParams));
    in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
#if __GNUC__ >= 12
    m_detParams = std::make_shared<pixelCPEforGPU::DetParams[]>(ndetParams);
#else
    m_detParams = std::shared_ptr<pixelCPEforGPU::DetParams[]>(new pixelCPEforGPU::DetParams[ndetParams]);
#endif
    in.read(reinterpret_cast<char *>(m_detParams.get()), ndetParams * sizeof(pixelCPEforGPU::DetParams));
    in.read(reinterpret_cast<char *>(m_averageGeometry.get()), sizeof(pixelCPEforGPU::AverageGeometry));
    in.read(reinterpret_cast<char *>(m_layerGeometry.get()), sizeof(pixelCPEforGPU::LayerGeometry));
  }

  m_params->m_commonParams = m_commonParams;
  m_params->m_detParams = m_detParams;
  m_params->m_layerGeometry = m_layerGeometry;
  m_params->m_averageGeometry = m_averageGeometry;
}

PixelCPEFast::~PixelCPEFast() {}
