#include <iostream>
#include <fstream>
#include <memory>
#include <cstddef>

#include "CondFormats/PixelCPEFast.h"

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEFast::PixelCPEFast(std::string const &path)
    : m_params{std::make_unique<pixelCPEforGPU::ParamsOnGPU>()},
      m_commonParams{std::make_unique<pixelCPEforGPU::CommonParams>()},
      m_layerGeometry{std::make_unique<pixelCPEforGPU::LayerGeometry>()},
      m_averageGeometry{std::make_unique<pixelCPEforGPU::AverageGeometry>()} {
  unsigned int ndetParams;
  {
    std::ifstream in(path, std::ios::binary);
    in.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    in.read(reinterpret_cast<char *>(m_commonParams.get()), sizeof(pixelCPEforGPU::CommonParams));
    in.read(reinterpret_cast<char *>(&ndetParams), sizeof(unsigned int));
    m_detParams = std::make_unique<pixelCPEforGPU::DetParams[]>(ndetParams);
    in.read(reinterpret_cast<char *>(m_detParams.get()), ndetParams * sizeof(pixelCPEforGPU::DetParams));
    in.read(reinterpret_cast<char *>(m_averageGeometry.get()), sizeof(pixelCPEforGPU::AverageGeometry));
    in.read(reinterpret_cast<char *>(m_layerGeometry.get()), sizeof(pixelCPEforGPU::LayerGeometry));
  }

  m_params->m_commonParams = m_commonParams.get();
  m_params->m_detParams = m_detParams.get();
  m_params->m_layerGeometry = m_layerGeometry.get();
  m_params->m_averageGeometry = m_averageGeometry.get();
}

PixelCPEFast::~PixelCPEFast() {}
