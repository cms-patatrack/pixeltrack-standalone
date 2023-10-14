#ifndef CondFormats_SiPixelFedIds_h
#define CondFormats_SiPixelFedIds_h

#include <vector>

// Stripped-down version of SiPixelFedCablingMap
class SiPixelFedIds {
public:
  explicit SiPixelFedIds(std::vector<unsigned int> fedIds) : fedIds_(std::move(fedIds)) {}

  std::vector<unsigned int> const& fedIds() const { return fedIds_; }

private:
  std::vector<unsigned int> fedIds_;
};

#endif
