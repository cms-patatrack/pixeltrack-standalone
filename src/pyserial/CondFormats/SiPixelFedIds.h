#ifndef CondFormats_SiPixelFedIds_h
#define CondFormats_SiPixelFedIds_h

#include <span>
#include <vector>

// Stripped-down version of SiPixelFedCablingMap
class SiPixelFedIds {
public:
  explicit SiPixelFedIds(std::vector<unsigned int> fedIds) : fedIds_(std::move(fedIds)) {}

  std::vector<unsigned int> const& fedIds() const { return fedIds_; }

  /// The same list as a column, for a module that wants it as an array.
  std::span<const unsigned int> fedIdsSpan() const { return fedIds_; }

private:
  std::vector<unsigned int> fedIds_;
};

#endif
