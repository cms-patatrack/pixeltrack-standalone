#ifndef DataFormats_FEDRawDataCollection_h
#define DataFormats_FEDRawDataCollection_h

/** \class FEDRawDataCollection
 *  An EDCollection storing the raw data for all  FEDs in a Event.
 *  
 *  Reference: DaqPrototype/DaqPersistentData/interface/DaqFEDOpaqueData.h
 *
 *  \author N. Amapane - S. Argiro'
 */

#include "DataFormats/FEDRawData.h"

#include <vector>

class FEDRawDataCollection {
public:
  FEDRawDataCollection();

  virtual ~FEDRawDataCollection();

  /// retrieve data for fed @param fedid
  const FEDRawData& FEDData(int fedid) const;

  /// retrieve data for fed @param fedid
  FEDRawData& FEDData(int fedid);

  FEDRawDataCollection(const FEDRawDataCollection&);

  void swap(FEDRawDataCollection& other) { data_.swap(other.data_); }

private:
  std::vector<FEDRawData> data_;  ///< the raw data
};

inline void swap(FEDRawDataCollection& a, FEDRawDataCollection& b) { a.swap(b); }

#endif  // DataFormats_FEDRawDataCollection_h
