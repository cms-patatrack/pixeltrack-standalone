/** \file
 *  implementation of DaqRawDataCollection
 *
 *  \author N. Amapane - S. Argiro'
 */

#include "DataFormats/FEDRawDataCollection.h"
#include "DataFormats/FEDNumbering.h"

FEDRawDataCollection::FEDRawDataCollection() : data_(FEDNumbering::lastFEDId() + 1) {}

FEDRawDataCollection::FEDRawDataCollection(const FEDRawDataCollection& in) : data_(in.data_) {}
FEDRawDataCollection::~FEDRawDataCollection() {}

const FEDRawData& FEDRawDataCollection::FEDData(int fedid) const { return data_[fedid]; }

FEDRawData& FEDRawDataCollection::FEDData(int fedid) { return data_[fedid]; }
