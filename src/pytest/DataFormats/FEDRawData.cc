/** \file
   implementation of class FedRawData

   \author Stefano ARGIRO
   \date 28 Jun 2005
*/

#include "DataFormats/FEDRawData.h"
#include <iostream>

using namespace std;

FEDRawData::FEDRawData() {}

FEDRawData::FEDRawData(size_t newsize) : data_(newsize) {
  if (newsize % 8 != 0)
    throw std::runtime_error("FEDRawData::resize: " + std::to_string(newsize) + " is not a multiple of 8 bytes.");
}

FEDRawData::FEDRawData(const FEDRawData &in) : data_(in.data_) {}
FEDRawData::~FEDRawData() {}
const unsigned char *FEDRawData::data() const { return data_.data(); }

unsigned char *FEDRawData::data() { return data_.data(); }

void FEDRawData::resize(size_t newsize) {
  if (size() == newsize)
    return;

  data_.resize(newsize);

  if (newsize % 8 != 0)
    throw std::runtime_error("FEDRawData::resize: " + std::to_string(newsize) + " is not a multiple of 8 bytes.");
}
