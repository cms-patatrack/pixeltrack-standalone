#include "ErrorChecker.h"

#include "DataFormats/FEDHeader.h"
#include "DataFormats/FEDTrailer.h"

#include <bitset>
#include <sstream>
#include <iostream>

namespace {
  constexpr int CRC_bits = 1;
  constexpr int LINK_bits = 6;
  constexpr int ROC_bits = 5;
  constexpr int DCOL_bits = 5;
  constexpr int PXID_bits = 8;
  constexpr int ADC_bits = 8;
  constexpr int OMIT_ERR_bits = 1;

  constexpr int CRC_shift = 2;
  constexpr int ADC_shift = 0;
  constexpr int PXID_shift = ADC_shift + ADC_bits;
  constexpr int DCOL_shift = PXID_shift + PXID_bits;
  constexpr int ROC_shift = DCOL_shift + DCOL_bits;
  constexpr int LINK_shift = ROC_shift + ROC_bits;
  constexpr int OMIT_ERR_shift = 20;

  constexpr uint32_t dummyDetId = 0xffffffff;

  constexpr ErrorChecker::Word64 CRC_mask = ~(~ErrorChecker::Word64(0) << CRC_bits);
  constexpr ErrorChecker::Word32 ERROR_mask = ~(~ErrorChecker::Word32(0) << ROC_bits);
  constexpr ErrorChecker::Word32 LINK_mask = ~(~ErrorChecker::Word32(0) << LINK_bits);
  constexpr ErrorChecker::Word32 ROC_mask = ~(~ErrorChecker::Word32(0) << ROC_bits);
  constexpr ErrorChecker::Word32 OMIT_ERR_mask = ~(~ErrorChecker::Word32(0) << OMIT_ERR_bits);
}  // namespace

ErrorChecker::ErrorChecker() { includeErrors = false; }

bool ErrorChecker::checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer, Errors& errors) {
  int CRC_BIT = (*trailer >> CRC_shift) & CRC_mask;
  if (CRC_BIT == 0)
    return true;
  errorsInEvent = true;
  if (includeErrors) {
    int errorType = 39;
    SiPixelRawDataError error(*trailer, errorType, fedId);
    errors[dummyDetId].push_back(error);
  }
  return false;
}

bool ErrorChecker::checkHeader(bool& errorsInEvent, int fedId, const Word64* header, Errors& errors) {
  FEDHeader fedHeader(reinterpret_cast<const unsigned char*>(header));
  if (!fedHeader.check())
    return false;  // throw exception?
  if (fedHeader.sourceID() != fedId) {
    std::cout << "PixelDataFormatter::interpretRawData, fedHeader.sourceID() != fedId"
              << ", sourceID = " << fedHeader.sourceID() << ", fedId = " << fedId << ", errorType = 32" << std::endl;
    errorsInEvent = true;
    if (includeErrors) {
      int errorType = 32;
      SiPixelRawDataError error(*header, errorType, fedId);
      errors[dummyDetId].push_back(error);
    }
  }
  return fedHeader.moreHeaders();
}

bool ErrorChecker::checkTrailer(
    bool& errorsInEvent, int fedId, unsigned int nWords, const Word64* trailer, Errors& errors) {
  FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer));
  if (!fedTrailer.check()) {
    if (includeErrors) {
      int errorType = 33;
      SiPixelRawDataError error(*trailer, errorType, fedId);
      errors[dummyDetId].push_back(error);
    }
    errorsInEvent = true;
    std::cout << "fedTrailer.check failed, Fed: " << fedId << ", errorType = 33" << std::endl;
    return false;
  }
  if (fedTrailer.fragmentLength() != nWords) {
    std::cout << "fedTrailer.fragmentLength()!= nWords !! Fed: " << fedId << ", errorType = 34" << std::endl;
    errorsInEvent = true;
    if (includeErrors) {
      int errorType = 34;
      SiPixelRawDataError error(*trailer, errorType, fedId);
      errors[dummyDetId].push_back(error);
    }
  }
  return fedTrailer.moreTrailers();
}
