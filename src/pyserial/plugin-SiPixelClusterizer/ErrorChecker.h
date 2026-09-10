#ifndef ErrorChecker_H
#define ErrorChecker_H
/** \class ErrorChecker
 *
 *  
 */

#include <vector>
#include <map>

#include "DataFormats/SiPixelRawDataError.h"

class ErrorChecker {
public:
  typedef uint32_t Word32;
  typedef uint64_t Word64;

  typedef std::vector<SiPixelRawDataError> DetErrors;
  typedef std::map<uint32_t, DetErrors> Errors;

  ErrorChecker();

  bool checkCRC(bool& errorsInEvent, int fedId, const Word64* trailer, Errors& errors);

  bool checkHeader(bool& errorsInEvent, int fedId, const Word64* header, Errors& errors);

  bool checkTrailer(bool& errorsInEvent, int fedId, unsigned int nWords, const Word64* trailer, Errors& errors);

private:
  bool includeErrors;
};

#endif
