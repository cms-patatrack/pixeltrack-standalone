
#ifndef Profile_h
#define Profile_h

#include <string>

// Functions that get called before and after the produce and acquire functions
// get called for each module.
// Intended to be a place to put profiling, instrumentation points, and/or task annotations.
//
//  The 'name' parameter is the name of the module being called.

namespace edm {
  class Event;
};

void beginProduce(const std::string& name, const edm::Event& event);

void endProduce(const std::string& name, const edm::Event& event);

void beginAcquire(const std::string& name, const edm::Event& event);

void endAcquire(const std::string& name, const edm::Event& event);

#endif
