
#ifndef Profile_h
#define Profile_h
#include <string>

namespace edm {
class Event;
};

void dumpEventRecord();
void beginProduce(const std::string& name, const edm::Event& event);
void endProduce(const std::string& name, const edm::Event& event);
void beginAcquire(const std::string& name, const edm::Event& event);
void endAcquire(const std::string& name, const edm::Event& event);
#endif
