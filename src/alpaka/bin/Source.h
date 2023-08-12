#ifndef bin_Source_h
#define bin_Source_h

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

#include "Framework/Event.h"
#include "DataFormats/FEDRawDataCollection.h"
#include "DataFormats/DigiClusterCount.h"
#include "DataFormats/TrackCount.h"
#include "DataFormats/VertexCount.h"
#include "Timestamp.h"

namespace edm {
  class Source {
  public:
    explicit Source(int warmupEvents,
                    int maxEvents,
                    int runForMinutes,
                    ProductRegistry& reg,
                    std::filesystem::path const& datadir,
                    bool validation);

    int maxEvents() const {
      return runForMinutes_ < 0 ? maxEvents_ - warmupEvents_ : -1; 
    }
    int processedEvents() const { return numEvents_ - warmupEvents_; }
    Timestamp const& start() const { return start_; }

    // thread safe
    std::unique_ptr<Event> produce(int streamId, ProductRegistry const& reg);

  private:
    int warmupEvents_;
    int maxEvents_;

    // these are all for the mode where the processing length is limited by time
    int const runForMinutes_;
    std::mutex timeMutex_;
    std::atomic<int> numEventsTimeLastCheck_ = 0;
    std::atomic<bool> shouldStop_ = false;

    std::atomic<int> numEvents_ = 0;
    EDPutTokenT<FEDRawDataCollection> const rawToken_;
    EDPutTokenT<DigiClusterCount> digiClusterToken_;
    EDPutTokenT<TrackCount> trackToken_;
    EDPutTokenT<VertexCount> vertexToken_;
    std::vector<FEDRawDataCollection> raw_;
    std::vector<DigiClusterCount> digiclusters_;
    std::vector<TrackCount> tracks_;
    std::vector<VertexCount> vertices_;
    bool const validation_;
    Timestamp start_;
  };
}  // namespace edm

#endif  // bin_Source_h
