#ifndef Source_h
#define Source_h

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

#include "Framework/Event.h"
#include "DataFormats/FEDRawDataCollection.h"
#include "DataFormats/DigiClusterCount.h"
#include "DataFormats/TrackCount.h"
#include "DataFormats/VertexCount.h"

namespace edm {
  class Source {
  public:
    explicit Source(int maxEvents,
                    int runForMinutes,
                    ProductRegistry& reg,
                    std::filesystem::path const& datadir,
                    bool validation);

    void startProcessing();

    int maxEvents() const { return maxEvents_; }
    int processedEvents() const { return numEvents_; }

    // thread safe
    std::unique_ptr<Event> produce(int streamId, ProductRegistry const& reg);

  private:
    int maxEvents_;

    // these are all for the mode where the processing length is limited by time
    int const runForMinutes_;
    std::chrono::steady_clock::time_point startTime_;
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
  };
}  // namespace edm

#endif
