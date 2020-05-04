#ifndef Source_h
#define Source_h

#include <atomic>
#include <filesystem>
#include <string>
#include <memory>

#include "Framework/Event.h"
#include "DataFormats/FEDRawDataCollection.h"
#include "DataFormats/DigiClusterCount.h"
#include "DataFormats/TrackCount.h"
#include "DataFormats/VertexCount.h"

namespace edm {
  class Source {
  public:
    explicit Source(int maxEvents, ProductRegistry& reg, std::filesystem::path const& datadir, bool validation);

    int maxEvents() const { return maxEvents_; }

    // thread safe
    std::unique_ptr<Event> produce(int streamId, ProductRegistry const& reg);

  private:
    int maxEvents_;
    std::atomic<int> numEvents_;
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
