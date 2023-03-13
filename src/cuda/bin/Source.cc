#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "Source.h"

namespace {
  FEDRawDataCollection readRaw(std::ifstream &is, unsigned int nfeds) {
    FEDRawDataCollection rawCollection;
    for (unsigned int ifed = 0; ifed < nfeds; ++ifed) {
      unsigned int fedId;
      is.read(reinterpret_cast<char *>(&fedId), sizeof(unsigned int));
      unsigned int fedSize;
      is.read(reinterpret_cast<char *>(&fedSize), sizeof(unsigned int));
      FEDRawData &rawData = rawCollection.FEDData(fedId);
      rawData.resize(fedSize);
      is.read(reinterpret_cast<char *>(rawData.data()), fedSize);
    }
    return rawCollection;
  }

}  // namespace

namespace edm {
  Source::Source(
      int batchEvents, int maxEvents, int runForMinutes, ProductRegistry &reg, std::filesystem::path const &datadir, bool validation)
      : batchEvents_(batchEvents),
        maxEvents_(maxEvents),
        runForMinutes_(runForMinutes),
        numEvents_(0),
        rawToken_(reg.produces<FEDRawDataCollection>()),
        validation_(validation) {
    std::ifstream in_raw(datadir / "raw.bin", std::ios::binary);
    std::ifstream in_digiclusters;
    std::ifstream in_tracks;
    std::ifstream in_vertices;

    if (validation_) {
      digiClusterToken_ = reg.produces<DigiClusterCount>();
      trackToken_ = reg.produces<TrackCount>();
      vertexToken_ = reg.produces<VertexCount>();

      in_digiclusters = std::ifstream(datadir / "digicluster.bin", std::ios::binary);
      in_tracks = std::ifstream(datadir / "tracks.bin", std::ios::binary);
      in_vertices = std::ifstream(datadir / "vertices.bin", std::ios::binary);
      in_digiclusters.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
      in_tracks.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
      in_vertices.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
    }

    unsigned int nfeds;
    in_raw.exceptions(std::ifstream::badbit);
    in_raw.read(reinterpret_cast<char *>(&nfeds), sizeof(unsigned int));
    while (not in_raw.eof()) {
      in_raw.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);

      raw_.emplace_back(readRaw(in_raw, nfeds));

      if (validation_) {
        unsigned int nm, nd, nc, nt, nv;
        in_digiclusters.read(reinterpret_cast<char *>(&nm), sizeof(unsigned int));
        in_digiclusters.read(reinterpret_cast<char *>(&nd), sizeof(unsigned int));
        in_digiclusters.read(reinterpret_cast<char *>(&nc), sizeof(unsigned int));
        in_tracks.read(reinterpret_cast<char *>(&nt), sizeof(unsigned int));
        in_vertices.read(reinterpret_cast<char *>(&nv), sizeof(unsigned int));
        digiclusters_.emplace_back(nm, nd, nc);
        tracks_.emplace_back(nt);
        vertices_.emplace_back(nv);
      }

      // next event
      in_raw.exceptions(std::ifstream::badbit);
      in_raw.read(reinterpret_cast<char *>(&nfeds), sizeof(unsigned int));
    }

    if (validation_) {
      assert(raw_.size() == digiclusters_.size());
      assert(raw_.size() == tracks_.size());
      assert(raw_.size() == vertices_.size());
    }

    if (batchEvents_ < 1) {
      batchEvents_ = 1;
    }

    if (runForMinutes_ < 0 and maxEvents_ < 0) {
      maxEvents_ = raw_.size();
    }
  }

  void Source::startProcessing() {
    if (runForMinutes_ >= 0) {
      startTime_ = std::chrono::steady_clock::now();
    }
  }

  EventBatch Source::produce(int streamId, ProductRegistry const &reg) {
    if (shouldStop_) {
      return {};
    }

    // atomically increase the event counter, without overflowing over maxEvents_
    int old_value, new_value;

    if (runForMinutes_ < 0) {
      // atomically increase the event counter, without overflowing over maxEvents_
      old_value = numEvents_;
      do {
        new_value = std::min(old_value + batchEvents_, maxEvents_);
      }
      while (not numEvents_.compare_exchange_weak(old_value, new_value));
      if (old_value >= maxEvents_) {
        shouldStop_ = true;
        return {};
      }
    } else {
      // atomically increase the event counter, and periodically check if runForMinutes_ have passed
      old_value = numEvents_.fetch_add(batchEvents_);
      new_value = old_value + batchEvents_;
      if (numEvents_ - numEventsTimeLastCheck_ > static_cast<int>(raw_.size())) {
        std::scoped_lock lock(timeMutex_);
        // if some other thread beat us, no need to do anything
        if (numEvents_ - numEventsTimeLastCheck_ > static_cast<int>(raw_.size())) {
          auto processingTime = std::chrono::steady_clock::now() - startTime_;
          if (std::chrono::duration_cast<std::chrono::minutes>(processingTime).count() >= runForMinutes_) {
            shouldStop_ = true;
          }
          numEventsTimeLastCheck_ = (numEvents_ / raw_.size()) * raw_.size();
        }
        if (shouldStop_) {
          numEvents_ -= batchEvents_;
          return {};
        }
      }
    }

    // check how many events should be read
    const int size = new_value - old_value;
    EventBatch events;
    events.reserve(size);
    for (int iev = old_value + 1; iev <= new_value; ++iev) {
      Event &event = events.emplace(streamId, iev, reg);
      const int index = (iev - 1) % raw_.size();

      event.emplace(rawToken_, raw_[index]);
      if (validation_) {
        event.emplace(digiClusterToken_, digiclusters_[index]);
        event.emplace(trackToken_, tracks_[index]);
        event.emplace(vertexToken_, vertices_[index]);
      }
    }

    return events;
  }
}  // namespace edm
