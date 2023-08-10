#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <ios>
#include <memory>
#include <mutex>

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
  Source::Source(int warmupEvents,
                 int maxEvents,
                 int runForMinutes,
                 ProductRegistry &reg,
                 std::filesystem::path const &datadir,
                 bool validation)
      : warmupEvents_(warmupEvents),
        maxEvents_(maxEvents),
        runForMinutes_(runForMinutes),
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

    // default value for maxEvents_
    if (runForMinutes_ < 0 and maxEvents_ < 0) {
      maxEvents_ = raw_.size();
    }

    // run for warmupEvents before checking for the requested maxEvents
    if (runForMinutes_ < 0 and warmupEvents_ > 0) {
      maxEvents_ += warmupEvents_;
    }
  }

  std::unique_ptr<Event> Source::produce(int streamId, ProductRegistry const &reg) {
    if (shouldStop_) {
      return nullptr;
    }

    // old is the number of events that have been produced until now
    const int old = numEvents_.fetch_add(1);
    if (old == warmupEvents_) {
      // reset the time stamps after the warmup period
      start_.mark();
    }

    // check for the stop condition
    if (runForMinutes_ < 0) {
      // check based on the number of processed events
      if (old >= maxEvents_) {
        shouldStop_ = true;
        --numEvents_;
        return nullptr;
      }
    } else {
      // check based on the elapsed time
      // check only once per full processing of the input
      if (numEvents_ - numEventsTimeLastCheck_ > static_cast<int>(raw_.size())) {
        std::scoped_lock lock(timeMutex_);
        // check again, in case another thread beat us while acuiring the lock, and we don't need to do anything
        if (numEvents_ - numEventsTimeLastCheck_ > static_cast<int>(raw_.size())) {
          auto processingTime = std::chrono::steady_clock::now() - start_.time;
          if (std::chrono::duration_cast<std::chrono::minutes>(processingTime).count() >= runForMinutes_) {
            shouldStop_ = true;
          }
          numEventsTimeLastCheck_ = (numEvents_ / raw_.size()) * raw_.size();
        }
        if (shouldStop_) {
          --numEvents_;
          return nullptr;
        }
      }
    }

    // iev is the event that we are producing
    const int iev = old + 1;
    auto ev = std::make_unique<Event>(streamId, iev, reg);
    // use old rather than iev because events count from 1, but the buffer is 0-based
    const int index = old % raw_.size();

    ev->emplace(rawToken_, raw_[index]);
    if (validation_) {
      ev->emplace(digiClusterToken_, digiclusters_[index]);
      ev->emplace(trackToken_, tracks_[index]);
      ev->emplace(vertexToken_, vertices_[index]);
    }

    return ev;
  }
}  // namespace edm
