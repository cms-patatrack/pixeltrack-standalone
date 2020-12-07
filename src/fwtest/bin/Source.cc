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
      int batchEvents, int maxEvents, ProductRegistry &reg, std::filesystem::path const &datadir, bool validation)
      : batchEvents_(batchEvents),
        maxEvents_(maxEvents),
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

    if (maxEvents_ < 0) {
      maxEvents_ = raw_.size();
    }
  }

  EventBatch Source::produce(int streamId, ProductRegistry const &reg) {
    const int old = numEvents_.fetch_add(batchEvents_);
    const int size = std::min(batchEvents_, maxEvents_ - old);
    EventBatch events;
    if (size <= 0) {
      return events;
    }

    events.reserve(size);
    for (int i = 1; i <= size; ++i) {
      const int iev = old + i;
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
