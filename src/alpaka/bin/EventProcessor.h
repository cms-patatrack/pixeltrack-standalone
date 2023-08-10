#ifndef bin_EventProcessor_h
#define bin_EventProcessor_h

#include <filesystem>
#include <string>
#include <vector>

#include "AlpakaCore/backend.h"
#include "Framework/EventSetup.h"

#include "PluginManager.h"
#include "StreamSchedule.h"
#include "Source.h"
#include "Timestamp.h"

namespace edm {
  struct Alternative {
    Alternative() = default;
    Alternative(Backend backend, float weight, std::vector<std::string> path)
        : backend{backend}, weight{weight}, path{std::move(path)} {}

    Backend backend;
    float weight;
    std::vector<std::string> path;
  };

  using Alternatives = std::vector<Alternative>;

  class EventProcessor {
  public:
    explicit EventProcessor(int warmupEvents,
                            int maxEvents,
                            int runForMinutes,
                            int numberOfStreams,
                            Alternatives alternatives,
                            std::vector<std::string> const& esproducers,
                            std::filesystem::path const& datadir,
                            bool validation);

    int maxEvents() const { return source_.maxEvents(); }
    int processedEvents() const { return source_.processedEvents(); }
    std::vector<std::pair<Backend, int>> const& backends() const { return streamsPerBackend_; }
    Timestamp const& start() const { return source_.start(); }
    Timestamp const& stop() const { return stop_; }

    void runToCompletion();

    void endJob();

  private:
    edmplugin::PluginManager pluginManager_;
    ProductRegistry registry_;
    Timestamp stop_;
    Source source_;
    EventSetup eventSetup_;
    std::vector<StreamSchedule> schedules_;
    std::vector<std::pair<Backend, int>> streamsPerBackend_;
  };
}  // namespace edm

#endif  // bin_EventProcessor_h
