#include <cmath>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "Framework/ESPluginFactory.h"
#include "Framework/WaitingTask.h"
#include "Framework/WaitingTaskHolder.h"

#include "EventProcessor.h"

namespace edm {
  EventProcessor::EventProcessor(int warmupEvents,
                                 int maxEvents,
                                 int runForMinutes,
                                 int numberOfStreams,
                                 Alternatives alternatives,
                                 std::vector<std::string> const& esproducers,
                                 std::filesystem::path const& datadir,
                                 bool validation)
      : source_(warmupEvents, maxEvents, runForMinutes, registry_, datadir, validation) {
    for (auto const& name : esproducers) {
      pluginManager_.load(name);
      auto esp = ESPluginFactory::create(name, datadir);
      esp->produce(eventSetup_);
    }

    // normalise the total weight to the number of streams
    float total = 0.;
    for (auto const& alternative : alternatives) {
      total += alternative.weight;
    }
    schedules_.reserve(numberOfStreams);
    float cumulative = 0.;
    int lower_range = 0;
    int upper_range = 0;
    for (auto& alternative : alternatives) {
      cumulative += alternative.weight;
      lower_range = upper_range;
      upper_range = static_cast<int>(std::round(cumulative * numberOfStreams / total));
      for (int i = lower_range; i < upper_range; ++i) {
        schedules_.emplace_back(registry_, pluginManager_, &source_, &eventSetup_, i, alternative.path);
      }
      streamsPerBackend_.emplace_back(alternative.backend, upper_range - lower_range);
    }
  }

  void EventProcessor::runToCompletion() {
    // The task that waits for all other work
    FinalWaitingTask globalWaitTask;
    tbb::task_group group;
    for (auto& s : schedules_) {
      s.runToCompletionAsync(WaitingTaskHolder(group, &globalWaitTask));
    }
    group.wait();
    assert(globalWaitTask.done());
    if (globalWaitTask.exceptionPtr()) {
      std::rethrow_exception(*(globalWaitTask.exceptionPtr()));
    }
    stop_.mark();
  }

  void EventProcessor::endJob() {
    // Only on the first stream...
    schedules_[0].endJob();
  }
}  // namespace edm
