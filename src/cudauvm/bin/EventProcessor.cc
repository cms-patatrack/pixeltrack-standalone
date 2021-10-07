#include "Framework/EmptyWaitingTask.h"
#include "Framework/ESPluginFactory.h"
#include "Framework/WaitingTask.h"
#include "Framework/WaitingTaskHolder.h"

#include "EventProcessor.h"

namespace edm {
  EventProcessor::EventProcessor(int maxEvents,
                                 int runForMinutes,
                                 int numberOfStreams,
                                 std::vector<std::string> const& path,
                                 std::vector<std::string> const& esproducers,
                                 std::filesystem::path const& datadir,
                                 bool validation)
      : source_(maxEvents, runForMinutes, registry_, datadir, validation) {
    for (auto const& name : esproducers) {
      pluginManager_.load(name);
      auto esp = ESPluginFactory::create(name, datadir);
      esp->produce(eventSetup_);
    }

    //schedules_.reserve(numberOfStreams);
    for (int i = 0; i < numberOfStreams; ++i) {
      schedules_.emplace_back(registry_, pluginManager_, &source_, &eventSetup_, i, path);
    }
  }

  void EventProcessor::runToCompletion() {
    source_.startProcessing();
    // The task that waits for all other work
    auto globalWaitTask = make_empty_waiting_task();
    globalWaitTask->increment_ref_count();
    for (auto& s : schedules_) {
      s.runToCompletionAsync(WaitingTaskHolder(globalWaitTask.get()));
    }
    globalWaitTask->wait_for_all();
    if (globalWaitTask->exceptionPtr()) {
      std::rethrow_exception(*(globalWaitTask->exceptionPtr()));
    }
  }

  void EventProcessor::endJob() {
    // Only on the first stream...
    schedules_[0].endJob();
  }
}  // namespace edm
