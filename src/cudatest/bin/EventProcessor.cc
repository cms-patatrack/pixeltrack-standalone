#include "Framework/ESPluginFactory.h"
#include "Framework/WaitingTask.h"
#include "Framework/WaitingTaskHolder.h"

#include "EventProcessor.h"

namespace edm {
  EventProcessor::EventProcessor(int warmupEvents,
                                 int maxEvents,
                                 int runForMinutes,
                                 int numberOfStreams,
                                 std::vector<std::string> const& path,
                                 std::vector<std::string> const& esproducers,
                                 std::filesystem::path const& datadir,
                                 bool validation)
      : source_(maxEvents, runForMinutes, registry_, datadir, validation),
        warmupEvents_(warmupEvents),
        maxEvents_(source_.maxEvents()),
        runForMinutes_(runForMinutes) {
    for (auto const& name : esproducers) {
      pluginManager_.load(name);
      auto esp = ESPluginFactory::create(name, datadir);
      esp->produce(eventSetup_);
    }

    schedules_.reserve(numberOfStreams);
    for (int i = 0; i < numberOfStreams; ++i) {
      schedules_.emplace_back(registry_, pluginManager_, &source_, &eventSetup_, i, path);
    }
  }

  void EventProcessor::warmUp() {
    if (warmupEvents_ <= 0)
      return;

    // Configure the source for the warmup step
    source_.reconfigure(warmupEvents_, -1);
    process();
  }

  void EventProcessor::runToCompletion() {
    // Configure the source for the actual reconstrction
    source_.reconfigure(maxEvents_, runForMinutes_);
    process();
  }

  void EventProcessor::process() {
    source_.startProcessing();

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
  }

  void EventProcessor::endJob() {
    // Only on the first stream...
    schedules_[0].endJob();
  }
}  // namespace edm
