#include <algorithm>
#include <iostream>

#include <tbb/task.h>

#include "Framework/FunctorTask.h"
#include "Framework/PluginFactory.h"
#include "Framework/WaitingTask.h"
#include "Framework/Worker.h"

#include "PluginManager.h"
#include "Source.h"
#include "StreamSchedule.h"

namespace edm {
  StreamSchedule::StreamSchedule(ProductRegistry reg,
                                 edmplugin::PluginManager& pluginManager,
                                 Source* source,
                                 EventSetup const* eventSetup,
                                 int streamId,
                                 std::vector<std::string> const& path)
      : registry_(std::move(reg)), source_(source), eventSetup_(eventSetup), streamId_(streamId) {
    path_.reserve(path.size());
    int modInd = 1;
    for (auto const& name : path) {
      pluginManager.load(name);
      registry_.beginModuleConstruction(modInd);
      path_.emplace_back(PluginFactory::create(name, registry_));
      //std::cout << "module " << modInd << " " << path_.back().get() << std::endl;
      std::vector<Worker*> consumes;
      for (unsigned int depInd : registry_.consumedModules()) {
        if (depInd != ProductRegistry::kSourceIndex) {
          //std::cout << "module " << modInd << " depends on " << (depInd-1) << " " << path_[depInd-1].get() << std::endl;
          consumes.push_back(path_[depInd - 1].get());
        }
      }
      path_.back()->setItemsToGet(std::move(consumes));
      ++modInd;
    }
  }

  StreamSchedule::~StreamSchedule() = default;
  StreamSchedule::StreamSchedule(StreamSchedule&&) = default;
  StreamSchedule& StreamSchedule::operator=(StreamSchedule&&) = default;

  void StreamSchedule::runToCompletionAsync(WaitingTaskHolder h) {
    auto task =
        make_functor_task(tbb::task::allocate_root(), [this, h]() mutable { processEventBatchAsync(std::move(h)); });
    if (streamId_ == 0) {
      tbb::task::spawn(*task);
    } else {
      tbb::task::enqueue(*task);
    }
  }

  void StreamSchedule::processEventBatchAsync(WaitingTaskHolder h) {
    auto events = source_->produce(streamId_, registry_);
    if (not events.empty()) {
      // Pass the event batch ownership to the "end-of-event" task
      // Pass non-owning pointers to the event to preceding tasks
      //std::cout << "Begin processing a batch of " << events.size() << " events starting from " << events.front().eventID() << std::endl;
      auto eventsPtr = std::vector<Event*>(events.size(), nullptr);
      std::transform(events.begin(), events.end(), eventsPtr.begin(), [](auto& event) { return &event; });
      auto nextEventTask = make_waiting_task(
          tbb::task::allocate_root(),
          [this, h = std::move(h), events = std::move(events)](std::exception_ptr const* iPtr) mutable {
            events.clear();
            if (iPtr) {
              h.doneWaiting(*iPtr);
            } else {
              for (auto const& worker : path_) {
                worker->reset();
              }
              processEventBatchAsync(std::move(h));
            }
          });
      // To guarantee that the nextEventTask is spawned also in
      // absence of Workers, and also to prevent spawning it before
      // all workers have been processed (should not happen though)
      auto nextEventTaskHolder = WaitingTaskHolder(nextEventTask);

      for (auto iWorker = path_.rbegin(); iWorker != path_.rend(); ++iWorker) {
        //std::cout << "calling doWorkAsync for " << iWorker->get() << " with nextEventTask " << nextEventTask << std::endl;
        (*iWorker)->doWorkAsync(eventsPtr, *eventSetup_, nextEventTask);
      }
    } else {
      h.doneWaiting();
    }
  }

  void StreamSchedule::endJob() {
    for (auto& w : path_) {
      w->doEndJob();
    }
  }
}  // namespace edm
