//#include <iostream>

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
                                 Configuration const& configuration)
      : registry_(std::move(reg)), source_(source), eventSetup_(eventSetup), streamId_(streamId) {
    path_.reserve(configuration.path.size());
    int modInd = 1;
    for (auto const& label : configuration.path) {
      auto const& moduleConfig = configuration.module(label);
      auto const type = moduleConfig.type();
      pluginManager.load(type);
      // Products are keyed by the module's label, so the label -- not the
      // plugin type -- is what the registry records as this module's own.
      registry_.beginModuleConstruction(modInd, label);
      path_.emplace_back(PluginFactory::create(type, moduleConfig, registry_));
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
    auto task = make_functor_task([this, h]() mutable { processOneEventAsync(std::move(h)); });
    if (streamId_ == 0) {
      h.group()->run([task]() {
        TaskSentry s{task};
        task->execute();
      });
    } else {
      tbb::task_arena arena{tbb::task_arena::attach()};
      arena.enqueue([task]() {
        TaskSentry s{task};
        task->execute();
      });
    }
  }

  void StreamSchedule::processOneEventAsync(WaitingTaskHolder h) {
    auto event = source_->produce(streamId_, registry_);
    if (event) {
      // Pass the event object ownership to the "end-of-event" task
      // Pass a non-owning pointer to the event to preceding tasks
      //std::cout << "Begin processing event " << event->eventID() << std::endl;
      auto eventPtr = event.get();
      auto* group = h.group();
      auto nextEventTask =
          make_waiting_task([this, h = std::move(h), ev = std::move(event)](std::exception_ptr const* iPtr) mutable {
            ev.reset();
            if (iPtr) {
              h.doneWaiting(*iPtr);
            } else {
              for (auto const& worker : path_) {
                worker->reset();
              }
              processOneEventAsync(std::move(h));
            }
          });
      // To guarantee that the nextEventTask is spawned also in
      // absence of Workers, and also to prevent spawning it before
      // all workers have been processed (should not happen though)
      auto nextEventTaskHolder = WaitingTaskHolder(*group, nextEventTask);

      for (auto iWorker = path_.rbegin(); iWorker != path_.rend(); ++iWorker) {
        //std::cout << "calling doWorkAsync for " << iWorker->get() << " with nextEventTask " << nextEventTask << std::endl;
        (*iWorker)->doWorkAsync(*eventPtr, *eventSetup_, nextEventTaskHolder);
      }
    } else {
      h.doneWaiting(std::exception_ptr{});
    }
  }

  void StreamSchedule::endJob() {
    for (auto& w : path_) {
      w->doEndJob();
    }
  }
}  // namespace edm
