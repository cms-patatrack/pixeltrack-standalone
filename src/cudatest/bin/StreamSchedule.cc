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
        make_functor_task(tbb::task::allocate_root(), [this, h]() mutable { processOneEventAsync(std::move(h)); });
    if (streamId_ == 0) {
      tbb::task::spawn(*task);
    } else {
      tbb::task::enqueue(*task);
    }
  }

  void StreamSchedule::processOneEventAsync(WaitingTaskHolder h) {
    auto event = source_->produce(streamId_, registry_);
    if (event) {
      // Pass the event object ownership to the "end-of-event" task
      // Pass a non-owning pointer to the event to preceding tasks
      auto eventPtr = event.get();
      auto nextEventTask =
          make_waiting_task(tbb::task::allocate_root(),
                            [this, h = std::move(h), ev = std::move(event)](std::exception_ptr const* iPtr) mutable {
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

      for (auto iWorker = path_.rbegin(); iWorker != path_.rend(); ++iWorker) {
        //std::cout << "calling doWorkAsync for " << iWorker->get() << " with nextEventTask " << nextEventTask << std::endl;
        (*iWorker)->doWorkAsync(*eventPtr, *eventSetup_, nextEventTask);
      }
    } else {
      h.doneWaiting(std::exception_ptr{});
    }
  }
}  // namespace edm
