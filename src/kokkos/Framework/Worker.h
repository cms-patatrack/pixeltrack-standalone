#ifndef Worker_h
#define Worker_h

#include <atomic>
#include <vector>
//#include <iostream>

#include "Framework/WaitingTask.h"
#include "Framework/WaitingTaskHolder.h"
#include "Framework/WaitingTaskList.h"
#include "Framework/WaitingTaskWithArenaHolder.h"
#include "Framework/Profile.h"

namespace edm {
  class Event;
  class EventSetup;
  class ProductRegistry;

  class Worker {
  public:
    Worker(const std::string& name) : name_(name), prefetchRequested_{false} {}
    virtual ~Worker() = default;

    // not thread safe
    void setItemsToGet(std::vector<Worker*> workers) { itemsToGet_ = std::move(workers); }

    // thread safe
    void prefetchAsync(Event& event, EventSetup const& eventSetup, WaitingTaskHolder iTask);

    // not thread safe
    virtual void doWorkAsync(Event& event, EventSetup const& eventSetup, WaitingTaskHolder iTask) = 0;

    // definitively not thread safe
    virtual void doWork(Event& event, EventSetup const& eventSetup) = 0;

    // not thread safe
    virtual void doEndJob() = 0;

    // not thread safe
    void reset() {
      prefetchRequested_ = false;
      doReset();
    }

  protected:
    virtual void doReset() = 0;
    std::string name_;

  private:
    std::vector<Worker*> itemsToGet_;
    std::atomic<bool> prefetchRequested_;
  };

  template <typename T>
  class WorkerT : public Worker {
  public:
    explicit WorkerT(ProductRegistry& reg, const std::string& name)
        : Worker(name), producer_(reg), workStarted_{false} {}

    void doWorkAsync(Event& event, EventSetup const& eventSetup, WaitingTaskHolder task) override {
      waitingTasksWork_.add(task);
      //std::cout << "doWorkAsync for " << this << " with iTask " << iTask << std::endl;
      bool expected = false;
      if (workStarted_.compare_exchange_strong(expected, true)) {
        //std::cout << "first doWorkAsync call" << std::endl;

        WaitingTask* moduleTask =
            make_waiting_task([this, &event, &eventSetup](std::exception_ptr const* iPtr) mutable {
              if (iPtr) {
                waitingTasksWork_.doneWaiting(*iPtr);
              } else {
                std::exception_ptr exceptionPtr;
                try {
                  //std::cout << "calling doProduce " << this << std::endl;
                  beginProduce(name_, event);
                  producer_.doProduce(event, eventSetup);
                  endProduce(name_, event);
                } catch (...) {
                  exceptionPtr = std::current_exception();
                }
                //std::cout << "waitingTasksWork_.doneWaiting " << this << std::endl;
                waitingTasksWork_.doneWaiting(exceptionPtr);
              }
            });
        auto* group = task.group();
        if (producer_.hasAcquire()) {
          WaitingTaskWithArenaHolder runProduceHolder{*group, moduleTask};
          moduleTask = make_waiting_task([this, &event, &eventSetup, runProduceHolder = std::move(runProduceHolder)](
                                             std::exception_ptr const* iPtr) mutable {
            if (iPtr) {
              runProduceHolder.doneWaiting(*iPtr);
            } else {
              std::exception_ptr exceptionPtr;
              try {
                beginAcquire(name_, event);
                producer_.doAcquire(event, eventSetup, runProduceHolder);
                endAcquire(name_, event);
              } catch (...) {
                exceptionPtr = std::current_exception();
              }
              runProduceHolder.doneWaiting(exceptionPtr);
            }
          });
        }
        //std::cout << "calling prefetchAsync " << this << " with moduleTask " << moduleTask << std::endl;
        prefetchAsync(event, eventSetup, WaitingTaskHolder(*group, moduleTask));
      }
    }

    void doWork(Event& event, EventSetup const& eventSetup) override {
      if (producer_.hasAcquire()) {
        FinalWaitingTask waitTask;
        tbb::task_group group;
        {
          WaitingTaskWithArenaHolder runProducerHolder{group, &waitTask};
          beginAcquire(name_, event);
          producer_.doAcquire(event, eventSetup, runProducerHolder);
          endAcquire(name_, event);
        }
        do {
          group.wait();
        } while (not waitTask.done());
        if (waitTask.exceptionPtr()) {
          std::rethrow_exception(*(waitTask.exceptionPtr()));
        }
      }
      beginProduce(name_, event);
      producer_.doProduce(event, eventSetup);
      endProduce(name_, event);
    }

    void doEndJob() override { producer_.doEndJob(); }

  private:
    void doReset() override {
      waitingTasksWork_.reset();
      workStarted_ = false;
    }

    T producer_;
    WaitingTaskList waitingTasksWork_;
    std::atomic<bool> workStarted_;
  };
}  // namespace edm
#endif
