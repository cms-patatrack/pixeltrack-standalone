#ifndef Worker_h
#define Worker_h

#include <atomic>
#include <vector>
//#include <iostream>

#include "Framework/WaitingTask.h"
#include "Framework/WaitingTaskHolder.h"
#include "Framework/WaitingTaskList.h"
#include "Framework/WaitingTaskWithArenaHolder.h"

namespace edm {
  class Event;
  class EventSetup;
  class ProductRegistry;

  class Worker {
  public:
    virtual ~Worker() = default;

    // not thread safe
    void setItemsToGet(std::vector<Worker*> workers) { itemsToGet_ = std::move(workers); }

    // thread safe
    void prefetchAsync(Event& event, EventSetup const& eventSetup, WaitingTask* iTask);

    // not thread safe
    virtual void doWorkAsync(Event& event, EventSetup const& eventSetup, WaitingTask* iTask) = 0;

    // not thread safe
    void reset() {
      prefetchRequested_ = false;
      doReset();
    }

  protected:
    virtual void doReset() = 0;

  private:
    std::vector<Worker*> itemsToGet_;
    std::atomic<bool> prefetchRequested_ = false;
  };

  template <typename T>
  class WorkerT : public Worker {
  public:
    explicit WorkerT(ProductRegistry& reg) : producer_(reg) {}

    void doWorkAsync(Event& event, EventSetup const& eventSetup, WaitingTask* iTask) override {
      waitingTasksWork_.add(iTask);
      //std::cout << "doWorkAsync for " << this << " with iTask " << iTask << std::endl;
      bool expected = false;
      if (workStarted_.compare_exchange_strong(expected, true)) {
        //std::cout << "first doWorkAsync call" << std::endl;

        WaitingTask* moduleTask = make_waiting_task(
            tbb::task::allocate_root(), [this, &event, &eventSetup](std::exception_ptr const* iPtr) mutable {
              if (iPtr) {
                waitingTasksWork_.doneWaiting(*iPtr);
              } else {
                std::exception_ptr exceptionPtr;
                try {
                  //std::cout << "calling doProduce " << this << std::endl;
                  producer_.doProduce(event, eventSetup);
                } catch (...) {
                  exceptionPtr = std::current_exception();
                }
                //std::cout << "waitingTasksWork_.doneWaiting " << this << std::endl;
                waitingTasksWork_.doneWaiting(exceptionPtr);
              }
            });
        if (producer_.hasAcquire()) {
          WaitingTaskWithArenaHolder runProduceHolder{moduleTask};
          moduleTask = make_waiting_task(tbb::task::allocate_root(),
                                         [this, &event, &eventSetup, runProduceHolder = std::move(runProduceHolder)](
                                             std::exception_ptr const* iPtr) mutable {
                                           if (iPtr) {
                                             runProduceHolder.doneWaiting(*iPtr);
                                           } else {
                                             std::exception_ptr exceptionPtr;
                                             try {
                                               producer_.doAcquire(event, eventSetup, runProduceHolder);
                                             } catch (...) {
                                               exceptionPtr = std::current_exception();
                                             }
                                             runProduceHolder.doneWaiting(exceptionPtr);
                                           }
                                         });
        }
        //std::cout << "calling prefetchAsync " << this << " with moduleTask " << moduleTask << std::endl;
        prefetchAsync(event, eventSetup, moduleTask);
      }
    }

  private:
    void doReset() override {
      waitingTasksWork_.reset();
      workStarted_ = false;
    }

    T producer_;
    WaitingTaskList waitingTasksWork_;
    std::atomic<bool> workStarted_ = false;
  };
}  // namespace edm
#endif
