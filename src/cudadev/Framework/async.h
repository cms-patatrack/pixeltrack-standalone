#ifndef Framework_async_h
#define Framework_async_h

#include "Framework/ReusableObjectHolder.h"
#include "Framework/WaitingTaskWithArenaHolder.h"

#include <condition_variable>
#include <mutex>
#include <thread>

namespace edm {
  namespace impl {
    class WaitingThread {
    public:
      WaitingThread();
      ~WaitingThread();

      template <typename F>
      void run(WaitingTaskWithArenaHolder holder, F&& func, std::shared_ptr<WaitingThread> thisPtr) {
        std::unique_lock lk(mutex_);
        func_ = [holder = std::move(holder), func = std::forward<F>(func)]() mutable {
          try {
            func();
            holder.doneWaiting(nullptr);
          } catch (...) {
            holder.doneWaiting(std::current_exception());
          }
        };
        thisPtr_ = std::move(thisPtr);
        cond_.notify_one();
      }

      void stopThread() {
        std::unique_lock lk(mutex_);
        stopThread_ = true;
        cond_.notify_one();
      }

    private:
      void threadLoop();

      std::thread thread_;
      std::mutex mutex_;
      std::condition_variable cond_;
      std::function<void()> func_;
      std::shared_ptr<WaitingThread> thisPtr_;
      bool ready_ = false;
      bool stopThread_ = false;
    };

    class WaitingThreadPool {
    public:
      template <typename F>
      void run(WaitingTaskWithArenaHolder holder, F&& func) {
        auto thread = pool_.makeOrGet([]() { return std::make_unique<WaitingThread>(); });
        thread->run(std::move(holder), std::forward<F>(func), thread);
      }

    private:
      edm::ReusableObjectHolder<WaitingThread> pool_;
    };

    WaitingThreadPool& getWaitingThreadPool() {
      static WaitingThreadPool pool;
      return pool;
    }
  }  // namespace impl

  template <typename F>
  void async(WaitingTaskWithArenaHolder holder, F&& func) {
    auto& pool = impl::getWaitingThreadPool();
    pool.run(std::move(holder), std::forward<F>(func));
  }
}  // namespace edm

#endif
