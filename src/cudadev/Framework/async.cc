#include "Framework/async.h"

namespace edm::impl {
  WaitingThread::WaitingThread() { thread_ = std::thread(&WaitingThread::threadLoop, this); }

  WaitingThread::~WaitingThread() {
    if (not stopThread_) {
      stopThread();
    }
    thread_.join();
  }

  void WaitingThread::threadLoop() {
    std::unique_lock lk(mutex_);

    do {
      cond_.wait(lk, [this]() { return static_cast<bool>(func_) or stopThread_; });
      if (func_) {
        func_();
        decltype(func_)().swap(func_);
        thisPtr_.reset();
      }
    } while (not stopThread_);
  }
}  // namespace edm::impl
