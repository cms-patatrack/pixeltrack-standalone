#ifndef ResourceMonitor_h
#define ResourceMonitor_h

#include <atomic>
#include <cstdint>
#include <ctime>
#include <deque>
#include <mutex>
#include <string>

namespace edm {

  // The real time and the thread CPU time spent in each module, in the source, in the EventSetup modules and in the
  // framework's tasks, summed over every stream, written at the end of the job as a JSON file in the format of
  // CMSSW's FastTimerService.  Nothing is measured unless it is enabled, which main() does for --resources.
  class ResourceMonitor {
  public:
    // One time interval, in nanoseconds of real and of thread CPU time.
    struct Clock {
      int64_t real;
      int64_t thread;

      static Clock now() noexcept {
        timespec r, t;
        clock_gettime(CLOCK_MONOTONIC, &r);
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
        return {r.tv_sec * 1'000'000'000LL + r.tv_nsec, t.tv_sec * 1'000'000'000LL + t.tv_nsec};
      }
    };

    // What one module, or one of the other entries, accumulates over the job.
    struct Counters {
      std::atomic<int64_t> events = 0;
      std::atomic<int64_t> real = 0;
      std::atomic<int64_t> thread = 0;

      void add(Clock const& start, Clock const& stop) noexcept {
        real.fetch_add(stop.real - start.real, std::memory_order_relaxed);
        thread.fetch_add(stop.thread - start.thread, std::memory_order_relaxed);
      }
      void reset() noexcept {
        events = 0;
        real = 0;
        thread = 0;
      }
    };

    static ResourceMonitor& instance();

    static bool enabled() noexcept { return enabled_.load(std::memory_order_relaxed); }
    static void enable() noexcept { enabled_ = true; }

    // The counters of the module with this label: every stream has its own instance of a module, and they all add to
    // the same counters.  The modules are reported in the order they are first registered.
    Counters* module(std::string const& label, std::string const& type);
    Counters& source() noexcept { return source_; }
    Counters& eventSetup() noexcept { return eventSetup_; }

    // The measurement covers the same interval as the throughput: everything accumulated before it starts, for
    // example during the warm up, is discarded, except the EventSetup, which is produced once before any event.
    void beginMeasurement(int threads);
    void endMeasurement(int64_t events);

    void writeJson(std::string const& fileName, std::string const& jobLabel) const;

    // Times the outermost framework task running on this thread; a task that runs another one inline is counted once.
    class TaskTiming {
    public:
      TaskTiming() noexcept {
        if (enabled()) {
          counted_ = true;
          if (depth_++ == 0) {
            start_ = Clock::now();
            active_ = true;
          }
        }
      }
      ~TaskTiming() {
        if (counted_) {
          --depth_;
          if (active_) {
            instance().tasks_.add(start_, Clock::now());
          }
        }
      }
      TaskTiming(TaskTiming const&) = delete;
      TaskTiming& operator=(TaskTiming const&) = delete;

    private:
      static thread_local int depth_;
      Clock start_;
      bool counted_ = false;
      bool active_ = false;
    };

    // Times a scope and adds it to the given counters, if the monitor is enabled.
    class ScopedTiming {
    public:
      explicit ScopedTiming(Counters* counters) noexcept : counters_(enabled() ? counters : nullptr) {
        if (counters_) {
          start_ = Clock::now();
        }
      }
      ~ScopedTiming() {
        if (counters_) {
          counters_->add(start_, Clock::now());
        }
      }
      // Counts one event for these counters.
      void countEvent() noexcept {
        if (counters_) {
          counters_->events.fetch_add(1, std::memory_order_relaxed);
        }
      }
      ScopedTiming(ScopedTiming const&) = delete;
      ScopedTiming& operator=(ScopedTiming const&) = delete;

    private:
      Counters* counters_;
      Clock start_;
    };

  private:
    struct Module {
      std::string label;
      std::string type;
      Counters counters;
    };

    static std::atomic<bool> enabled_;

    std::mutex mutex_;
    std::deque<Module> modules_;  // a deque does not move its elements, so the pointers given out stay valid
    Counters source_;
    Counters eventSetup_;
    Counters tasks_;

    int threads_ = 0;
    int64_t events_ = 0;
    int64_t wallStart_ = 0;
    int64_t wallStop_ = 0;
    int64_t cpuStart_ = 0;
    int64_t cpuStop_ = 0;
  };

}  // namespace edm

#endif  // ResourceMonitor_h
