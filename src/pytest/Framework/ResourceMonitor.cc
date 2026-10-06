#include "Framework/ResourceMonitor.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace edm {
  std::atomic<bool> ResourceMonitor::enabled_ = false;
  thread_local int ResourceMonitor::TaskTiming::depth_ = 0;

  namespace {
    int64_t nanoseconds(clockid_t clock) {
      timespec t;
      clock_gettime(clock, &t);
      return t.tv_sec * 1'000'000'000LL + t.tv_nsec;
    }

    // A label or a type, as a JSON string.
    std::string quoted(std::string const& value) {
      std::string out = "\"";
      for (char c : value) {
        if (c == '"' or c == '\\') {
          out += '\\';
        }
        out += c;
      }
      return out + '"';
    }

    struct Entry {
      std::string label;
      std::string type;
      int64_t events;
      int64_t real;
      int64_t thread;
    };

    // The times in milliseconds, as the FastTimerService writes them.
    std::string format(Entry const& e, char const* indent) {
      char buffer[256];
      std::snprintf(buffer,
                    sizeof(buffer),
                    "%s  \"events\": %lld,\n%s  \"label\": ",
                    indent,
                    static_cast<long long>(e.events),
                    indent);
      std::string out = std::string(indent) + "{\n" + buffer + quoted(e.label) + ",\n";
      std::snprintf(buffer,
                    sizeof(buffer),
                    "%s  \"time_real\": %.6f,\n%s  \"time_thread\": %.6f,\n%s  \"type\": ",
                    indent,
                    e.real * 1e-6,
                    indent,
                    e.thread * 1e-6,
                    indent);
      return out + buffer + quoted(e.type) + "\n" + indent + "}";
    }
  }  // namespace

  ResourceMonitor& ResourceMonitor::instance() {
    static ResourceMonitor monitor;
    return monitor;
  }

  ResourceMonitor::Counters* ResourceMonitor::module(std::string const& label, std::string const& type) {
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto& m : modules_) {
      if (m.label == label) {
        return &m.counters;
      }
    }
    auto& m = modules_.emplace_back();
    m.label = label;
    m.type = type;
    return &m.counters;
  }

  void ResourceMonitor::beginMeasurement(int threads) {
    for (auto& m : modules_) {
      m.counters.reset();
    }
    source_.reset();
    tasks_.reset();
    threads_ = threads;
    wallStart_ = nanoseconds(CLOCK_MONOTONIC);
    cpuStart_ = nanoseconds(CLOCK_PROCESS_CPUTIME_ID);
  }

  void ResourceMonitor::endMeasurement(int64_t events) {
    wallStop_ = nanoseconds(CLOCK_MONOTONIC);
    cpuStop_ = nanoseconds(CLOCK_PROCESS_CPUTIME_ID);
    events_ = events;
  }

  void ResourceMonitor::writeJson(std::string const& fileName, std::string const& jobLabel) const {
    std::vector<Entry> entries;
    entries.push_back({"source", "Source", source_.events, source_.real, source_.thread});
    int64_t inModulesReal = source_.real;
    int64_t inModulesThread = source_.thread;
    for (auto const& m : modules_) {
      entries.push_back({m.label, m.type, m.counters.events, m.counters.real, m.counters.thread});
      inModulesReal += m.counters.real;
      inModulesThread += m.counters.thread;
    }
    // The time in the framework's tasks outside of any module or of the source: scheduling, prefetching, starting
    // the next event, and cleaning up the last one.
    entries.push_back({"other", "other", events_, tasks_.real - inModulesReal, tasks_.thread - inModulesThread});
    entries.push_back({"eventsetup", "eventsetup", events_, eventSetup_.real, eventSetup_.thread});
    // Every thread of the pool is either running a task or idle; the CPU time of the process outside of the tasks is
    // what the threads used while waiting for work, and anything run outside of them, like the threads of an
    // external worker.
    entries.push_back({"idle",
                       "idle",
                       events_,
                       threads_ * (wallStop_ - wallStart_) - tasks_.real,
                       (cpuStop_ - cpuStart_) - tasks_.thread});

    Entry total{jobLabel, "Job", events_, 0, 0};
    for (auto const& e : entries) {
      total.real += e.real;
      total.thread += e.thread;
    }

    std::ofstream out(fileName);
    if (not out) {
      throw std::runtime_error("ResourceMonitor: cannot write " + fileName);
    }
    out << "{\n  \"modules\": [\n";
    for (std::size_t i = 0; i < entries.size(); ++i) {
      out << format(entries[i], "    ") << (i + 1 < entries.size() ? ",\n" : "\n");
    }
    out << "  ],\n"
        << "  \"resources\": [\n"
        << "    {\n      \"time_real\": \"real time\"\n    },\n"
        << "    {\n      \"time_thread\": \"cpu time\"\n    }\n"
        << "  ],\n"
        << "  \"total\": " << format(total, "  ").substr(2) << "\n}\n";
  }
}  // namespace edm
