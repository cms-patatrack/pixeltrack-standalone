#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <malloc.h>

#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/task_arena.h>

#include "Framework/Configuration.h"
#include "Framework/PythonRuntime.h"

#include "EventProcessor.h"
#include "Framework/ResourceMonitor.h"
#include "PosixClockGettime.h"

namespace {
  // The event data products and the per-event working buffers are large (up to ~20 MB each) and are allocated
  // and freed for every event. By default glibc serves such blocks with mmap() and returns them to the system
  // with munmap() on free, so every event pays for page faults and for the kernel zeroing the new pages.
  // Keep these blocks in the malloc heaps instead, and do not trim the heaps, so that the memory is reused.
  void configureMalloc() {
    mallopt(M_MMAP_THRESHOLD, 32 * 1024 * 1024);  // the maximum value allowed by glibc on 64-bit systems
    mallopt(M_TRIM_THRESHOLD, 1024 * 1024 * 1024);
  }

  void print_help(std::string const& name) {
    std::cout
        << "Usage: " << name
        << " CONFIG.ini [--numberOfThreads NT] [--numberOfStreams NS] [--warmupEvents WE] [--maxEvents ME]"
        << " [--runForMinutes RM] [--data PATH] [--validation] [--histogram] [--empty] [--resources FILE.json]\n";
    std::cout << R"(
Arguments:
  CONFIG.ini                    Configuration file: the module path in [options],
                                one section per module giving its '@type' and
                                parameters.  See test.ini.

Options:
  --numberOfThreads             Number of threads to use (default 1, use 0 to use all CPU cores).
  --numberOfStreams             Number of concurrent events (default 0 = numberOfThreads).
  --warmupEvents                Number of events to process before starting the benchmark (default 0).
                                These four override the matching keys in [options].
  --maxEvents                   Number of events to process (default -1 for all events in the input file).
  --runForMinutes               Continue processing the set of 1000 events until this many minutes have passed
                                (default -1 for disabled; conflicts with --maxEvents).
  --data                        Path to the 'data' directory (default 'data' in the directory of the executable).
  --validation                  Run (rudimentary) validation at the end.
  --histogram                   Produce histograms at the end.
                                Each appends its module to the configured path.
  --empty                       Ignore all producers (for testing only).
  --resources                   Write the real and CPU time spent in each module, in the source, in the EventSetup,
                                elsewhere in the framework and idle, summed over the measured events, to FILE.json.
)";
  }
}  // namespace

int main(int argc, char** argv) {
  configureMalloc();

  // Parse command line arguments
  std::vector<std::string> args(argv, argv + argc);
  int numberOfThreads = -1;
  int numberOfStreams = -1;
  int warmupEvents = -1;
  int maxEvents = -1;
  int runForMinutes = -1;
  std::filesystem::path configFile;
  std::filesystem::path datadir;
  bool validation = false;
  bool histogram = false;
  bool empty = false;
  std::string resources;
  for (auto i = args.begin() + 1, e = args.end(); i != e; ++i) {
    if (*i == "-h" or *i == "--help") {
      print_help(args.front());
      return EXIT_SUCCESS;
    } else if (*i == "--numberOfThreads") {
      ++i;
      numberOfThreads = std::stoi(*i);
    } else if (*i == "--numberOfStreams") {
      ++i;
      numberOfStreams = std::stoi(*i);
    } else if (*i == "--warmupEvents") {
      ++i;
      warmupEvents = std::stoi(*i);
    } else if (*i == "--maxEvents") {
      ++i;
      maxEvents = std::stoi(*i);
    } else if (*i == "--runForMinutes") {
      ++i;
      runForMinutes = std::stoi(*i);
    } else if (*i == "--data") {
      ++i;
      datadir = *i;
    } else if (*i == "--validation") {
      validation = true;
    } else if (*i == "--histogram") {
      histogram = true;
    } else if (*i == "--empty") {
      empty = true;
    } else if (*i == "--resources") {
      ++i;
      resources = *i;
    } else if (i->size() > 2 and i->substr(0, 2) == "--") {
      std::cout << "Invalid parameter " << *i << std::endl << std::endl;
      print_help(args.front());
      return EXIT_FAILURE;
    } else if (configFile.empty()) {
      configFile = *i;
    } else {
      std::cout << "More than one configuration file given" << std::endl << std::endl;
      print_help(args.front());
      return EXIT_FAILURE;
    }
  }
  if (configFile.empty()) {
    std::cout << "No configuration file given" << std::endl << std::endl;
    print_help(args.front());
    return EXIT_FAILURE;
  }

  edm::Configuration configuration;
  try {
    configuration = edm::readConfiguration(configFile);
  } catch (std::exception& e) {
    std::cout << "error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  // The command line wins over [options]; [options] wins over the built-in
  // default.  run-scan.py and the test targets drive the command line, so the
  // configuration must not take those away from them.
  auto resolve = [](int cmdline, int fromConfig, int fallback) {
    return cmdline >= 0 ? cmdline : (fromConfig >= 0 ? fromConfig : fallback);
  };
  numberOfThreads = resolve(numberOfThreads, configuration.numberOfThreads, 1);
  numberOfStreams = resolve(numberOfStreams, configuration.numberOfStreams, 0);
  warmupEvents = resolve(warmupEvents, configuration.warmupEvents, 0);
  if (maxEvents < 0) {
    maxEvents = configuration.maxEvents;
  }
  validation = validation or configuration.validation;
  histogram = histogram or configuration.histogram;
  if (empty) {
    configuration.path.clear();
    configuration.esmodules.clear();
  }

  // --validation and --histogram append their module to the path, so that the
  // existing test targets and run-scan.py keep working now that the schedule
  // comes from the configuration.  The label has to exist in the file: adding
  // a module the configuration says nothing about would be a schedule nobody
  // wrote down.
  auto const appendModule = [&](bool wanted, char const* label, char const* option) {
    if (not wanted or empty) {
      return true;
    }
    if (not configuration.modules.count(label)) {
      std::cout << "error: " << option << " needs a '[" << label << "]' section in the configuration"
                << std::endl;
      return false;
    }
    if (std::find(configuration.path.begin(), configuration.path.end(), label) == configuration.path.end()) {
      configuration.path.emplace_back(label);
    }
    return true;
  };
  if (not appendModule(validation, "countValidator", "--validation") or
      not appendModule(histogram, "histoValidator", "--histogram")) {
    return EXIT_FAILURE;
  }
  if (maxEvents >= 0 and runForMinutes >= 0) {
    std::cout << "Got both --maxEvents and --runForMinutes, please give only one of them" << std::endl;
    return EXIT_FAILURE;
  }
  if (numberOfThreads == 0) {
    numberOfThreads = tbb::info::default_concurrency();
  }
  if (numberOfStreams == 0) {
    numberOfStreams = numberOfThreads;
  }
  if (datadir.empty()) {
    datadir = std::filesystem::path(args[0]).parent_path() / "data";
  }
  if (not std::filesystem::exists(datadir)) {
    std::cout << "Data directory '" << datadir << "' does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  // Before the modules are constructed, so that the EventSetup is measured too.
  if (not resources.empty()) {
    edm::ResourceMonitor::enable();
  }

  // Initialize EventProcessor.  Constructing the modules is what starts the
  // interpreter, if any module is implemented in Python.
  edm::EventProcessor processor(
      warmupEvents, maxEvents, runForMinutes, numberOfStreams, configuration, datadir, validation);

  // Say plainly whether the GIL is on.  With it enabled the Python modules of
  // every stream serialise, and any throughput compared against an all-C++ run
  // measures that rather than anything about Python -- so it is reported next
  // to the stream count rather than left to be inferred.
  if (edm::PythonRuntime::running()) {
    auto const& python = edm::PythonRuntime::instance();
    std::cout << "Python " << python.version().substr(0, python.version().find(' ')) << ", GIL "
              << (python.gilEnabled() ? "ENABLED -- Python modules will serialise across streams"
                                      : "disabled")
              << std::endl;
  }

  if (runForMinutes < 0) {
    std::cout << "Processing " << processor.maxEvents() << " events,";
  } else {
    std::cout << "Processing for about " << runForMinutes << " minutes,";
  }
  if (warmupEvents > 0) {
    std::cout << " after " << warmupEvents << " events of warm up,";
  }
  std::cout << " with " << numberOfStreams << " concurrent events and " << numberOfThreads << " threads." << std::endl;

  // Initialize the TBB thread pool
  tbb::global_control tbb_max_threads{tbb::global_control::max_allowed_parallelism,
                                      static_cast<std::size_t>(numberOfThreads)};

  // Warm up
  try {
    tbb::task_arena arena(numberOfThreads);
    arena.execute([&] { processor.warmUp(); });
  } catch (std::runtime_error& e) {
    std::cout << "\n----------\nCaught std::runtime_error" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    std::cout << "\n----------\nCaught std::exception" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cout << "\n----------\nCaught exception of unknown type" << std::endl;
    return EXIT_FAILURE;
  }

  // Run work
  edm::ResourceMonitor::instance().beginMeasurement(numberOfThreads);
  auto cpu_start = PosixClockGettime<CLOCK_PROCESS_CPUTIME_ID>::now();
  auto start = std::chrono::high_resolution_clock::now();
  try {
    tbb::task_arena arena(numberOfThreads);
    arena.execute([&] { processor.runToCompletion(); });
  } catch (std::runtime_error& e) {
    std::cout << "\n----------\nCaught std::runtime_error" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    std::cout << "\n----------\nCaught std::exception" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cout << "\n----------\nCaught exception of unknown type" << std::endl;
    return EXIT_FAILURE;
  }
  auto cpu_stop = PosixClockGettime<CLOCK_PROCESS_CPUTIME_ID>::now();
  auto stop = std::chrono::high_resolution_clock::now();
  edm::ResourceMonitor::instance().endMeasurement(processor.processedEvents());

  // Run endJob
  try {
    processor.endJob();
  } catch (std::runtime_error& e) {
    std::cout << "\n----------\nCaught std::runtime_error" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    std::cout << "\n----------\nCaught std::exception" << std::endl;
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cout << "\n----------\nCaught exception of unknown type" << std::endl;
    return EXIT_FAILURE;
  }

  // Work done, report timing
  auto diff = stop - start;
  auto time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(diff).count()) / 1e6;
  auto cpu_diff = cpu_stop - cpu_start;
  auto cpu = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(cpu_diff).count()) / 1e6;
  maxEvents = processor.processedEvents();
  std::cout << "Processed " << maxEvents << " events in " << std::scientific << time << " seconds, throughput "
            << std::defaultfloat << (maxEvents / time) << " events/s, CPU usage per thread: " << std::fixed
            << std::setprecision(1) << (cpu / time / numberOfThreads * 100) << "%" << std::endl;
  if (not resources.empty()) {
    try {
      edm::ResourceMonitor::instance().writeJson(resources, configFile.stem().string());
    } catch (std::exception& e) {
      std::cout << "error: " << e.what() << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
