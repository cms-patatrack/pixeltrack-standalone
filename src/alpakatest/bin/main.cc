#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/task_arena.h>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/backend.h"
#include "AlpakaCore/initialise.h"
#include "EventProcessor.h"
#include "PosixClockGettime.h"

namespace {
  void print_help(std::string const& name) {
    std::cout
        << name
        << ": [--serial] [--tbb] [--cuda] [--numberOfThreads NT] [--numberOfStreams NS] [--maxEvents ME] [--data PATH] "
           "[--transfer]\n\n"
        << "Options\n"
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
        << " --serial            Use CPU Serial backend\n"
#endif
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
        << " --tbb               Use CPU TBB backend\n"
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        << " --cuda              Use CUDA backend\n"
#endif
        << " --numberOfThreads   Number of threads to use (default 1, use 0 to use all CPU cores)\n"
        << " --numberOfStreams   Number of concurrent events (default 0 = numberOfThreads)\n"
        << " --maxEvents         Number of events to process (default -1 for all events in the input file)\n"
        << " --runForMinutes     Continue processing the set of 1000 events until this many minutes have passed"
           "(default -1 for disabled; conflicts with --maxEvents)\n"
        << " --data              Path to the 'data' directory (default 'data' in the directory of the executable)\n"
        << " --transfer          Transfer results from GPU to CPU (default is to leave them on GPU)\n"
        << " --empty             Ignore all producers (for testing only)\n"
        << std::endl;
  }
}  // namespace

int main(int argc, char** argv) {
  // Parse command line arguments
  std::vector<std::string> args(argv, argv + argc);
  std::vector<Backend> backends;
  int numberOfThreads = 1;
  int numberOfStreams = 0;
  int maxEvents = -1;
  int runForMinutes = -1;
  std::filesystem::path datadir;
  bool transfer = false;
  for (auto i = args.begin() + 1, e = args.end(); i != e; ++i) {
    if (*i == "-h" or *i == "--help") {
      print_help(args.front());
      return EXIT_SUCCESS;
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    } else if (*i == "--serial") {
      backends.emplace_back(Backend::SERIAL);
#endif
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    } else if (*i == "--tbb") {
      backends.emplace_back(Backend::TBB);
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    } else if (*i == "--cuda") {
      backends.emplace_back(Backend::CUDA);
#endif
    } else if (*i == "--numberOfThreads") {
      ++i;
      numberOfThreads = std::stoi(*i);
    } else if (*i == "--numberOfStreams") {
      ++i;
      numberOfStreams = std::stoi(*i);
    } else if (*i == "--maxEvents") {
      ++i;
      maxEvents = std::stoi(*i);
    } else if (*i == "--runForMinutes") {
      ++i;
      runForMinutes = std::stoi(*i);
    } else if (*i == "--data") {
      ++i;
      datadir = *i;
    } else if (*i == "--transfer") {
      transfer = true;
    } else {
      std::cout << "Invalid parameter " << *i << std::endl << std::endl;
      print_help(args.front());
      return EXIT_FAILURE;
    }
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

  // Initialiase the selected backends
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  if (std::find(backends.begin(), backends.end(), Backend::SERIAL) != backends.end()) {
    cms::alpakatools::initialise<alpaka_serial_sync::Platform>();
  }
#endif
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  if (std::find(backends.begin(), backends.end(), Backend::TBB) != backends.end()) {
    cms::alpakatools::initialise<alpaka_tbb_async::Platform>();
  }
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  if (std::find(backends.begin(), backends.end(), Backend::CUDA) != backends.end()) {
    cms::alpakatools::initialise<alpaka_cuda_async::Platform>();
  }
#endif

  // Initialize EventProcessor
  std::vector<std::string> edmodules;
  std::vector<std::string> esmodules;
  if (not backends.empty()) {
    auto addModules = [&](std::string const& prefix, Backend backend) {
      if (std::find(backends.begin(), backends.end(), backend) != backends.end()) {
        /*
        edmodules.emplace_back(prefix + "TestProducer");
        edmodules.emplace_back(prefix + "TestProducer3");
        edmodules.emplace_back(prefix + "TestProducer2");
        */
        edmodules.emplace_back(prefix + "TestProducerIsolated");
        edmodules.emplace_back(prefix + "TestProducerIsolatedMember");
        edmodules.emplace_back(prefix + "TestProducerProduce");
        edmodules.emplace_back(prefix + "TestProducerConsume");
      }
    };

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    addModules("alpaka_serial_sync::", Backend::SERIAL);
#endif
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    addModules("alpaka_tbb_async::", Backend::TBB);
#endif
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    addModules("alpaka_cuda_async::", Backend::CUDA);
#endif
    esmodules = {"IntESProducer"};
    if (transfer) {
      // add modules for transfer
    }
  }
  edm::EventProcessor processor(
      maxEvents, runForMinutes, numberOfStreams, std::move(edmodules), std::move(esmodules), datadir, false);

  if (runForMinutes < 0) {
    std::cout << "Processing " << processor.maxEvents() << " events, of which " << numberOfStreams
              << " concurrently, with " << numberOfThreads << " threads." << std::endl;
  } else {
    std::cout << "Processing for about " << runForMinutes << " minutes with " << numberOfStreams
              << " concurrent events and " << numberOfThreads << " threads." << std::endl;
  }

  // Initialize the TBB thread pool
  tbb::global_control tbb_max_threads{tbb::global_control::max_allowed_parallelism,
                                      static_cast<std::size_t>(numberOfThreads)};

  // Run work
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
  return EXIT_SUCCESS;
}
