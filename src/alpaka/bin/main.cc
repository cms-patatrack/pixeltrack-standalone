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

#include "AlpakaCore/alpakaConfigCommon.h"
#include "EventProcessor.h"

namespace {
  void print_help(std::string const& name) {
    std::cout
        << name
        << ": [--serial] [--tbb] [--cuda] [--numberOfThreads NT] [--numberOfStreams NS] [--maxEvents ME] [--data PATH] "
           "[--transfer] [--validation]\n\n"
        << "Options\n"
        << " --serial            Use CPU Serial backend\n"
        << " --tbb               Use CPU TBB backend\n"
        << " --cuda              Use CUDA backend\n"
        << " --numberOfThreads   Number of threads to use (default 1, use 0 to use all CPU cores)\n"
        << " --numberOfStreams   Number of concurrent events (default 0 = numberOfThreads)\n"
        << " --maxEvents         Number of events to process (default -1 for all events in the input file)\n"
        << " --runForMinutes     Continue processing the set of 1000 events until this many minutes have passed "
           "(default -1 for disabled; conflicts with --maxEvents)\n"
        << " --data              Path to the 'data' directory (default 'data' in the directory of the executable)\n"
        << " --transfer          Transfer results from GPU to CPU (default is to leave them on GPU)\n"
        << " --validation        Run (rudimentary) validation at the end (implies --transfer)\n"
        << " --histogram         Produce histograms at the end (implies --transfer)\n"
        << " --empty             Ignore all producers (for testing only)\n"
        << std::endl;
  }

  enum class Backend { SERIAL, TBB, CUDA };
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
  bool validation = false;
  bool histogram = false;
  bool empty = false;
  for (auto i = args.begin() + 1, e = args.end(); i != e; ++i) {
    if (*i == "-h" or *i == "--help") {
      print_help(args.front());
      return EXIT_SUCCESS;
    } else if (*i == "--serial") {
      backends.emplace_back(Backend::SERIAL);
    } else if (*i == "--tbb") {
      backends.emplace_back(Backend::TBB);
    } else if (*i == "--cuda") {
      backends.emplace_back(Backend::CUDA);
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
    } else if (*i == "--validation") {
      transfer = true;
      validation = true;
    } else if (*i == "--histogram") {
      transfer = true;
      histogram = true;
    } else if (*i == "--empty") {
      empty = true;
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

  // NB: The choice & tuning of device at runtime needs to be handled properly
  // inside a ALPAKA_ACCELERATOR_NAMESPACE.
  // For now, the choice is made at run time
  // with --serial, --tbb, --cuda (1 GPU only).

  // Initialize EventProcessor
  std::vector<std::string> edmodules;
  std::vector<std::string> esmodules;
  if (not empty) {
    esmodules = {"BeamSpotESProducer", "SiPixelFedIdsESProducer"};
    if (not backends.empty()) {
      auto addModules = [&](std::string const& prefix, Backend backend) {
        if (std::find(backends.begin(), backends.end(), backend) != backends.end()) {
          edmodules.emplace_back(prefix + "BeamSpotToAlpaka");
          edmodules.emplace_back(prefix + "SiPixelRawToCluster");
          edmodules.emplace_back(prefix + "SiPixelRecHitAlpaka");
          edmodules.emplace_back(prefix + "CAHitNtupletAlpaka");
          edmodules.emplace_back(prefix + "PixelVertexProducerAlpaka");
          if (transfer) {
            edmodules.emplace_back(prefix + "PixelTrackSoAFromAlpaka");
            edmodules.emplace_back(prefix + "PixelVertexSoAFromAlpaka");
          }
          if (validation) {
            edmodules.emplace_back(prefix + "CountValidator");
          }
          if (histogram) {
            edmodules.emplace_back(prefix + "HistoValidator");
          }

          esmodules.emplace_back(prefix + "SiPixelFedCablingMapESProducer");
          esmodules.emplace_back(prefix + "SiPixelGainCalibrationForHLTESProducer");
          esmodules.emplace_back(prefix + "PixelCPEFastESProducer");
        }
      };
      addModules("alpaka_serial_sync::", Backend::SERIAL);
      addModules("alpaka_tbb_async::", Backend::TBB);
      addModules("alpaka_cuda_async::", Backend::CUDA);
    }
  }
  edm::EventProcessor processor(
      maxEvents, runForMinutes, numberOfStreams, std::move(edmodules), std::move(esmodules), datadir, validation);

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
  auto start = std::chrono::high_resolution_clock::now();
  try {
    processor.runToCompletion();
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
  maxEvents = processor.processedEvents();
  std::cout << "Processed " << maxEvents << " events in " << std::scientific << time << " seconds, throughput "
            << std::defaultfloat << (maxEvents / time) << " events/s." << std::endl;
  return EXIT_SUCCESS;
}
