#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

#include <tbb/task_scheduler_init.h>

#include <cuda_runtime.h>

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
        << " --numberOfThreads   Number of threads to use (default 1)\n"
        << " --numberOfStreams   Number of concurrent events (default 0=numberOfThreads)\n"
        << " --maxEvents         Number of events to process (default -1 for all events in the input file)\n"
        << " --data              Path to the 'data' directory (default 'data' in the directory of the executable)\n"
        << " --transfer          Transfer results from GPU to CPU (default is to leave them on GPU)\n"
        << " --validation        Run (rudimentary) validation at the end (implies --transfer)\n"
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
  std::filesystem::path datadir;
  bool transfer = false;
  bool validation = false;
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
    } else if (*i == "--data") {
      ++i;
      datadir = *i;
    } else if (*i == "--transfer") {
      transfer = true;
    } else if (*i == "--validation") {
      transfer = true;
      validation = true;
    } else {
      std::cout << "Invalid parameter " << *i << std::endl << std::endl;
      print_help(args.front());
      return EXIT_FAILURE;
    }
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
  if (auto found = std::find(backends.begin(), backends.end(), Backend::CUDA); found != backends.end()) {
    int numberOfDevices;
    auto status = cudaGetDeviceCount(&numberOfDevices);
    if (cudaSuccess != status) {
      std::cout << "Failed to initialize the CUDA runtime, disabling CUDA backend" << std::endl;
      backends.erase(found);
    } else {
      std::cout << "Found " << numberOfDevices << " devices" << std::endl;
    }
  }

  // Initialize EventProcessor
  std::vector<std::string> edmodules;
  std::vector<std::string> esmodules;
  esmodules = {"BeamSpotESProducer"};
  //esmodules = {"BeamSpotESProducer", "SiPixelFedIdsESProducer"};
  if (not backends.empty()) {
    auto addModules = [&](std::string const& prefix, Backend backend) {
      if (std::find(backends.begin(), backends.end(), backend) != backends.end()) {
	edmodules.emplace_back(prefix + "BeamSpotToAlpaka");
        /*edmodules.emplace_back(prefix + "SiPixelRawToCluster");
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
        esmodules.emplace_back(prefix + "PixelCPEFastESProducer");*/
      }
    };
    addModules("alpaka_serial_sync::", Backend::SERIAL);
    addModules("alpaka_tbb_async::", Backend::TBB);
    addModules("alpaka_cuda_async::", Backend::CUDA);
    if (transfer) {
      // add modules for transfer
    }
  }
  edm::EventProcessor processor(
      maxEvents, numberOfStreams, std::move(edmodules), std::move(esmodules), datadir, validation);
  maxEvents = processor.maxEvents();

  std::cout << "Processing " << maxEvents << " events, of which " << numberOfStreams << " concurrently, with "
            << numberOfThreads << " threads." << std::endl;

  // Initialize tasks scheduler (thread pool)
  // TO DO: Warning: this will not work, need to be included in header.
  // See external/alpaka/include/alpaka/kernel/TaskKernelCpuTbbBlocks.hpp
  tbb::task_scheduler_init tsi(numberOfThreads);

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
  std::cout << "Processed " << maxEvents << " events in " << std::scientific << time << " seconds, throughput "
            << std::defaultfloat << (maxEvents / time) << " events/s." << std::endl;
  return EXIT_SUCCESS;
}
