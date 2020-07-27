#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

#include <tbb/task_scheduler_init.h>

#include "EventProcessor.h"

namespace {
  void print_help(std::string const& name) {
    std::cout
        << name
        << ": [--numberOfThreads NT] [--numberOfStreams NS] [--maxEvents ME] [--data PATH] [--transfer] [--validation] "
           "[--empty]\n\n"
        << "Options\n"
        << " --numberOfThreads   Number of threads to use (default 1)\n"
        << " --numberOfStreams   Number of concurrent events (default 0=numberOfThreads)\n"
        << " --maxEvents         Number of events to process (default -1 for all events in the input file)\n"
        << " --data              Path to the 'data' directory (default 'data' in the directory of the executable)\n"
        << " --transfer          Transfer results from GPU to CPU (default is to leave them on GPU)\n"
        << " --validation        Run (rudimentary) validation at the end (implies --transfer)\n"
        << " --empty             Ignore all producers (for testing only)\n"
        << std::endl;
  }
}  // namespace

int main(int argc, char** argv) try {
  // Parse command line arguments
  std::vector<std::string> args(argv, argv + argc);
  int numberOfThreads = 1;
  int numberOfStreams = 0;
  int maxEvents = -1;
  std::filesystem::path datadir;
  bool transfer = false;
  bool validation = false;
  bool empty = false;
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
    } else if (*i == "--empty") {
      empty = true;
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
  int numberOfDevices;
  /*
  DPCT1003:69: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  auto status = (numberOfDevices = dpct::dev_mgr::instance().device_count(), 0);
  /*
  DPCT1000:68: Error handling if-stmt was detected but could not be rewritten.
  */
  if (0 != status) {
    /*
    DPCT1001:67: The statement could not be removed.
    */
    std::cout << "Failed to initialize the SYCL runtime";
    return EXIT_FAILURE;
  }
  std::cout << "Found " << numberOfDevices << " devices" << std::endl;

  // Initialize EventProcessor
  std::vector<std::string> edmodules;
  std::vector<std::string> esmodules;
  if (not empty) {
    edmodules = {"TestProducer", "TestProducer3", "TestProducer2"};
    esmodules = {"IntESProducer"};
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
catch (::sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
