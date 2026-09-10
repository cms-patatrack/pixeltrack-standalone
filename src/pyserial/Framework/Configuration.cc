#include "Framework/Configuration.h"

#include <algorithm>
#include <set>

#include <boost/property_tree/ini_parser.hpp>

namespace edm {
  namespace {
    /// Splits "a, b, c" into its trimmed, non-empty parts.
    std::vector<std::string> splitList(std::string const& value) {
      std::vector<std::string> out;
      std::string item;
      for (char c : value + ",") {
        if (c == ',') {
          const auto first = item.find_first_not_of(" \t");
          const auto last = item.find_last_not_of(" \t");
          if (first != std::string::npos) {
            out.push_back(item.substr(first, last - first + 1));
          }
          item.clear();
        } else {
          item += c;
        }
      }
      return out;
    }

    /// The keys [options] understands.  A typo there used to be a silent
    /// no-op in the nanobind demo and cost a misleading measurement, so an
    /// unknown key is an error.
    const std::set<std::string> kOptionKeys = {"modules",
                                               "esmodules",
                                               "numberOfThreads",
                                               "numberOfStreams",
                                               "maxEvents",
                                               "warmupEvents",
                                               "validation",
                                               "histogram",
                                               "transfer"};
  }  // namespace

  ModuleConfig const& Configuration::module(std::string const& label) const {
    const auto found = modules.find(label);
    if (found == modules.end()) {
      throw std::runtime_error("no configuration section '[" + label + "]'");
    }
    return found->second;
  }

  Configuration readConfiguration(std::filesystem::path const& file) {
    if (not std::filesystem::exists(file)) {
      throw std::runtime_error("configuration file '" + file.string() + "' does not exist");
    }

    ParameterSet root;
    try {
      boost::property_tree::read_ini(file.string(), root);
    } catch (boost::property_tree::ini_parser_error const& e) {
      throw std::runtime_error("cannot parse '" + file.string() + "': " + e.message());
    }

    Configuration config;

    const auto options = root.get_child_optional("options");
    if (not options) {
      throw std::runtime_error("the configuration has no '[options]' section");
    }
    for (auto const& [key, value] : *options) {
      if (not kOptionKeys.count(key)) {
        throw std::runtime_error("unknown key '" + key + "' in [options]");
      }
    }

    config.path = splitList(options->get<std::string>("modules", ""));
    config.esmodules = splitList(options->get<std::string>("esmodules", ""));
    config.numberOfThreads = options->get<int>("numberOfThreads", -1);
    config.numberOfStreams = options->get<int>("numberOfStreams", -1);
    config.maxEvents = options->get<int>("maxEvents", -1);
    config.warmupEvents = options->get<int>("warmupEvents", -1);
    config.validation = options->get<bool>("validation", false);
    config.histogram = options->get<bool>("histogram", false);
    config.transfer = options->get<bool>("transfer", false);

    if (config.path.empty()) {
      throw std::runtime_error("the configuration has no '[options] modules = ...' entry");
    }

    // Every section other than [options] describes a module, whether or not it
    // is scheduled: --validation and --histogram append their module to the
    // path, so a section has to be readable before it is in the path.
    for (auto const& [label, section] : root) {
      if (label == "options") {
        continue;
      }
      if (not section.get_optional<std::string>("@type")) {
        throw std::runtime_error("section '[" + label + "]' declares no '@type'");
      }
      config.modules.emplace(label, ModuleConfig(label, section));
    }

    // A scheduled label with no section used to fail much later and much less
    // clearly.
    for (auto const& label : config.path) {
      if (not config.modules.count(label)) {
        throw std::runtime_error("module '" + label + "' is scheduled but has no '[" + label + "]' section");
      }
    }

    // A label may appear only once in the path: products are keyed by label,
    // and two instances would collide on the second produces().
    std::set<std::string> seen;
    for (auto const& label : config.path) {
      if (not seen.insert(label).second) {
        throw std::runtime_error("module '" + label + "' appears more than once in [options] modules");
      }
    }

    return config;
  }
}  // namespace edm
