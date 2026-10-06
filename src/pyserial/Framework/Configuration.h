#ifndef Configuration_h
#define Configuration_h

#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/property_tree/ptree.hpp>

namespace edm {
  using ParameterSet = boost::property_tree::ptree;

  // One module's configuration section.  `label` is the section name and the
  // name the module's products are published under; `@type` names the plugin to
  // instantiate.  The same plugin may appear several times under different
  // labels, which is the point of labelling products in ProductRegistry.
  class ModuleConfig {
  public:
    ModuleConfig() = default;
    ModuleConfig(std::string label, ParameterSet parameters)
        : label_(std::move(label)), parameters_(std::move(parameters)) {}

    std::string const& label() const { return label_; }
    std::string type() const { return required<std::string>("@type"); }
    ParameterSet const& parameters() const { return parameters_; }

    /// Mandatory parameter.  Names the offending module rather than letting a
    /// bare ptree_bad_path escape.
    template <typename T>
    T required(std::string const& key) const {
      if (auto const value = parameters_.get_optional<T>(key)) {
        return *value;
      }
      throw std::runtime_error("module '" + label_ + "': missing or malformed parameter '" + key + "'");
    }

    /// Optional parameter, with a fallback used when the key is absent.
    template <typename T>
    T optional(std::string const& key, T fallback) const {
      if (auto const value = parameters_.get_optional<T>(key)) {
        return *value;
      }
      return fallback;
    }

  private:
    std::string label_;
    ParameterSet parameters_;
  };

  // A whole job.  The configuration owns the module path and the modules'
  // parameters; the run-time options (threads, streams, events) stay on the
  // command line, where run-scan.py and the existing test targets already drive
  // them, and the values here are only defaults for those.
  struct Configuration {
    std::vector<std::string> path;       ///< module labels, in schedule order
    std::vector<std::string> esmodules;  ///< EventSetup plugin type names
    std::map<std::string, ModuleConfig> modules;

    // Defaults for the command line; < 0 means "not specified here".
    int numberOfThreads = -1;
    int numberOfStreams = -1;
    int maxEvents = -1;
    int warmupEvents = -1;
    bool validation = false;
    bool histogram = false;
    bool transfer = false;

    ModuleConfig const& module(std::string const& label) const;
  };

  /// Reads an INI configuration.  Throws describing the first problem found.
  Configuration readConfiguration(std::filesystem::path const& file);
}  // namespace edm

#endif
