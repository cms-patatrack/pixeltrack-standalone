// A producer whose produce() is implemented in Python.
//
// Parameters: `script` (the importable module name, default: the module's own
// label) and `factory` (default "create").  Every key of the configuration
// section is handed to the factory as a dict -- the Python counterpart of
// passing the ParameterSet to a C++ constructor -- along with the very
// ProductRegistry this module is being constructed against:
//
//     def create(config: dict[str, str], registry) -> object   # with .produce(event, eventSetup)
//
// So a script declares exactly the way a C++ module does, against the same
// object, and the tokens it gets back are the only way it can reach a product.
//
// One instance is constructed per stream, each holding its own Python object,
// which is what lets the streams run their Python modules concurrently on a
// free-threaded interpreter.
//
// Everything that touches nanobind is behind PythonModuleRunner; see the note
// there on why a plugin cannot use nanobind directly.

#include <map>
#include <string>

#include "Framework/Configuration.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/PythonModuleRunner.h"

class PythonProducer : public edm::EDProducer {
public:
  PythonProducer(edm::ModuleConfig const& config, edm::ProductRegistry& reg)
      : runner_(config.optional<std::string>("script", config.label()),
                config.optional<std::string>("factory", std::string("create")),
                parameters(config),
                reg) {}

private:
  /// The whole section as plain strings, which is what the factory receives.
  static std::map<std::string, std::string> parameters(edm::ModuleConfig const& config) {
    std::map<std::string, std::string> out;
    for (auto const& [key, value] : config.parameters()) {
      out.emplace(key, value.get_value<std::string>());
    }
    return out;
  }

  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override {
    runner_.produce(event, eventSetup);
  }

  /// Only the first stream's modules are asked, as for a C++ module, so a
  /// script that accumulates over the whole job keeps its counters at module
  /// level and shares them across streams.
  void endJob() override { runner_.endJob(); }

  edm::PythonModuleRunner runner_;
};

DEFINE_FWK_MODULE(PythonProducer);
