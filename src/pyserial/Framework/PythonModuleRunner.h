#ifndef PythonModuleRunner_h
#define PythonModuleRunner_h

#include <map>
#include <memory>
#include <string>

namespace edm {
  class Event;
  class EventSetup;
  class ProductRegistry;

  /// Owns one Python module object and calls its produce() for each event.
  ///
  /// Deliberately free of nanobind in its interface.  The nanobind runtime is
  /// compiled into libFramework.so with hidden visibility, so a plugin that
  /// used nanobind directly would fail to link against it; and building the
  /// runtime into each plugin instead would give every one its own type
  /// registry, so casts would stop working across the boundary.  Everything
  /// that touches nanobind therefore lives in the implementation of this class,
  /// and a plugin sees only ordinary C++.
  class PythonModuleRunner {
  public:
    /// Imports `script`, calls `factory`(config, registry) and keeps the object
    /// it returns.  Throws if the script has no such factory, or if what it
    /// returns has no produce() method.
    PythonModuleRunner(std::string script,
                       std::string factory,
                       std::map<std::string, std::string> const& config,
                       ProductRegistry& registry);
    ~PythonModuleRunner();

    PythonModuleRunner(PythonModuleRunner const&) = delete;
    PythonModuleRunner& operator=(PythonModuleRunner const&) = delete;

    void produce(Event& event, EventSetup const& eventSetup);

    /// Calls the object's endJob(), if it has one.  Optional because most
    /// modules have nothing to do at the end of the job, and a C++ module says
    /// so by not overriding it.
    void endJob();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string script_;
  };
}  // namespace edm

#endif
