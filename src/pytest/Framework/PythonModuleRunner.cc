#include "Framework/PythonModuleRunner.h"

#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/ProductRegistry.h"
#include "Framework/PythonRuntime.h"

namespace nb = nanobind;

namespace edm {
  struct PythonModuleRunner::Impl {
    nb::object worker;
  };

  PythonModuleRunner::PythonModuleRunner(std::string script,
                                         std::string factory,
                                         std::map<std::string, std::string> const& config,
                                         ProductRegistry& registry)
      : impl_(std::make_unique<Impl>()), script_(std::move(script)) {
    // Starts the interpreter if this is the first Python module built.
    PythonRuntime::instance();

    nb::gil_scoped_acquire guard;
    try {
      nb::dict arguments;
      for (auto const& [key, value] : config) {
        arguments[nb::str(key.c_str())] = value;
      }

      impl_->worker = nb::module_::import_(script_.c_str())
                          .attr(factory.c_str())(arguments, nb::cast(&registry, nb::rv_policy::reference));
      if (not nb::hasattr(impl_->worker, "produce")) {
        throw std::runtime_error(factory + "() returned an object without a produce() method");
      }
    } catch (nb::python_error const& e) {
      throw std::runtime_error("python module '" + script_ + "': " + e.what());
    }
  }

  PythonModuleRunner::~PythonModuleRunner() {
    nb::gil_scoped_acquire guard;
    impl_->worker.reset();
  }

  void PythonModuleRunner::produce(Event& event, EventSetup const& eventSetup) {
    // On a free-threaded interpreter this attaches a thread state rather than
    // taking a lock, so streams do not serialise here.
    nb::gil_scoped_acquire guard;
    try {
      // Both passed by reference: Python works on this very Event, no copy.
      impl_->worker.attr("produce")(nb::cast(&event, nb::rv_policy::reference),
                                    nb::cast(&const_cast<EventSetup&>(eventSetup), nb::rv_policy::reference));
    } catch (nb::python_error const& e) {
      throw std::runtime_error(script_ + ".produce() failed: " + e.what());
    }
  }
}  // namespace edm
