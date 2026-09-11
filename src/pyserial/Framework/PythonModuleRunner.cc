#include "Framework/PythonModuleRunner.h"

#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/ProductRegistry.h"
#include "Framework/PythonRuntime.h"

namespace nb = nanobind;

namespace {
  // The modules run on TBB's threads, which have no Python thread state of
  // their own.  Without one, nb::gil_scoped_acquire creates a thread state on
  // every call into Python and deletes it on the way out, freeing the thread's
  // memory and mapping it back on the next call, through locks shared by the
  // whole interpreter: with many streams the threads spend most of their time
  // waiting on each other there.  So the first time a thread runs a Python
  // module it gets a thread state that it keeps, detached, until it exits;
  // from then on gil_scoped_acquire only attaches and detaches it.
  void keepThreadState() {
    thread_local bool kept = false;
    if (not kept) {
      PyGILState_Ensure();  // creates this thread's state and attaches it
      PyEval_SaveThread();  // detaches it, but the thread keeps it
      kept = true;
    }
  }
}  // namespace

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

  void PythonModuleRunner::endJob() {
    nb::gil_scoped_acquire guard;
    if (not nb::hasattr(impl_->worker, "endJob")) {
      return;
    }
    try {
      impl_->worker.attr("endJob")();
    } catch (nb::python_error const& e) {
      throw std::runtime_error(script_ + ".endJob() failed: " + e.what());
    }
  }

  void PythonModuleRunner::produce(Event& event, EventSetup const& eventSetup) {
    keepThreadState();
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
