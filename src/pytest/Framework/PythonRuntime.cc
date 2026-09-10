#include "Framework/PythonRuntime.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

// Baked in by the backend makefile.  Each is overridable at run time through
// the environment variable of the same name.
#ifndef EDM_PYTHON_HOME
#define EDM_PYTHON_HOME ""
#endif
// site-packages of external/venv, so that packages installed for this project
// (numpy, ...) are importable even though PYTHONHOME points at the base
// installation.
#ifndef EDM_PYTHON_SITE
#define EDM_PYTHON_SITE ""
#endif
// Where the backend's own Python modules live.
#ifndef EDM_PYTHON_DIR
#define EDM_PYTHON_DIR "python"
#endif
// Byte-code cache, kept out of the source tree.
#ifndef EDM_PYTHON_PYCACHE
#define EDM_PYTHON_PYCACHE ""
#endif

namespace nb = nanobind;


// LIB_DIR arrives from the makefile unquoted, as it does in PluginManager.
#define STR_EXPAND(x) #x
#define STR(x) STR_EXPAND(x)

namespace edm {
  namespace {
    PythonRuntime* g_runtime = nullptr;

    std::string envOr(char const* name, std::string fallback) {
      if (char const* value = std::getenv(name); value != nullptr and *value != '\0') {
        return value;
      }
      return fallback;
    }

    /// Confines the thread pools that numeric libraries start behind our back.
    ///
    /// OpenBLAS, and OpenMP runtimes generally, size a pool to the whole
    /// machine on first use and spin it, which is neither counted nor capped by
    /// TBB because TBB does not know it exists.  On a 62-core machine that
    /// showed up in the nanobind demo as a supposedly serial run burning 180%
    /// cpu while doing no extra work at all.  The throughput barely moved, but
    /// the baseline was wrong and every speedup measured against it was wrong
    /// with it -- which matters rather more here, where measuring the Python
    /// modules against the C++ ones is the entire point.
    ///
    /// This lives here rather than in main() because starting the interpreter
    /// is what causes numpy, and therefore OpenBLAS, to be loaded.  Existing
    /// settings are respected, so a caller who deliberately wants a threaded
    /// BLAS still gets one.
    void confineNestedThreadPools() {
      static constexpr char const* variables[] = {"OMP_NUM_THREADS",
                                                  "OPENBLAS_NUM_THREADS",
                                                  "MKL_NUM_THREADS",
                                                  "BLIS_NUM_THREADS",
                                                  "NUMEXPR_NUM_THREADS",
                                                  "VECLIB_MAXIMUM_THREADS"};

      for (char const* variable : variables) {
        setenv(variable, "1", 0);  // 0: do not overwrite what the caller set
      }
    }

    [[noreturn]] void fail(PyStatus const& status, char const* what) {
      std::string message = what;
      if (status.err_msg != nullptr) {
        message += std::string(": ") + status.err_msg;
      }
      throw std::runtime_error(message);
    }
  }  // namespace

  std::string PythonRuntime::home() { return envOr("EDM_PYTHON_HOME", EDM_PYTHON_HOME); }

  std::string PythonRuntime::sitePackages() { return envOr("EDM_PYTHON_SITE", EDM_PYTHON_SITE); }

  std::string PythonRuntime::scriptDir() { return envOr("EDM_PYTHON_DIR", EDM_PYTHON_DIR); }

  PythonRuntime& PythonRuntime::instance() {
    if (g_runtime == nullptr) {
      g_runtime = new PythonRuntime();
    }
    return *g_runtime;
  }

  bool PythonRuntime::running() noexcept { return g_runtime != nullptr; }

  PythonRuntime::Shutdown::~Shutdown() {
    delete g_runtime;
    g_runtime = nullptr;
  }

  PythonRuntime::PythonRuntime() {
    // Before the interpreter exists, and so before it can import numpy.
    confineNestedThreadPools();

    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);

    auto const set = [&](wchar_t** target, std::string const& value, char const* what) {
      if (value.empty()) {
        return;
      }
      const PyStatus status = PyConfig_SetBytesString(&config, target, value.c_str());
      if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        fail(status, what);
      }
    };

    set(&config.program_name, "edm", "PythonRuntime: cannot set program_name");
    set(&config.home, home(), "PythonRuntime: cannot set PYTHONHOME");
    set(&config.pycache_prefix,
        envOr("EDM_PYTHON_PYCACHE", EDM_PYTHON_PYCACHE),
        "PythonRuntime: cannot set the byte-code cache directory");

#ifdef Py_GIL_DISABLED
    // A measurement control, not a feature.  The isolated configuration
    // deliberately ignores the environment, so PYTHON_GIL does not reach the
    // interpreter; EDM_PYTHON_GIL=1 forces the GIL back on so that the same
    // job can be timed with and without it.  That comparison is the only way
    // to show what free-threading is actually worth on this workload, rather
    // than asserting it.
    if (std::string const gil = envOr("EDM_PYTHON_GIL", ""); not gil.empty()) {
      config.enable_gil = (gil == "0") ? 0 : 1;
    }
#endif

    const PyStatus status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
      fail(status, "PythonRuntime: interpreter initialisation failed");
    }

    // edm_core is an extension in the library directory rather than a module
    // built into this binary: it names every product type, so it belongs above
    // the product libraries rather than inside the framework.  Its directory
    // has to be on the path before it can be imported, and the raw C API is
    // used for that because nanobind is not usable until it is.
    {
      PyObject* const path = PySys_GetObject("path");  // borrowed
      PyObject* const directory = PyUnicode_FromString(STR(LIB_DIR));
      const bool ok = path != nullptr and directory != nullptr and PyList_Insert(path, 0, directory) == 0;
      Py_XDECREF(directory);
      if (not ok) {
        PyErr_Print();
        Py_Finalize();
        throw std::runtime_error("PythonRuntime: could not put " STR(LIB_DIR) " on sys.path");
      }
    }

    PyObject* const core = PyImport_ImportModule("edm_core");
    if (core == nullptr) {
      PyErr_Print();
      Py_Finalize();
      throw std::runtime_error("PythonRuntime: importing edm_core failed");
    }

    // Every shared object that calls nanobind keeps its own pointer to the
    // backend state, and it is filled when that object initialises a module.
    // This library initialises none -- edm_core.so does -- so it asks nanobind
    // to fill it here.  register_module() is the documented way in: it joins
    // the domain the extension already created rather than making a second
    // one, so there is still exactly one type registry in the process.  Until
    // this returns, no nanobind call in this library is safe.
    const bool registered = nb::register_module(nb::handle(core));
    Py_DECREF(core);
    if (not registered) {
      PyErr_Print();
      Py_Finalize();
      throw std::runtime_error("PythonRuntime: could not attach to edm_core's nanobind state");
    }

    try {
      nb::object sysPath = nb::module_::import_("sys").attr("path");
      if (std::string const site = sitePackages(); not site.empty()) {
        sysPath.attr("append")(site);
      }
      if (std::string const scripts = scriptDir(); not scripts.empty()) {
        sysPath.attr("insert")(0, scripts);
      }
    } catch (...) {
      Py_Finalize();
      throw;
    }
  }

  PythonRuntime::~PythonRuntime() { Py_Finalize(); }

  std::string PythonRuntime::version() const {
    return nb::cast<std::string>(nb::module_::import_("sys").attr("version"));
  }

  bool PythonRuntime::gilEnabled() const {
    nb::object sys = nb::module_::import_("sys");
    if (not nb::hasattr(sys, "_is_gil_enabled")) {
      return true;
    }
    return nb::cast<bool>(sys.attr("_is_gil_enabled")());
  }
}  // namespace edm
