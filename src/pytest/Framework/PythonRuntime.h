#ifndef PythonRuntime_h
#define PythonRuntime_h

#include <string>

namespace edm {
  /// The embedded CPython interpreter.
  ///
  /// It is started lazily, the first time a module implemented in Python is
  /// constructed, so an all-C++ configuration never pays for it.  Because the
  /// interpreter has to outlive every Python object held by a module, shutdown
  /// is explicit: main() declares a PythonRuntime::Shutdown before the
  /// EventProcessor, so that it is destroyed after it.
  class PythonRuntime {
  public:
    /// Starts the interpreter on the first call and returns it thereafter.
    /// Initialisation installs the built-in `edm_core` extension module and
    /// puts the backend's python/ directory and the venv's site-packages on
    /// sys.path.
    static PythonRuntime& instance();

    /// True once instance() has started an interpreter.
    [[nodiscard]] static bool running() noexcept;

    /// Scope guard finalising the interpreter, if one was ever started.
    struct Shutdown {
      Shutdown() = default;
      ~Shutdown();

      Shutdown(Shutdown const&) = delete;
      Shutdown& operator=(Shutdown const&) = delete;
    };

    /// e.g. "3.14.7 free-threading build (...)"
    [[nodiscard]] std::string version() const;

    /// False on a free-threaded interpreter that really runs without the GIL.
    /// A true here means every Python module in the job is serialised, and any
    /// throughput measured against the C++ path is meaningless.
    [[nodiscard]] bool gilEnabled() const;

    /// Compile-time defaults, each overridable through the environment
    /// variable of the same name.
    [[nodiscard]] static std::string home();
    [[nodiscard]] static std::string sitePackages();
    [[nodiscard]] static std::string scriptDir();

  private:
    PythonRuntime();
    ~PythonRuntime();

    PythonRuntime(PythonRuntime const&) = delete;
    PythonRuntime& operator=(PythonRuntime const&) = delete;
  };
}  // namespace edm

#endif
