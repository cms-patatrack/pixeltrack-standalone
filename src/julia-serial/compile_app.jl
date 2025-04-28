using PackageCompiler

create_app(".", "compile/bin";
    executables=["run_main.jl" => "julia_main"],
    precompile_execution_file="run_main.jl",
    cpu_target="native",
    force=true,
    filter_stdlibs=false,
    include_lazy_artifacts=true,
    include_transitive_dependencies=true  #
)

println("Compilation complete! Executable is in compile/bin/julia_main.exe")
