# The main function is artificially split into several parts to accommodate 
# for lack of common entrypoint for julia and julia compilers:
# - bin/main.jl is a script for running the application and defines
#   - `@main(args)::Cint` - the entrypoint for julia and juliac/JuliaC.jl compiler
# src/entrypoints.jl defines the main logic and entrypoint for PackageCompiler:
# - `julia_serial_main()::Cint` - the entrypoint for PackageCompiler, must be part of a module
# - `julia_serial_main(raw_args)` - is the main logic called by all the entrypoints

import Patatrack

# Entrypoint for julia and juliac/JuliaC.jl
function @main(args)::Cint
    return Patatrack.julia_serial_main(args)
end
