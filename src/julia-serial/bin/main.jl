# The main function is artificially split into several parts to accommodate 
# for lack of common entrypoint for julia and julia compilers:
# - bin/main.jl is a script for running the application and defines
#   - `@main(args)::Cint` - the entrypoint for julia
#   - `main(argc, argv)::Cint` - the entrypoint for the juliac compiler
# src/entrypoints.jl defines the main logic and entrypoint for PackageCompiler:
# - `julia_serial_main()::Cint` - the entrypoint for PackageCompiler, must be part of a module
# - `julia_serial_main(raw_args)` - is the main logic called by all the entrypoints

import Patatrack

# Entrypoint for julia
function (@main)(args)::Cint
    return Patatrack.julia_serial_main(args)
end

# Entrypoint for juliac compiler
Base.@ccallable function main(argc::Cint, argv::Ptr{Ptr{UInt8}})::Cint
    args = [unsafe_string(unsafe_load(argv, i)) for i in 2:argc]
    return main(args)
end
