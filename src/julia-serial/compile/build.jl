using PackageCompiler
using ArgParse
import Pkg

function parse_args(args)
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--source-dir"
        help = "Directory containing the source files"
        arg_type = String
        default = joinpath(splitdir(@__DIR__) |> first)

        "--output-dir", "-o"
        help = "Directory to save the compiled library"
        arg_type = String
        default = "JuliaSerialCompiled"

        "--juliac"
        help = "Use juliac compiler"
        action = :store_true
    end

    return ArgParse.parse_args(args, s)
end

function packagecompiler_compile(source_dir, output_dir)
    Pkg.activate(source_dir)
    Pkg.instantiate()
    return @elapsed PackageCompiler.create_app(source_dir, output_dir;
                                               executables = ["julia-serial" => "julia_serial_main"],
                                               precompile_execution_file = [joinpath(source_dir, "bin", "main.jl")],
                                               incremental = false,
                                               filter_stdlibs = false,
                                               force = true)
end

function juliac_compile(source_dir, output_dir)
    julia_path = joinpath(Sys.BINDIR, Base.julia_exename())
    juliac_path = joinpath(Sys.BINDIR, "..", "share", "julia", "juliac", "juliac.jl")
    main_path = joinpath(source_dir, "bin", "main.jl")
    bin_dir = joinpath(output_dir, "bin")
    mkpath(bin_dir)
    output_bin = joinpath(bin_dir, "julia-serial")
    cmd = "$(julia_path) --project=$(source_dir) $(juliac_path) --experimental --trim=no --output-exe $(output_bin) $(main_path)"
    @info "Running command: $cmd"
    return @elapsed run(`$(split(cmd))`)
end

function (@main)(args)
    parsed_args = parse_args(args)
    source_dir = parsed_args["source-dir"]
    output_dir = parsed_args["output-dir"]

    @info "Compiling package from $source_dir"
    @info "Creating output in $output_dir"

    compilation_time = if parsed_args["juliac"]
        @info "Compiling with juliac"
        juliac_compile(source_dir, output_dir)
    else
        @info "Compiling with PackageCompiler"
        packagecompiler_compile(source_dir, output_dir)
    end
    @info "Compiled in $(compilation_time) seconds"
end
