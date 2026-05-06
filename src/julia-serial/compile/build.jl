using PackageCompiler
using ArgParse
import Pkg
import JuliaC

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
    img = JuliaC.ImageRecipe(
        output_type="--output-exe",
        file=joinpath(source_dir, "bin", "main.jl"),
        project=source_dir,
        trim_mode="no",
        add_ccallables=false,
        verbose=false
    )

    link = JuliaC.LinkRecipe(
        image_recipe=img,
        outname="julia-serial",
    )

    bun = JuliaC.BundleRecipe(
        link_recipe=link,
        output_dir=output_dir, # or `nothing` to skip bundling
    )

    return @elapsed begin
        JuliaC.compile_products(img)
        JuliaC.link_products(link)
        JuliaC.bundle_products(bun)
    end
end

function @main(args)
    parsed_args = parse_args(args)
    source_dir = parsed_args["source-dir"]
    output_dir = parsed_args["output-dir"]

    @info "Compiling package from $source_dir"
    @info "Creating output in $output_dir"

    compilation_time = if parsed_args["juliac"]
        @info "Compiling with JuliaC.jl"
        juliac_compile(source_dir, output_dir)
    else
        @info "Compiling with PackageCompiler.jl"
        packagecompiler_compile(source_dir, output_dir)
    end
    @info "Compiled in $(compilation_time) seconds"
end
