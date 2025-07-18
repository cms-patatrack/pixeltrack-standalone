# The main function is artificially split into several parts to accommodate 
# for lack of common entrypoint for julia and julia compilers:
# - bin/main.jl is a script for running the application and defines
#   - `@main(args)::Cint` - the entrypoint for julia
#   - `main(argc, argv)::Cint` - the entrypoint for the juliac compiler
# src/entrypoints.jl defines the main logic and entrypoint for PackageCompiler:
# - `julia_serial_main()::Cint` - the entrypoint for PackageCompiler, must be part of a module
# - `julia_serial_main(raw_args)::Cint` - is the main logic called by all the entrypoints


using BenchmarkTools
using ArgParse
using Printf
using Dates

function print_help()
    println("""
    Usage: julia main.jl [--numberOfStreams NS] [--warmupEvents WE] [--maxEvents ME] [--runForMinutes RM]
           [--data PATH] [--validation] [--histogram] [--empty]

    Options:
      --numberOfStreams        Number of concurrent events (default 0 = numberOfThreads)
      --warmupEvents          Number of events to process before starting the benchmark (default 0)
      --maxEvents             Number of events to process (default -1 for all events in the input file)
      --runForMinutes         Continue processing the set of 1000 events until this many minutes have passed
                             (default -1 for disabled; conflicts with --maxEvents)
      --data                  Path to the 'data' directory (default 'data' in the directory of the executable)
      --validation           Run (rudimentary) validation at the end
      --histogram            Produce histograms at the end
      --empty               Ignore all producers (for testing only)
    """)
end

function parse_commandline(args)
    s = ArgParseSettings(description="CMS Julia Event Processing")
    s.add_help = false  # Disable the default --help option

    @add_arg_table! s begin
        "--numberOfStreams"
        help = "Number of concurrent events"
        arg_type = Int
        default = 0
        "--warmupEvents"
        help = "Number of warmup events"
        arg_type = Int
        default = 0
        "--maxEvents"
        help = "Maximum number of events"
        arg_type = Int
        default = 1000
        "--runForMinutes"
        help = "Run for specified minutes"
        arg_type = Int
        default = -1
        "--data"
        help = "Path to data directory"
        arg_type = String
        default = "data"
        "--validation"
        help = "Enable validation"
        action = :store_true
        "--histogram"
        help = "Enable histogram"
        action = :store_true
        "--empty"
        help = "Ignore all producers"
        action = :store_true
        "-h", "--help"
        help = "Show this help message"
        action = :store_true
    end

    return parse_args(args, s)
end

# Main logic used by the entrypoints
function julia_serial_main(raw_args)::Cint
    # println("Hello from julia_main()!")
    args = parse_commandline(raw_args)

    if args["help"]
        print_help()
        return 0
    end

    # Validate arguments
    if args["maxEvents"] >= 0 && args["runForMinutes"] >= 0
        println("Got both --maxEvents and --runForMinutes, please give only one of them")
        return 1
    end

    # Set number of streams
    num_streams = args["numberOfStreams"]
    if num_streams == 0
        num_streams = Threads.nthreads()
    end

    # Validate data directory
    data_dir = args["data"]
    if !isdir(data_dir)
        println("Data directory '$(data_dir)' does not exist")
        return 1
    end

    # Initialize modules
    ed_modules = String[]
    es_modules = String[]

    if !args["empty"]
        ed_modules = [
            "BeamSpotToPOD",
            "SiPixelRawToClusterCUDA",
            "SiPixelRecHitCUDA",
            "CAHitNtupletCUDA",
            "PixelVertexProducerCUDA"
        ]
        es_modules = [
            "BeamSpotESProducer",
            "SiPixelFedCablingMapGPUWrapperESProducer",
            "SiPixelGainCalibrationForHLTGPUESProducer",
            "PixelCPEFastESProducer"
        ]

        if args["validation"]
            push!(ed_modules, "CountValidator")
        end
        if args["histogram"]
            push!(ed_modules, "HistoValidator")
        end
    end

    # Print processing information
    if args["runForMinutes"] < 0
        print("Processing $(args["maxEvents"]) events,")
    else
        print("Processing for about $(args["runForMinutes"]) minutes,")
    end

    if args["warmupEvents"] > 0
        print(" after $(args["warmupEvents"]) events of warm up,")
    else
        args["warmupEvents"] = args["maxEvents"]
    end
    println(" with $num_streams concurrent events and $(Threads.nthreads()) threads.")

    # Initialize EventProcessor
    ev = EventProcessor(
        num_streams,
        ed_modules,
        es_modules,
        data_dir,
        args["validation"],
        args["warmupEvents"]
    )
    # Warm up
    # try
    if args["warmupEvents"] > 0
        println("Warming up...")
        @time warm_up(ev, args["maxEvents"])
        GC.gc() # garbage collect so main processing starts without garbage from warmup
    end

    # Main processing
    println("Processing...")
    start_time = now()
    cpu_start = time_ns()

    @time run_processor(ev)

    cpu_end = time_ns()
    end_time = now()

    # Calculate timing
    elapsed_seconds = Dates.value(end_time - start_time) / 1000
    cpu_time = (cpu_end - cpu_start) / 1e9

    # Report results
    processed_events = ev.source.numEvents[] - 1  # Adjust this based on your actual event counter
    throughput = processed_events / elapsed_seconds
    cpu_usage = (cpu_time / elapsed_seconds / Threads.nthreads()) * 100

    @printf("Processed %d events in %.6e seconds, throughput %.2f events/s, CPU usage per thread: %.1f%%\n",
        processed_events, elapsed_seconds, throughput, cpu_usage)

    # catch e
    #     println("\n----------\nCaught exception:")
    #     println(e)
    #     return 1
    # end
    println("Finished processing events.")

    return 0
end

# Entrypoint for PackageCompiler compiler
function julia_serial_main()::Cint
    return julia_serial_main(ARGS)
end

