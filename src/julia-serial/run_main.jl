# println("Starting run_main.jl…")

include("main.jl")
# ARGS = ["--maxEvents", "10", "--warmupEvents", "5"]

# println("Calling julia_main()…")
exit_code = julia_main()
# println("julia_main() finished with exit code: ", exit_code)
# println("Exiting run_main.jl…")
exit(exit_code)
