using Pkg

Pkg.activate(".")


deps = [
    "ArgParse",
    "BenchmarkTools",
    "CUDA",
    "Dagger",
    "Dates",
    "Printf",
    "InteractiveUtils",
    "DataStructures",
    "LinearAlgebra",
    "Statistics",
    "Test"
]

for dep in deps
    try
        println("Ensuring dependency: $dep")
        Pkg.add(dep)
    catch e
        println("Failed to add $dep: $e")
    end
end

Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()