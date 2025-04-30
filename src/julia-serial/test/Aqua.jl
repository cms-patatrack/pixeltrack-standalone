using Test
using Aqua
using Patatrack

@testset "Aqua.jl" begin
    Aqua.test_stale_deps(Patatrack; ignore = [:ArgParse, :BenchmarkTools, # used in main
                             :Distributions]) # used in test/
    Aqua.test_deps_compat(Patatrack)                    
end