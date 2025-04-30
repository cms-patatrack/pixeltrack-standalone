using Test
using Aqua
using Patatrack

@testset "Aqua.jl" begin
    Aqua.test_all(Patatrack; stale_deps=(; ignore=[:ArgParse, :BenchmarkTools, # used in main
        :Distributions])) # used in test/
end