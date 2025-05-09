using Test
using Aqua
using Patatrack

@testset "Aqua.jl" begin
    Aqua.test_all(Patatrack; stale_deps=(; ignore=[:Distributions]))
    # Distributions are used in test/
end