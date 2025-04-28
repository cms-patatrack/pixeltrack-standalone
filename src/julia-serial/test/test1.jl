using BenchmarkTools
# const x = [fill(0,100) for i ∈ 1:1000]
function test(x)
    r = 0
    for i ∈ 1 :100
        temp = length(x[i])
        # print(temp)
        r = temp
    end
    print(r)
end



include("test2.jl")
include("test2.jl")
print(5)