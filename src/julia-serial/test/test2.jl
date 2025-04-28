# function test!(x::Vector{Int64})
#     # Directly use a lambda function that does not capture external state
#     increment! = (v) -> x[v] += 1

#     for i in 1:10
#         increment!(1)  # Note: `1` here is used for illustration; adjust as needed.
#     end  
# end

# # Initialize the vector
# x = Vector{Int64}(undef, 10)

# # Measure the time for the modified function
# @time test!(x)


function factoriall(::Val{N}) where N
    if N == 1
        return 1
    end
    return N*factoriall(Val{N-1}())
end
function factoriall(x)
    if x == 1
        return 1
    end
    return x*factoriall(x-1)
end

@btime factoriall(Val{15}())
@btime factoriall(15)
@btime factorial(15)