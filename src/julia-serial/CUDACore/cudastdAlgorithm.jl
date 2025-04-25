# Module definition for CUDA utilities and algorithms
module HeterogeneousCoreCUDAUtilitiesCudastdAlgorithm

# Include the CUDA compatibility definitions
include("../CUDACore/cudaCompat.jl")
using .heterogeneousCoreCUDAUtilitiesInterfaceCudaCompat

# Nested module for standard CUDA algorithms
module cuda_std

    """
    Struct representing a comparator for less-than operation.

    ## Fields
    - `T`: The type of elements to be compared.
    """
    struct Less{T}
    end

    """
    Comparator function for `Less` struct to compare two values.
    
    ## Arguments
    - `op::Less{T}`: The comparator object.
    - `lhs::T`: The left-hand side value.
    - `rhs::T`: The right-hand side value.
    
    ## Returns
    - `Bool`: `true` if `lhs` is less than `rhs`, otherwise `false`.
    """
    function (op::Less{T})(lhs::T, rhs::T) where T
        return lhs < rhs
    end

    """
    Function to find the lower bound in a range using binary search.
    
    ## Arguments
    - `first::Int`: The beginning of the range.
    - `last::Int`: The end of the range.
    - `value::T`: The value to compare against.
    - `comp::Less{T}`: The comparator of type `Less`.
    
    ## Returns
    - `Int`: The iterator pointing to the lower bound.
    """
    function lower_bound(first::Int, last::Int, value::T, comp::Less{T}) where T
        count = last - first
        while count > 0
            it = first
            step = count ÷ 2  # Use integer division to avoid potential type issues
            it = it + step
            if comp(it, value)
                first = it + 1
                count = count - (step + 1)
            else
                count = step
            end
        end
        return first
    end

    """
    Function to find the upper bound in a range using binary search.
    
    ## Arguments
    - `first::Int`: The beginning of the range.
    - `last::Int`: The end of the range.
    - `value::T`: The value to compare against.
    - `comp::Less{T}`: The comparator of type `Less`.
    
    ## Returns
    - `Int`: The iterator pointing to the upper bound.
    """
    function upper_bound(first::Int, last::Int, value::T, comp::Less{T}) where T
        count = last - first
        while count > 0
            step = count ÷ 2  # Use integer division to avoid potential type issues
            it = first + step  # Ensure `it` is properly initialized
            if !comp(value, it)
                first = it + 1
                count = count - (step + 1)
            else
                count = step
            end
        end
        return first  # Ensure a return value to avoid issues
    end
    
    """
    Function to perform a binary search in a range.
    
    ## Arguments
    - `first::Int`: The beginning of the range.
    - `last::Int`: The end of the range.
    - `value::T`: The value to search for.
    - `comp::Less{T}`: The comparator of type `Less`.
    
    ## Returns
    - `Int`: The iterator pointing to the found element, or `last` if not found.
    """
    function binary_find(first::Int, last::Int, value::T, comp::Less{T}) where T
        first = cuda_std::lower_bound(first, last, value, comp)
        return first != last && !comp(value, first) ? first : last
    end

end  # End of nested module `cuda_std`
end  # End of module `HeterogeneousCoreCUDAUtilitiesCudastdAlgorithm`
