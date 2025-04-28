module eigenSOA
    using StaticArrays:MArray
    is_power_2(v::Integer) = ((v != 0) && ((v & (v-1)) == 0)) # For positive integers
    using ..Patatrack:Quality,bad,dup,loose,strict,tight,highPurity
    """
    A structure that holds a static array of objects of type `Scalar`.

    # Type Parameters
    - `Scalar`: The type of elements stored in the array.
    - `S`: The size of the static array, which must be a power of two and the total size of the array in bytes must be a multiple of 128.

    # Fields
    - `data::MArray{Tuple{S}, Scalar}`: A mutable array that stores objects of type `Scalar`.

    # Constructor
    The constructor performs the following checks:
    - `S` must be an integer.
    - `S` must be a power of two.
    - The total size of the data (calculated as `sizeof(Scalar) * S`) must be a multiple of 128 bytes.
    """
    struct ScalarSOA{Scalar,S}
        data::MArray{Tuple{S},Scalar}
        function ScalarSOA{Scalar,S}() where {Scalar,S}
            @assert typeof(S) <: Integer
            @assert is_power_2(S) "SOA Stride not power of 2"
            @assert sizeof(Scalar)*S % 128 == 0
            new{Scalar,S}(MArray{Tuple{S},Scalar}(undef))
        end
    end
    
    Base.getindex(self::ScalarSOA,i::Integer) = self.data[i]
    Base.setindex!(self::ScalarSOA,value,i::Integer) = self.data[i] = value
    Base.setindex!(self::ScalarSOA,value,i::Quality) = self.data[i] = value


end