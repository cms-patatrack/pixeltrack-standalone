mutable struct PreAllocMatrix{T} <: AbstractMatrix{T}
    data::Matrix{T}
    col_lengths::Vector{UInt32}
    ncol::UInt32
end

function PreAllocMatrix{T}(nrowmax, ncolmax) where T
    PreAllocMatrix(Matrix{T}(undef, convert(UInt32, nrowmax), convert(UInt32, ncolmax)),
                   zeros(UInt32, ncolmax),
                   zero(UInt32))
end

function extend!(self::PreAllocMatrix) 
    col = size(self)[2]
    if self.ncol == col
        return UInt32(col) 
    end
    self.ncol+=1
    return self.ncol
end

Base.getindex(m::PreAllocMatrix, inds...) = getindex(m.data, inds...)
function Base.setindex!(m::PreAllocMatrix, value, irow, icol)
    #Expand matrix to include element (irow, icol):
    m.nrows >= irow || (m.nrows = irow)
    m.rowlengths[irow] >= icol || (m.rowlengths[irow] = icol)
    setindex!(m.data, value, irow, icol)
end
Base.size(m::PreAllocMatrix) = size(m.data)

# 1) If someone calls M[:, c], return a view of the "used" rows in column c
function Base.getindex(m::PreAllocMatrix{T}, ::Colon, c::Integer) where {T}
    # @boundscheck @assert 1 <= c <= m.ncol  "Column out of range"
    used_len = m.col_lengths[c]
    # Return a SubArray covering rows [1:used_len, c]
    return @view m.data[1:used_len, c]
end

function Base.push!(m::PreAllocMatrix{T},col::Integer,val::T) where T 
    row_max = size(m.data,1)
    row_index = m.col_lengths[col]
    if row_index == row_max
        return row_index
    end
    m.col_lengths[col] += 1 
    row_index += UInt32(1) 
    m.data[row_index,col] = val 
    return row_index
end

function length(m::PreAllocMatrix{T},col::Integer) where T 
    m.col_lengths[col]
end

function reset!(m::PreAllocMatrix{T}) where T
    m.ncol = 0 
    m.col_lengths .= 0 
end

nrows(m::PreAllocMatrix) = m.nrows
rowlength(m::PreAllocMatrix, irow) = m.rowlengths[irow]
# CAHitNTupletGeneratorKernels()
# @allocations a = CAHitNTupletGeneratorKernels()
# # -> 5 allocations

# # use @timev instead of @allocations to get more information:
# @timev a = CAHitNTupletGeneratorKernels();
# for i in 1:30
#     for j in 1:(2*i)
#         a.device_is_outer_hit_of_cell[i,j] = i*j
#     end
# end
# success = true
# for i in 1:30
#     success &= rowlength(a.device_is_outer_hit_of_cell, i) == 2*i
# end         
# @test success   