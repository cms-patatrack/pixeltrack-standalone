mutable struct SimpleVector{T}
    m_size::Int
    m_capacity::Int
    m_data:: Vector{T}
    function SimpleVector{T}(capacity::Int,data::Vector{T}) where T
        new(0,capacity,data)
    end
end

function Base.push!(self::SimpleVector{T},element::T) where T
    self.m_size +=1 
    if(self.m_size <= self.m_capacity)
        m_data[self.m_size] = element
        return self.m_size-1
    else
        m_size-=1
        return -1
    end
end

function extend!(self::SimpleVector{T},size::Integer = 1) where T # check feels wrong
    self.m_size += size
    if(self.m_size <= self.m_capacity)
        return self.m_size - size
    else
        self.m_size -= size
        return -1 
    end
end

function shrink!(self::SimpleVector{T},size::Integer = 1) where T
    previous_size = self.m_size
    if(previous_size >= size)
        self.m_size -= size
        return self.m_size
    else
        return -1
    end
end

Base.empty(self::SimpleVector{T}) where T = self.m_size <= 0
full(self::SimpleVector{T}) where T = self.m_size >= self.m_capacity
Base.getindex(self::SimpleVector{T},i::Integer) where T = self.m_data[i]
reset!(self::SimpleVector{T}) where T = (self.m_size = 0)
Base.length(self::SimpleVector{T}) where T = self.m_size
capacity(self::SimpleVector{T}) where T = self.m_capacity
data(self::SimpleVector{T}) where T = self.m_data
set_data(self::SimpleVector{T},data::Vector{T}) where T = self.m_data = data


 
