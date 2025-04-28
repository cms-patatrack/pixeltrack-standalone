struct SimpleAtomicHisto
    data::Vector{Int32}
    min::Float32
    max::Float32
    function SimpleAtomicHisto(n_bins::Integer,min::Float32,max::Float32)
        new(zeros(Int32,n_bins + 2),min,max)
    end
end

function fill!(self::SimpleAtomicHisto, value)
    i = 0
    n_bins = length(self.data) - 2
    if value < self.min
        i = 1 # Julia indices start at 1
    elseif value >= self.max
        i = length(self.data)
    else # min up to max exclusive
        i = Int(floor((value - self.min) / (self.max - self.min) * (n_bins)))
        # Handle rounding near maximum
        if i == n_bins
            i -= 1
        end
        if !(i >= 0 && i < n_bins)
            throw(ArgumentError("fill!($value): i = $i, min = $(self.min), max = $(self.max), nbins = $n_bins"))
        end
        i += 2
    end
    @assert i >= 1 && i <= length(self.data)
    self.data[i] += 1
end

function dump(io::IO, self::SimpleAtomicHisto)
    println(io," min: ", self.min, " max: ", self.max)
    for item ∈ self.data
        print(io, " ", item)
    end
    println(io)
end

# Overload `show` for printing
Base.show(io::IO, histo::SimpleAtomicHisto) = dump(io, histo)
