abstract type WrapperBase end

struct Wrapper{T} <: WrapperBase
    obj::T
end

struct Event
    streamId::Int
    eventId::Int
    products::Vector{WrapperBase}  # Union type to allow for null elements

    function Event(streamIDD::Integer,eventIDD::Integer, reg::ProductRegistry)
        return new(streamIDD,eventIDD,Vector{WrapperBase}(undef,length(reg))) # length(reg)
    end
end

Event(reg::ProductRegistry) = Event(0,0,reg)

# Accessor functions for Event
streamID(event::Event) = event.streamId
eventID(event::Event) = event.eventId

# Function to retrieve a product of type T from Event
function Base.get(event::Event, token::EDGetTokenT{T})::T where T
    wrapper = event.products[token.value]
    return wrapper.obj
end

# Function to insert a product of type T into Event
function emplace(event::Event, token::EDPutTokenT{T}, args...) where T
    event.products[token.value] = Wrapper{T}(args...)
end

########################################################################


abstract type ESWrapperBase end

struct ESWrapper{T} <: ESWrapperBase
    obj::T
end

mutable struct EventSetup
    typeToProduct::Dict{DataType, ESWrapperBase}

    function EventSetup()
        return new(Dict{DataType, ESWrapperBase}())
    end
end

function Base.put!(es::EventSetup, prod::T) where T
    es.typeToProduct[T] = ESWrapper(prod)
end

function Base.get(es::EventSetup, ::Type{T}) where T
    if haskey(es.typeToProduct, T)
        return es.typeToProduct[T].obj
    else
        throw(ErrorException("Product of type $(T) is not produced"))
    end
end

