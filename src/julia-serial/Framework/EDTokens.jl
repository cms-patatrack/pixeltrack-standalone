const UNINITIALIZED_VALUE = UInt32(0xFFFFFFFF)

"""
EDGetToken with type Parameter
"""
struct EDGetTokenT{T}
    value::UInt32
    EDGetTokenT{T}() where T = new(UNINITIALIZED_VALUE)
    function EDGetTokenT{T}(x::Integer) where T
        new(x)
    end 
end

function index(token::EDGetTokenT{T}) where T 
    return token.value
end

function isUninitialized(token::EDGetTokenT{T}) where T
    return token.value == UNINITIALIZED_VALUE
end

"""
EDGetToken without type Parameter
"""
struct EDGetToken
    value::UInt32
    EDGetToken() = new(UNINITIALIZED_VALUE) # Default Constructor

    function EDGetToken(token::EDGetTokenT{T}) where T # Takes an EDGetToken with a type Parameter and initializes its index to the current EDGetToken
        new(token.value)
    end
end

function index(token::EDGetToken)
    return token.value
end

function isUninitialized(token::EDGetToken)
    return token.value == UNINITIALIZED_VALUE
end

struct EDPutTokenT{T}
    value::UInt32
    EDPutTokenT{T}() where T = new(UNINITIALIZED_VALUE)

    function EDPutTokenT{T}(x::UInt32) where T
        new(x)
    end

    function EDPutTokenT(token::EDPutTokenT{T}) where T # Takes an EDPutToken with a type Parameter and initializes its index to the current EDPutToken
        new{T}(token.value)
    end
end

function index(token::EDPutTokenT{T})::UInt32 where T
    return token.value
end

function isUninitialized(token::EDPutTokenT{T})::Bool where T
    return token.value == UNINITIALIZED_VALUE
end


struct EDPutToken
    value::UInt32

    EDPutToken() = new(UNINITIALIZED_VALUE) # Default Constructor

    function EDPutToken(token::EDPutTokenT{T}) where T # Takes an EDPutToken with a type Parameter and initializes its index to the current EDPutToken
        new(token.value)
    end

end


function index(token::EDPutToken)::UInt32
    return token.value
end

function isUninitialized(token::EDPutToken)::Bool
    return token.value == UNINITIALIZED_VALUE
end

