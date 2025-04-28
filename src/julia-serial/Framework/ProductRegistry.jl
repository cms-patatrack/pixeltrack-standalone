mutable struct ProductRegistry
    current_module_index::Int
    consumed_modules::Set{UInt}
    type_to_index::Dict{DataType, Tuple{UInt, UInt}} # module Index for first pair, product Index for second pair value
end
ProductRegistry() = ProductRegistry(0,Set{UInt}(),Dict{DataType, Tuple{UInt, UInt}}())

function produces(registry::ProductRegistry,::Type{T}) where T
    ind::UInt32 = length(registry.type_to_index) + 1
    if haskey(registry.type_to_index, T)
        throw(ErrorException("Product of type $T already exists"))
    end
    registry.type_to_index[T] = (registry.current_module_index, ind)
    return EDPutTokenT{T}(ind)
end

function consumes(registry::ProductRegistry,::Type{T}) where T
    if !haskey(registry.type_to_index, T)
        throw(ErrorException("Product of type $T is not produced"))
    end
    indices = registry.type_to_index[T]
    push!(registry.consumed_modules, indices[1])
    return EDGetTokenT{T}(indices[2])
end

function length(registry::ProductRegistry)::Int
    return length(registry.type_to_index)
end

function begin_module_construction(registry::ProductRegistry, i::Int)
    registry.current_module_index = i
    empty!(registry.consumed_modules)
end

function consumed_modules(registry::ProductRegistry)
    return registry.consumed_modules
end

