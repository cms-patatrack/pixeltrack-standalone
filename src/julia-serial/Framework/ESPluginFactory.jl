module ESPluginFactory

    export add_plugin, create_plugin
    const plugin_registry = Dict{String, Function}()

    function add_plugin(name::String, constructor::Function)
        plugin_registry[name] = constructor
    end

    function create_plugin(name::String, datadir::String)
        constructor = get(plugin_registry, name, nothing)
        if constructor === nothing
            error("Plugin '$name' not found in registry.")
        end
        return constructor(datadir)
    end

end



