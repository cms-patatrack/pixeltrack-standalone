module PluginFactory
    import ..ProductRegistry
    export add_plugin_module, create_plugin_module
    const plugin_registry_modules = Dict{String, Function}()

    function add_plugin_module(name::String, constructor::Function)
        plugin_registry_modules[name] = constructor
    end

    function create_plugin_module(name::String,reg::ProductRegistry)
        constructor = get(plugin_registry_modules, name, nothing)
        if constructor === nothing
            error("Plugin '$name' not found in registry.")
        end
        return constructor(reg)
    end

end



