from std.pathlib import Path
from std.sys import get_defined_string, is_defined

from MojoSerial.Framework.ESPluginFactory import ESProducerPtr, ESRegistry
from MojoSerial.Framework.PluginFactory import Registry, Worker
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.Framework.Wrapper import ErasedPointer
from MojoSerial.bin.SharedLibrary import SharedLibrary

# The directory of the plugin libraries and of plugins.txt, set by the Makefile.
comptime LIB_DIR = get_defined_string["LIB_DIR", ""]()

# Each plugin library (MojoSerial/plugin_<X>/Plugin.mojo, built as
# pluginX.so) exports, for each of its modules, a function with the C ABI that
# registers the module with the plugin factory of the framework:
#   edmplugin_EDProducer_<module>(registry: Registry)
#   edmplugin_ESProducer_<module>(registry: ESRegistry)
# The Makefile lists these symbols in plugins.txt, as "<module> <library>".
comptime ED_SYMBOL_PREFIX = "edmplugin_EDProducer_"
comptime ES_SYMBOL_PREFIX = "edmplugin_ESProducer_"


# C++: edmplugin::PluginManager.
#
# C++ registers the modules of a library from the constructors of static
# objects, which run when the library is loaded, into registries that are static
# variables. Mojo has neither: the PluginManager owns the registries, and calls
# the registration functions of all the modules of a library when it loads it.
#
# C++ loads the plugins in the EventProcessor, before starting the streams; here
# each stream has its own EventProcessor, so main loads them before starting the
# streams, and the streams only create the modules. Nothing here is thread safe.
struct PluginManager(Movable):
    var _pluginToLibrary: Dict[String, String]
    var _loadedPlugins: Dict[String, SharedLibrary]
    var _registry: Registry
    var _esRegistry: ESRegistry

    # `streams` is the number of streams that share the globals of the plugins.
    def __init__(out self, streams: Int) raises:
        comptime assert is_defined["LIB_DIR"](), "LIB_DIR undefined"
        self._pluginToLibrary = Dict[String, String]()
        self._loadedPlugins = Dict[String, SharedLibrary]()
        self._registry = Registry(streams)
        self._esRegistry = ESRegistry()

        var pluginMap = Path(LIB_DIR) / "plugins.txt"
        var lines = pluginMap.read_text().splitlines()
        for line in lines:
            var words = line.split()
            if len(words) != 2:
                continue
            self._pluginToLibrary[String(words[0])] = String(words[1])

    def load(mut self, pluginName: String) raises:
        if pluginName not in self._pluginToLibrary:
            raise "RuntimeError: plugin " + pluginName + " is not in " + LIB_DIR + "/plugins.txt"
        var libName = self._pluginToLibrary[pluginName]
        if libName in self._loadedPlugins:
            return

        var library = SharedLibrary(LIB_DIR + "/" + libName)
        var modules = List[String]()
        for entry in self._pluginToLibrary.items():
            if entry.value == libName:
                modules.append(entry.key)
        for module in modules:
            if library.hasSymbol(ED_SYMBOL_PREFIX + module):
                library.call(ED_SYMBOL_PREFIX + module, _address_of(self._registry))
            else:
                library.call(ES_SYMBOL_PREFIX + module, _address_of(self._esRegistry))
        self._loadedPlugins[libName] = library^

    # C++: PluginFactory::create()
    def create(self, name: String, mut reg: ProductRegistry) raises -> Worker:
        return self._registry.create(name, reg)

    # C++: ESPluginFactory::create()
    def createES(self, name: String, datadir: Path) raises -> ESProducerPtr:
        return self._esRegistry.create(name, datadir)


def _address_of[T: AnyType](mut obj: T) -> ErasedPointer:
    return (
        Pointer(to=obj)
        .unsafe_bitcast[NoneType]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )
