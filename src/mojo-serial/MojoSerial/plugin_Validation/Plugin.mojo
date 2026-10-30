# The root of pluginValidation.so. C++ registers each module with
# DEFINE_FWK_MODULE or DEFINE_FWK_EVENTSETUP_MODULE, a static object whose
# constructor runs when the library is loaded; Mojo has no global variables, so
# the library exports for each module a function that registers it, called by
# the PluginManager (bin/PluginManager.mojo) when it loads the library. Only the
# functions exported by this file, the root of the build, are part of the library.

from MojoSerial.Framework.PluginFactory import define_fwk_module_with_globals
from MojoSerial.Framework.Wrapper import ErasedPointer
from MojoSerial.plugin_Validation.CountValidator import CountValidator
from MojoSerial.plugin_Validation.HistoValidator import HistoValidator


@export("edmplugin_EDProducer_CountValidator")
def register_CountValidator(registry: ErasedPointer) abi("C"):
    define_fwk_module_with_globals[CountValidator](registry)


@export("edmplugin_EDProducer_HistoValidator")
def register_HistoValidator(registry: ErasedPointer) abi("C"):
    define_fwk_module_with_globals[HistoValidator](registry)
