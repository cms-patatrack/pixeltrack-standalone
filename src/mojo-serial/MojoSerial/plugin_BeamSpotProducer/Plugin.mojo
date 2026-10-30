# The root of pluginBeamSpotProducer.so. C++ registers each module with
# DEFINE_FWK_MODULE or DEFINE_FWK_EVENTSETUP_MODULE, a static object whose
# constructor runs when the library is loaded; Mojo has no global variables, so
# the library exports for each module a function that registers it, called by
# the PluginManager (bin/PluginManager.mojo) when it loads the library. Only the
# functions exported by this file, the root of the build, are part of the library.

from MojoSerial.Framework.PluginFactory import define_fwk_module
from MojoSerial.Framework.ESPluginFactory import define_fwk_eventsetup_module
from MojoSerial.Framework.Wrapper import ErasedPointer
from MojoSerial.plugin_BeamSpotProducer.BeamSpotESProducer import BeamSpotESProducer
from MojoSerial.plugin_BeamSpotProducer.BeamSpotToPOD import BeamSpotToPOD


@export("edmplugin_ESProducer_BeamSpotESProducer")
def register_BeamSpotESProducer(registry: ErasedPointer) abi("C"):
    define_fwk_eventsetup_module[BeamSpotESProducer](registry)


@export("edmplugin_EDProducer_BeamSpotToPOD")
def register_BeamSpotToPOD(registry: ErasedPointer) abi("C"):
    define_fwk_module[BeamSpotToPOD](registry)
