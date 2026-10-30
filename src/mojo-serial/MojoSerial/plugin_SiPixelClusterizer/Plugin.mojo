# The root of pluginSiPixelClusterizer.so. C++ registers each module with
# DEFINE_FWK_MODULE or DEFINE_FWK_EVENTSETUP_MODULE, a static object whose
# constructor runs when the library is loaded; Mojo has no global variables, so
# the library exports for each module a function that registers it, called by
# the PluginManager (bin/PluginManager.mojo) when it loads the library. Only the
# functions exported by this file, the root of the build, are part of the library.

from MojoSerial.Framework.PluginFactory import define_fwk_module
from MojoSerial.Framework.ESPluginFactory import define_fwk_eventsetup_module
from MojoSerial.Framework.Wrapper import ErasedPointer
from MojoSerial.plugin_SiPixelClusterizer.SiPixelFedCablingMapGPUWrapperESProducer import SiPixelFedCablingMapGPUWrapperESProducer
from MojoSerial.plugin_SiPixelClusterizer.SiPixelGainCalibrationForHLTGPUESProducer import SiPixelGainCalibrationForHLTGPUESProducer
from MojoSerial.plugin_SiPixelClusterizer.SiPixelRawToClusterCUDA import SiPixelRawToClusterCUDA


@export("edmplugin_ESProducer_SiPixelFedCablingMapGPUWrapperESProducer")
def register_SiPixelFedCablingMapGPUWrapperESProducer(registry: ErasedPointer) abi("C"):
    define_fwk_eventsetup_module[SiPixelFedCablingMapGPUWrapperESProducer](registry)


@export("edmplugin_ESProducer_SiPixelGainCalibrationForHLTGPUESProducer")
def register_SiPixelGainCalibrationForHLTGPUESProducer(registry: ErasedPointer) abi("C"):
    define_fwk_eventsetup_module[SiPixelGainCalibrationForHLTGPUESProducer](registry)


@export("edmplugin_EDProducer_SiPixelRawToClusterCUDA")
def register_SiPixelRawToClusterCUDA(registry: ErasedPointer) abi("C"):
    define_fwk_module[SiPixelRawToClusterCUDA](registry)
