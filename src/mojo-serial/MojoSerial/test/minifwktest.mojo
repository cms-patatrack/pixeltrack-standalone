from std.pathlib import Path

from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.bin.PluginManager import PluginManager

# Build with -D LIB_DIR=<the directory of the plugin libraries>.


def main() raises:
    var esmodules: List[String] = [
        "BeamSpotESProducer",
        "SiPixelFedCablingMapGPUWrapperESProducer",
        "SiPixelGainCalibrationForHLTGPUESProducer",
        "PixelCPEFastESProducer",
    ]
    var pluginManager = PluginManager(1)
    var evt = EventSetup()
    for name in esmodules:
        pluginManager.load(name)
        var esp = pluginManager.createES(name, Path("data"))
        esp.produce(evt)
        print(name)
