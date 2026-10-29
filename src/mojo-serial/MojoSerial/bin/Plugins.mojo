from std.pathlib import Path
from std.utils import Variant

from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.ProductRegistry import ProductRegistry

from MojoSerial.plugin_BeamSpotProducer.BeamSpotESProducer import (
    BeamSpotESProducer,
)
from MojoSerial.plugin_BeamSpotProducer.BeamSpotToPOD import BeamSpotToPOD
from MojoSerial.plugin_PixelTriplets.CAHitNtupletCUDA import CAHitNtupletCUDA
from MojoSerial.plugin_PixelVertexFinding.PixelVertexProducerCUDA import (
    PixelVertexProducerCUDA,
)
from MojoSerial.plugin_SiPixelClusterizer.SiPixelFedCablingMapGPUWrapperESProducer import (
    SiPixelFedCablingMapGPUWrapperESProducer,
)
from MojoSerial.plugin_SiPixelClusterizer.SiPixelGainCalibrationForHLTGPUESProducer import (
    SiPixelGainCalibrationForHLTGPUESProducer,
)
from MojoSerial.plugin_SiPixelClusterizer.SiPixelRawToClusterCUDA import (
    SiPixelRawToClusterCUDA,
)
from MojoSerial.plugin_SiPixelRecHits.PixelCPEFastESProducer import (
    PixelCPEFastESProducer,
)
from MojoSerial.plugin_SiPixelRecHits.SiPixelRecHitCUDA import (
    SiPixelRecHitCUDA,
)
from MojoSerial.plugin_Validation.CountValidator import CountValidator


# C++ registers plugins by dlopen + a static `Registrar<T>` per module, and
# dispatches through `Worker`/`MakerBase` vtables. 1.0 has neither, so the
# closed plugin set is spelled out here and dispatched with a Variant. See §17.
comptime EDPlugin = Variant[
    BeamSpotToPOD,
    SiPixelRawToClusterCUDA,
    SiPixelRecHitCUDA,
    CAHitNtupletCUDA,
    PixelVertexProducerCUDA,
    CountValidator,
]

comptime ESPlugin = Variant[
    BeamSpotESProducer,
    SiPixelFedCablingMapGPUWrapperESProducer,
    SiPixelGainCalibrationForHLTGPUESProducer,
    PixelCPEFastESProducer,
]


# C++: main.cc `edmodules`. Order is the execution order.
def ed_path(validation: Bool) -> List[String]:
    var p: List[String] = [
        String("BeamSpotToPOD"),
        String("SiPixelRawToClusterCUDA"),
        String("SiPixelRecHitCUDA"),
        String("CAHitNtupletCUDA"),
        String("PixelVertexProducerCUDA"),
    ]
    if validation:
        p.append(String("CountValidator"))
    return p^


# C++: main.cc `esmodules`.
def es_path() -> List[String]:
    var p: List[String] = [
        String("BeamSpotESProducer"),
        String("SiPixelFedCablingMapGPUWrapperESProducer"),
        String("SiPixelGainCalibrationForHLTGPUESProducer"),
        String("PixelCPEFastESProducer"),
    ]
    return p^


def make_ed(name: String, mut reg: ProductRegistry) raises -> EDPlugin:
    if name == "BeamSpotToPOD":
        return EDPlugin(BeamSpotToPOD(reg))
    elif name == "SiPixelRawToClusterCUDA":
        return EDPlugin(SiPixelRawToClusterCUDA(reg))
    elif name == "SiPixelRecHitCUDA":
        return EDPlugin(SiPixelRecHitCUDA(reg))
    elif name == "CAHitNtupletCUDA":
        return EDPlugin(CAHitNtupletCUDA(reg))
    elif name == "PixelVertexProducerCUDA":
        return EDPlugin(PixelVertexProducerCUDA(reg))
    elif name == "CountValidator":
        return EDPlugin(CountValidator(reg))
    raise "RuntimeError: unknown EDProducer " + name


def ed_produce(mut p: EDPlugin, mut event: Event, ref es: EventSetup):
    if p.isa[BeamSpotToPOD]():
        p[BeamSpotToPOD].produce(event, es)
    elif p.isa[SiPixelRawToClusterCUDA]():
        p[SiPixelRawToClusterCUDA].produce(event, es)
    elif p.isa[SiPixelRecHitCUDA]():
        p[SiPixelRecHitCUDA].produce(event, es)
    elif p.isa[CAHitNtupletCUDA]():
        p[CAHitNtupletCUDA].produce(event, es)
    elif p.isa[PixelVertexProducerCUDA]():
        p[PixelVertexProducerCUDA].produce(event, es)
    elif p.isa[CountValidator]():
        p[CountValidator].produce(event, es)


def ed_end_job(mut p: EDPlugin) raises:
    if p.isa[BeamSpotToPOD]():
        p[BeamSpotToPOD].endJob()
    elif p.isa[SiPixelRawToClusterCUDA]():
        p[SiPixelRawToClusterCUDA].endJob()
    elif p.isa[SiPixelRecHitCUDA]():
        p[SiPixelRecHitCUDA].endJob()
    elif p.isa[CAHitNtupletCUDA]():
        p[CAHitNtupletCUDA].endJob()
    elif p.isa[PixelVertexProducerCUDA]():
        p[PixelVertexProducerCUDA].endJob()
    elif p.isa[CountValidator]():
        p[CountValidator].endJob()


def make_es(name: String, var data: Path) raises -> ESPlugin:
    if name == "BeamSpotESProducer":
        return ESPlugin(BeamSpotESProducer(data^))
    elif name == "SiPixelFedCablingMapGPUWrapperESProducer":
        return ESPlugin(SiPixelFedCablingMapGPUWrapperESProducer(data^))
    elif name == "SiPixelGainCalibrationForHLTGPUESProducer":
        return ESPlugin(SiPixelGainCalibrationForHLTGPUESProducer(data^))
    elif name == "PixelCPEFastESProducer":
        return ESPlugin(PixelCPEFastESProducer(data^))
    raise "RuntimeError: unknown ESProducer " + name


def es_produce(mut p: ESPlugin, mut es: EventSetup):
    if p.isa[BeamSpotESProducer]():
        p[BeamSpotESProducer].produce(es)
    elif p.isa[SiPixelFedCablingMapGPUWrapperESProducer]():
        p[SiPixelFedCablingMapGPUWrapperESProducer].produce(es)
    elif p.isa[SiPixelGainCalibrationForHLTGPUESProducer]():
        p[SiPixelGainCalibrationForHLTGPUESProducer].produce(es)
    elif p.isa[PixelCPEFastESProducer]():
        p[PixelCPEFastESProducer].produce(es)
