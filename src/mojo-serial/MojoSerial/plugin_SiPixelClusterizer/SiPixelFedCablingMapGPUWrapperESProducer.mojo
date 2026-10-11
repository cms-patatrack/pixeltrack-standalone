from std.pathlib import Path

from MojoSerial.CondFormats.PixelGPUDetails import PixelGPUDetails
from MojoSerial.CondFormats.SiPixelFedIds import SiPixelFedIds
from MojoSerial.CondFormats.SiPixelFedCablingMapGPU import (
    SiPixelFedCablingMapGPU,
)
from MojoSerial.CondFormats.SiPixelFedCablingMapGPUWrapper import (
    SiPixelFedCablingMapGPUWrapper,
)
from MojoSerial.Framework.ESProducer import ESProducer
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.MojoBridge.DTypes import UChar, Typeable
from MojoSerial.MojoBridge.File import (
    read_simd,
    read_list,
)


@fieldwise_init
struct SiPixelFedCablingMapGPUWrapperESProducer(ESProducer):
    var _data: Path

    @always_inline
    def __init__(out self):
        self._data = Path("")

    def produce(mut self, mut eventSetup: EventSetup):
        try:
            with open(self._data / "fedIds.bin", "r") as file:
                var nfeds = read_simd[DType.uint32](file)
                var fedIds = read_list[UInt32](file, Int(nfeds))
                eventSetup.put[SiPixelFedIds](SiPixelFedIds(fedIds^))
            with open(self._data / "cablingMap.bin", "r") as file:
                # C++ reads the whole 128-aligned struct in one go. Every field
                # already starts on a 128-byte boundary, so the fields read back
                # to back, followed by 124 bytes padding the struct after `size`.
                comptime n = Int(PixelGPUDetails.MAX_SIZE)
                var fed = read_list[UInt32](file, n)
                var link = read_list[UInt32](file, n)
                var roc = read_list[UInt32](file, n)
                var RawId = read_list[UInt32](file, n)
                var rocInDet = read_list[UInt32](file, n)
                var moduleId = read_list[UInt32](file, n)
                var badRocs = read_list[UChar](file, n)
                var size = read_simd[DType.uint32](file)
                _ = file.read_bytes(124)
                var obj = SiPixelFedCablingMapGPU(
                    fed^,
                    link^,
                    roc^,
                    RawId^,
                    rocInDet^,
                    moduleId^,
                    badRocs^,
                    size,
                )
                var modToUnpDefSize = read_simd[DType.uint32](file)
                var modToUnpDefault: List[UChar] = file.read_bytes(
                    Int(modToUnpDefSize)
                )
                eventSetup.put[SiPixelFedCablingMapGPUWrapper](
                    SiPixelFedCablingMapGPUWrapper(obj^, modToUnpDefault^)
                )
        except e:
            print(
                (
                    "Error during loading data in"
                    " SiPixelFedCablingMapGPUWrapperESProducer:"
                ),
                e,
            )

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelFedCablingMapGPUWrapperESProducer"
