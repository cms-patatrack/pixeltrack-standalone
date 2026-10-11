from std.pathlib import Path

from MojoSerial.CondFormats.SiPixelGainForHLTonGPU import SiPixelGainForHLTonGPU
from MojoSerial.CondFormats.SiPixelGainCalibrationForHLTGPU import (
    SiPixelGainCalibrationForHLTGPU,
)
from MojoSerial.Framework.ESProducer import ESProducer
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.MojoBridge.DTypes import Char, Typeable, UChar
from MojoSerial.MojoBridge.File import read_simd


@fieldwise_init
struct SiPixelGainCalibrationForHLTGPUESProducer(ESProducer):
    var _data: Path

    @always_inline
    def __init__(out self):
        self._data = Path("")

    def produce(mut self, mut eventSetup: EventSetup):
        try:
            with open(self._data / "gain.bin", "r") as file:
                # C++ reads the struct in one go, but its first field is an
                # 8-byte pointer while v_pedestals here is a 16-byte Span, so a
                # whole-struct read would be 8 bytes out of step (24048 vs 24056
                # bytes). Read the on-disk layout field by field instead.
                _ = file.read_bytes(8)
                var gain = SiPixelGainForHLTonGPU()
                for i in range(2000):
                    var first = read_simd[DType.uint32](file)
                    var second = read_simd[DType.uint32](file)
                    var ncols = read_simd[DType.int32](file)
                    gain.rangeAndCols[i] = Tuple[
                        SiPixelGainForHLTonGPU.Range, Int32
                    ](SiPixelGainForHLTonGPU.Range(first, second), ncols)
                gain._minPed = read_simd[DType.float32](file)
                gain._maxPed = read_simd[DType.float32](file)
                gain._minGain = read_simd[DType.float32](file)
                gain._maxGain = read_simd[DType.float32](file)
                gain.pedPrecision = read_simd[DType.float32](file)
                gain.gainPrecision = read_simd[DType.float32](file)
                gain._numberOfRowsAveragedOver = read_simd[DType.uint32](file)
                gain._nBinsToUseForEncoding = read_simd[DType.uint32](file)
                gain._deadFlag = read_simd[DType.uint32](file)
                gain._noisyFlag = read_simd[DType.uint32](file)
                var nbytes = read_simd[DType.uint32](file)
                var gainData: List[UChar] = file.read_bytes(Int(nbytes))
                eventSetup.put[SiPixelGainCalibrationForHLTGPU](
                    SiPixelGainCalibrationForHLTGPU(
                        gain^,
                        rebind[List[Char]](gainData).copy()
                        # rebind works because UChar and Char are bit-compatible;
                        # it yields a reference, so the copy is on its result
                    )
                )
        except e:
            print(
                (
                    "Error during loading data in"
                    " SiPixelGainCalibrationForHLTGPUESProducer:"
                ),
                e,
            )

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelGainCalibrationForHLTGPUESProducer"
