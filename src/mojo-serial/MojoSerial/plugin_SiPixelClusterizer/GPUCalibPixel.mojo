from std.collections import Span

from MojoSerial.CondFormats.SiPixelGainForHLTonGPU import SiPixelGainForHLTonGPU
from MojoSerial.CUDADataFormats.GPUClusteringConstants import (
    GPUClusteringConstants,
)
from MojoSerial.MojoBridge.DTypes import Float
struct GPUCalibPixel:
    comptime InvId: UInt16 = 9999  # must be > MaxNumModules

    # valid for run2
    comptime VCaltoElectronGain: Float = 47  # L2-4: 47 +- 4.7
    comptime VCaltoElectronGain_L1: Float = 50  # L1:   49.6 +- 2.6
    comptime VCaltoElectronOffset: Float = -60  # L2-4: -60 +- 130
    comptime VCaltoElectronOffset_L1: Float = -670  # L1:   -670 +- 220

    @staticmethod
    def calibDigis(
        isRun2: Bool,
        id: Span[mut=True, UInt16, _],
        x: Span[UInt16, _],
        y: Span[UInt16, _],
        adc: Span[mut=True, UInt16, _],
        ped: SiPixelGainForHLTonGPU,
        numElements: Int32,
        moduleStart: Span[mut=True, UInt32, _],  # just to zero first
        nClustersInModule: Span[mut=True, UInt32, _],  # just to zero them
        clusModuleStart: Span[mut=True, UInt32, _],  # just to zero first
    ):
        clusModuleStart[0] = 0
        moduleStart[0] = 0
        nClustersInModule[
            0 : Int(GPUClusteringConstants.MaxNumModules)
        ].fill(0)
        for i in range(numElements):
            if Self.InvId == id[i]:
                continue
            var conversionFactor: Float = (
                Self.VCaltoElectronGain_L1 if id[i]
                < 96 else Self.VCaltoElectronGain
            ) if isRun2 else 1.0
            var offset: Float = (
                Self.VCaltoElectronOffset_L1 if id[i]
                < 96 else Self.VCaltoElectronOffset
            ) if isRun2 else 0.0
            var isDeadColumn = False
            var isNoisyColumn = False

            var row = x[i].cast[DType.int32]()
            var col = y[i].cast[DType.int32]()
            var ret = ped.getPedAndGain(
                id[i].cast[DType.uint32](),
                col,
                row,
                isDeadColumn,
                isNoisyColumn,
            )
            ref pedestal = ret[0]
            ref gain = ret[1]
            if isDeadColumn or isNoisyColumn:
                id[i] = Self.InvId
                adc[i] = 0
                print("bad pixel at", i, "in", id[i])
            else:
                var rawAdcU16 = adc[i]
                var rawAdc = rawAdcU16.cast[DType.float32]()

                # Match C++ float arithmetic step-by-step:
                # float vcal = adc[i] * gain - pedestal * gain;
                var adcTimesGain = (rawAdc * gain).cast[DType.float32]()
                var pedTimesGain = (pedestal * gain).cast[DType.float32]()
                var vcal = (adcTimesGain - pedTimesGain).cast[DType.float32]()

                # Match C++:
                # adc[i] = std::max(100, int(vcal * conversionFactor + offset));
                var scaled = (vcal * conversionFactor).cast[DType.float32]()
                var calibratedF = (scaled + offset).cast[DType.float32]()

                # C++ int(float) truncates toward zero.
                var calibrated = calibratedF.cast[DType.int32]()

                var finalAdc: UInt16
                if calibrated < 100:
                    finalAdc = UInt16(100)
                else:
                    finalAdc = calibrated.cast[DType.uint16]()

                adc[i] = finalAdc
