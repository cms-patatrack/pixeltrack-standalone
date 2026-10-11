from std.pathlib import Path


from MojoSerial.CondFormats.PixelCPEforGPU import (
    CommonParams,
    DetParams,
    LayerGeometry,
    AverageGeometry,
)
from MojoSerial.Geometry.Phase1PixelTopology import Phase1PixelTopology
from MojoSerial.MojoBridge.DTypes import Float, Typeable
from MojoSerial.MojoBridge.File import read_simd, read_obj, read_list

comptime micronsToCm: Float = 1.0e-4


struct PixelCPEFast(Defaultable, Movable, Typeable):
    var m_detParamsGPU: List[DetParams]
    var m_commonParamsGPU: CommonParams
    var m_layerGeometry: LayerGeometry
    var m_averageGeometry: AverageGeometry

    @always_inline
    def __init__(out self):
        self.m_detParamsGPU = []
        self.m_commonParamsGPU = CommonParams()
        self.m_layerGeometry = LayerGeometry()
        self.m_averageGeometry = AverageGeometry()

    def __init__(out self, path: Path):
        try:
            with open(path, "r") as file:
                self.m_commonParamsGPU = read_obj[CommonParams](file)
                var ndetParams = read_simd[DType.uint32](file)
                self.m_detParamsGPU = read_list[DetParams](
                    file, Int(ndetParams)
                )
                self.m_averageGeometry = read_obj[AverageGeometry](file)
                self.m_layerGeometry = read_obj[LayerGeometry](file)
        except e:
            print(
                "Error during loading data in PixelCPEFast:",
                e,
            )
            return Self()

    # C++ reaches these through ParamsOnGPU; that handle is gone, so they are
    # accessors on the owner.

    @always_inline
    def commonParams(self) -> CommonParams:
        return self.m_commonParamsGPU

    @always_inline
    def detParams(ref self, i: Int) -> ref [self.m_detParamsGPU[i]] DetParams:
        return self.m_detParamsGPU[i]

    @always_inline
    def layerGeometry(ref self) -> ref [self.m_layerGeometry] LayerGeometry:
        return self.m_layerGeometry

    @always_inline
    def averageGeometry(
        ref self,
    ) -> ref [self.m_averageGeometry] AverageGeometry:
        return self.m_averageGeometry

    @always_inline
    def layer(self, id: UInt16) -> UInt8:
        return self.m_layerGeometry.layer[
            Int(id) // Int(Phase1PixelTopology.maxModuleStride)
        ]

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "PixelCPEFast"
