from std.utils import Variant

from MojoSerial.MojoBridge.DTypes import Typeable, TypeableOwnedPointer

from MojoSerial.CondFormats.PixelCPEFast import PixelCPEFast
from MojoSerial.CondFormats.SiPixelFedCablingMapGPUWrapper import (
    SiPixelFedCablingMapGPUWrapper,
)
from MojoSerial.CondFormats.SiPixelFedIds import SiPixelFedIds
from MojoSerial.CondFormats.SiPixelGainCalibrationForHLTGPU import (
    SiPixelGainCalibrationForHLTGPU,
)
from MojoSerial.DataFormats.BeamSpotPOD import BeamSpotPOD


# C++: vector<unique_ptr<ESWrapperBase>> with a virtual destructor. See §15.
comptime ESProduct = Variant[
    TypeableOwnedPointer[BeamSpotPOD],
    TypeableOwnedPointer[PixelCPEFast],
    TypeableOwnedPointer[SiPixelFedCablingMapGPUWrapper],
    TypeableOwnedPointer[SiPixelFedIds],
    TypeableOwnedPointer[SiPixelGainCalibrationForHLTGPU],
]


struct EventSetup(Defaultable, Movable, Typeable):
    var _typeToProduct: Dict[String, ESProduct]

    @always_inline
    def __init__(out self):
        self._typeToProduct = Dict[String, ESProduct]()

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self._typeToProduct = move._typeToProduct^

    def put[
        T: Typeable & Movable & Deinitable
    ](mut self, var prod: T) raises:
        if T.dtype() in self._typeToProduct:
            raise "RuntimeError: Product of type " + T.dtype() + " already exists."
        self._typeToProduct[T.dtype()] = ESProduct(
            TypeableOwnedPointer[T](prod^)
        )

    def get[
        T: Typeable & Movable & Deinitable
    ](self) raises -> ref [
        self._typeToProduct[T.dtype()][TypeableOwnedPointer[T]][]
    ] T:
        if T.dtype() not in self._typeToProduct:
            raise "RuntimeError: Product of type " + T.dtype() + " is not produced."
        return self._typeToProduct[T.dtype()][TypeableOwnedPointer[T]][]

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "EventSetup"
