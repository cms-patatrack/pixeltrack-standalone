from std.utils import Variant

from MojoSerial.Framework.EDGetToken import EDGetTokenT
from MojoSerial.Framework.EDPutToken import EDPutTokenT
from MojoSerial.Framework.ProductRegistry import ProductRegistry
from MojoSerial.MojoBridge.DTypes import Typeable

from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import (
    PixelTrackHeterogeneous,
)
from MojoSerial.CUDADataFormats.SiPixelClustersSoA import SiPixelClustersSoA
from MojoSerial.CUDADataFormats.SiPixelDigiErrorsSoA import SiPixelDigiErrorsSoA
from MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DCPU,
)
from MojoSerial.CUDADataFormats.ZVertexHeterogeneous import ZVertexHeterogeneous
from MojoSerial.DataFormats.BeamSpotPOD import BeamSpotPOD
from MojoSerial.DataFormats.DigiClusterCount import DigiClusterCount
from MojoSerial.DataFormats.FEDRawDataCollection import FEDRawDataCollection
from MojoSerial.CUDADataFormats.SiPixelDigisSoA import SiPixelDigisSoA
from MojoSerial.DataFormats.TrackCount import TrackCount
from MojoSerial.DataFormats.VertexCount import VertexCount

comptime StreamID = Int32


# C++ leaves the `unique_ptr<WrapperBase>` null; a Variant has no empty state.
@fieldwise_init
struct Unset(Copyable, ImplicitlyCopyable, Movable, Typeable):
    var _unused: UInt8

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "Unset"


# C++: vector<unique_ptr<WrapperBase>> with a virtual destructor. See §15.
comptime Product = Variant[
    Unset,
    FEDRawDataCollection,
    BeamSpotPOD,
    SiPixelDigisSoA,
    SiPixelClustersSoA,
    SiPixelDigiErrorsSoA,
    TrackingRecHit2DCPU,
    PixelTrackHeterogeneous,
    ZVertexHeterogeneous,
    DigiClusterCount,
    TrackCount,
    VertexCount,
]


struct Event(Defaultable, Movable, Typeable):
    var _streamId: StreamID
    var _eventId: Int32
    var _products: List[Product]

    @always_inline
    def __init__(out self):
        self._streamId = 0
        self._eventId = 0
        self._products = []

    @always_inline
    def __init__(
        out self,
        var streamId: Int32,
        var eventId: Int32,
        ref reg: ProductRegistry,
    ):
        self._streamId = streamId
        self._eventId = eventId
        self._products = List[Product]()
        for _ in range(reg.__len__()):
            self._products.append(Product(Unset(0)))

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self._streamId = move._streamId
        self._eventId = move._eventId
        self._products = move._products^

    @always_inline
    def streamID(self) -> StreamID:
        return self._streamId

    @always_inline
    def eventID(self) -> Int32:
        return self._eventId

    def get[
        T: Typeable & Movable
    ](self, ref token: EDGetTokenT[T]) -> ref [
        self._products[token.index()][T]
    ] T:
        return self._products[token.index()][T]

    # emplace is not possible due to failure in binding the constructor at compile time, so we provide put instead

    def put[
        T: Typeable & Movable
    ](mut self, ref token: EDPutTokenT[T], var prod: T):
        self._products[token.index()] = Product(prod^)

    @staticmethod
    @always_inline
    def dtype() -> String:
        return "Event"
