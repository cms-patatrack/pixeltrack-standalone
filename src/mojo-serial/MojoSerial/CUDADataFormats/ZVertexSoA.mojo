from MojoSerial.MojoBridge.DTypes import Float, Typeable


# C++ declares these columns as inline fixed arrays because the whole SoA is one
# cudaMalloc'd blob on the device. Here they are Lists: an inline 216 kB struct
# makes every deref of an OwnedPointer[ZVertexSoA] materialize the struct, and
# the compile cost is superlinear in the number of derefs (2 derefs exhausted
# 6 GB). Heap-backed columns keep the struct a handful of pointers. Capacity and
# indexing are unchanged, and hot loops bind Spans anyway (see doc §12).
@fieldwise_init
struct ZVertexSoA(Copyable, Defaultable, Movable, Typeable):
    comptime MAXTRACKS: UInt32 = 32 * 1024
    comptime MAXVTX: UInt32 = 1024

    var idv: List[
        Int16
    ]  # vertex index for each associated (original) track  (-1 == not associate)
    var zv: List[Float]  # output z-position of found vertices
    var wv: List[Float]  # output weight (1/error^2) on the above
    var chi2: List[Float]  # vertices chi2
    var ptv2: List[Float]  # vertices pt^2
    var ndof: List[Int32]  # vertices number of dof
    var sortInd: List[UInt16]  # sorted index (by pt2)  ascending
    var nvFinal: UInt32  # the number of vertices

    @always_inline
    def __init__(out self):
        self.idv = List[Int16](length=Int(Self.MAXTRACKS), fill=0)
        self.zv = List[Float](length=Int(Self.MAXVTX), fill=0.0)
        self.wv = List[Float](length=Int(Self.MAXVTX), fill=0.0)
        self.chi2 = List[Float](length=Int(Self.MAXVTX), fill=0.0)
        self.ptv2 = List[Float](length=Int(Self.MAXVTX), fill=0.0)
        self.ndof = List[Int32](length=Int(Self.MAXTRACKS), fill=0)
        self.sortInd = List[UInt16](length=Int(Self.MAXVTX), fill=0)
        self.nvFinal = 0

    @always_inline
    def init(mut self):
        self.nvFinal = 0

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "ZVertexSoA"
