from layout import Layout, LayoutTensor, IntTuple

from MojoSerial.CUDACore.EigenSoA import MatrixSoA
from MojoSerial.MojoBridge.DTypes import Float, Double, Typeable
from MojoSerial.MojoBridge.Matrix import Matrix


# WARNING: THIS STRUCT IS 128-ALIGNED
@fieldwise_init
struct TrajectoryStateSoA[S: Int32](Copyable, Defaultable, Movable, Typeable):
    comptime Vector5f = LayoutTensor[DType.float32, Layout.col_major(5, 1), _]
    comptime Vector15f = LayoutTensor[
        DType.float32, Layout.col_major(15, 1), _
    ]

    comptime Vector5d = LayoutTensor[DType.float64, Layout.col_major(5, 1), _]
    comptime Matrix5d = LayoutTensor[DType.float64, Layout.col_major(5, 5), _]

    var state: MatrixSoA[DType.float32, 5, 1, Int(Self.S)]
    var covariance: MatrixSoA[DType.float32, 15, 1, Int(Self.S)]

    @always_inline
    def __init__(out self):
        self.state = MatrixSoA[DType.float32, 5, 1, Int(Self.S)]()
        self.covariance = MatrixSoA[DType.float32, 15, 1, Int(Self.S)]()

    @staticmethod
    @always_inline
    def stride() -> Int32:
        return Self.S

    @always_inline
    def copyFromCircle[T: DType](
        mut self,
        cp: LayoutTensor[T, Layout.col_major(3, 1), _],
        ccov: LayoutTensor[T, Layout.col_major(3, 3), _],
        lp: LayoutTensor[T, Layout.col_major(2, 1), _],
        lcov: LayoutTensor[T, Layout.col_major(2, 2), _],
        b: Float,
        i: Int32,
    ):
        var st = self.state[i]
        st[0, 0] = cp[0, 0].cast[DType.float32]()
        st[1, 0] = cp[1, 0].cast[DType.float32]()
        st[2, 0] = cp[2, 0].cast[DType.float32]()
        st[3, 0] = lp[0, 0].cast[DType.float32]()
        st[4, 0] = lp[1, 0].cast[DType.float32]()
        st[2, 0] *= b
        var cov = self.covariance[i]
        cov[0, 0] = ccov[0, 0].cast[DType.float32]()
        cov[1, 0] = ccov[0, 1].cast[DType.float32]()
        cov[2, 0] = b * ccov[0, 2].cast[DType.float32]()
        cov[4, 0] = 0
        cov[3, 0] = 0
        cov[5, 0] = ccov[1, 1].cast[DType.float32]()
        cov[6, 0] = b * ccov[1, 2].cast[DType.float32]()
        cov[8, 0] = 0
        cov[7, 0] = 0
        cov[9, 0] = b * b * ccov[2, 2].cast[DType.float32]()
        cov[11, 0] = 0
        cov[10, 0] = 0
        cov[12, 0] = lcov[0, 0].cast[DType.float32]()
        cov[13, 0] = lcov[0, 1].cast[DType.float32]()
        cov[14, 0] = lcov[1, 1].cast[DType.float32]()

    @always_inline
    def copyFromDense(
        mut self,
        v: LayoutTensor[_, Layout.col_major(5, 1), ...],
        cov: LayoutTensor[_, Layout.col_major(5, 5), ...],
        i: Int32,
    ):
        self.state.__setitem__(i, v)
        var ind: Int = 0

        comptime for j in range(5):

            comptime for k in range(j, 5):
                self.covariance[i][ind, 0] = rebind[Scalar[DType.float32]](
                    cov[j, k].cast[DType.float32]()
                )
                ind += 1

    @always_inline
    # C++ takes `V5& v, M5& cov`; writing through a LayoutTensor needs `mut=True`
    # on the tensor's own origin, not `mut` on the binding.
    def copyToDense(
        self,
        v: LayoutTensor[mut=True, ...],
        cov: LayoutTensor[mut=True, DType.float32, _, ...],
        i: Int32,
    ):
        var wx = self.state[i]
        var rows = wx.layout.shape[0].value()
        var colns = wx.layout.shape[1].value()
        for i in range(rows):
            for j in range(colns):
                v[i, j] = rebind[Scalar[v.dtype]](wx[i, j].cast[v.dtype]())
        var ind: Int = 0

        comptime for j in range(5):
            cov[j, j] = rebind[Scalar[DType.float32]](
                self.covariance[i][ind, 0]
            )
            ind += 1

            comptime for k in range(j + 1, 5):
                cov[j, k] = rebind[Scalar[DType.float32]](
                    self.covariance[i][ind, 0]
                )
                cov[k, j] = cov[j, k]
                ind += 1

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "TrajectoryStateSoA[" + String(Self.S) + "]"
