from MojoSerial.CUDACore.CUDACompat import CUDACompat
from MojoSerial.MojoBridge.DTypes import Float
from MojoSerial.plugin_PixelVertexFinding.gpuVertexFinder import ZVertices, WorkSpace


# A vertex's sort key next to its index, so the sort comparator reads only its
# arguments. A comparator that reads the enclosing `ptv2` segfaults inside
# stdlib `sort` in Mojo 1.0.
@fieldwise_init
struct _PtIndex(Copyable, Movable, TrivialRegisterPassable):
    var pt2: Float
    var index: UInt16


@always_inline
def sortByPt2(mut data: ZVertices, mut ws: WorkSpace):
    var nt: UInt32 = ws.ntrks
    ref ptt2 = ws.ptt2
    var nvFinal: UInt32 = data.nvFinal

    ref iv = ws.iv
    ref ptv2 = data.ptv2
    ref sortInd = data.sortInd

    if nvFinal < 1:
        return

    # fill indexing
    for i in range(nt):
        data.idv[Int(ws.itrk[i])] = Int16(iv[i])

    # can be done asynchronously at the end of the previous event
    for i in range(nvFinal):
        ptv2[i] = 0

    for i in range(nt):
        if iv[i] > 9990:
            continue
        _ = CUDACompat.atomicAdd(ptv2[Int(iv[i])], ptt2[i])

    if nvFinal == 1:
        sortInd[0] = 0
        return

    # C++ fills sortInd with 0..nvFinal-1 and std::sorts it by ptv2.
    var n = Int(nvFinal)
    var keyed = List[_PtIndex](capacity=n)
    for i in range(n):
        keyed.append(_PtIndex(ptv2[i], UInt16(i)))
    var keyedSpan = Span(keyed)

    @parameter
    def less_than(a: _PtIndex, b: _PtIndex) -> Bool:
        return a.pt2 < b.pt2

    sort[less_than](keyedSpan)
    for i in range(n):
        sortInd[i] = keyed[i].index
