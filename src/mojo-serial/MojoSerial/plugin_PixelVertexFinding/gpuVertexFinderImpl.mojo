from std.sys import is_defined

from MojoSerial.MojoBridge.DTypes import Float
from MojoSerial.plugin_PixelVertexFinding.gpuClusterTracksByDensity import (
    clusterTracksByDensity,
)
from MojoSerial.plugin_PixelVertexFinding.gpuFitVertices import fitVertices
from MojoSerial.plugin_PixelVertexFinding.gpuSortByPt2 import sortByPt2
from MojoSerial.plugin_PixelVertexFinding.gpuSplitVertices import splitVertices
from MojoSerial.plugin_PixelVertexFinding.gpuVertexFinder import ZVertices, WorkSpace

# #define THREE_KERNELS


@always_inline
def vertexFinderOneKernel(
    mut data: ZVertices,
    mut ws: WorkSpace,
    minT: Int32,  # min number of neighbours to be "seed"
    eps: Float,  # max absolute distance to cluster
    errmax: Float,  # max error to be "seed"
    chi2max: Float,  # max normalized distance to cluster,
) raises:
    comptime if not is_defined["THREE_KERNELS"]():
        clusterTracksByDensity(data, ws, minT, eps, errmax, chi2max)

        fitVertices(data, ws, 50.0)

        splitVertices(data, ws, 9.0)

        fitVertices(data, ws, 5000.0)

        sortByPt2(data, ws)
    else:
        raise "vertexFinderOneKernel is not compiled in with THREE_KERNELS defined"


@always_inline
def vertexFinderKernel1(
    mut data: ZVertices,
    mut ws: WorkSpace,
    minT: Int32,  # min number of neighbours to be "seed"
    eps: Float,  # max absolute distance to cluster
    errmax: Float,  # max error to be "seed"
    chi2max: Float,  # max normalized distance to cluster,
) raises:
    comptime if is_defined["THREE_KERNELS"]():
        clusterTracksByDensity(data, ws, minT, eps, errmax, chi2max)

        fitVertices(data, ws, 50.0)
    else:
        raise "vertexFinderKernel1 requires THREE_KERNELS to be defined"


@always_inline
def vertexFinderKernel2(mut data: ZVertices, mut ws: WorkSpace) raises:
    comptime if is_defined["THREE_KERNELS"]():
        fitVertices(data, ws, 5000.0)

        sortByPt2(data, ws)
    else:
        raise "vertexFinderKernel2 requires THREE_KERNELS to be defined"
