from std.collections import Span

from MojoSerial.Geometry.Phase1PixelTopology import Phase1PixelTopology
from MojoSerial.CUDACore.HistoContainer import HistoContainer
from MojoSerial.CUDACore.CUDACompat import CUDACompat
from MojoSerial.CUDACore.PrefixScan import blockPrefixScan
from MojoSerial.CUDADataFormats.GPUClusteringConstants import (
    GPUClusteringConstants,
)
struct GPUClustering:
    @staticmethod
    def countModules(
        id: Span[UInt16, _],
        moduleStart: Span[mut=True, UInt32, _],
        clusterId: Span[mut=True, Int32, _],
        numElements: Int32,
    ):
        for i in range(numElements):
            clusterId[i] = i
            if id[i] == GPUClusteringConstants.InvId:
                continue
            var j = i - 1
            while j >= 0 and id[j] == GPUClusteringConstants.InvId:
                j -= 1
            if j < 0 or id[j] != id[i]:
                # boundary
                var loc = moduleStart[0]
                if moduleStart[0] < GPUClusteringConstants.MaxNumModules:
                    moduleStart[0] += 1
                moduleStart[loc + 1] = i.cast[DType.uint32]()

    @staticmethod
    def findClus(
        id: Span[UInt16, _],  # module id of each pixel
        x: Span[UInt16, _],  # local coordinates of each pixel
        y: Span[UInt16, _],  #
        moduleStart: Span[
            UInt32, _
        ],  # index of the first pixel of each module
        nClustersInModule: Span[
            mut=True, UInt32, _
        ],  # output: number of clusters found in each module
        moduleId: Span[
            mut=True, UInt32, _
        ],  # output: module id of each module
        clusterId: Span[
            mut=True, Int32, _
        ],  # output: cluster id of each pixel
        numElements: Int32,
    ):
        var msize: Int32

        # init hist  (ymax=416 < 512 : 9bits)
        comptime maxPixInModule: UInt32 = 4000
        comptime nbins = Phase1PixelTopology.numColsInModule.cast[
            DType.uint32
        ]() + 2
        comptime Hist = HistoContainer[
            DType.uint16,
            nbins,
            maxPixInModule,
            9,
            DType.uint16,
        ]
        var hist = Hist()

        var endModule = moduleStart[0]
        for module in range(endModule):
            var firstPixel = moduleStart[module + 1]
            var thisModuleId = id[firstPixel]
            debug_assert(
                thisModuleId.cast[DType.uint32]()
                < GPUClusteringConstants.MaxNumModules
            )

            var first = firstPixel

            # find the index of the first pixel not belonging to this module (or invalid)
            msize = numElements

            # skip threads not associated to an existing pixel
            for i in range(Int(first), Int(numElements)):
                if id[i] == GPUClusteringConstants.InvId:  # skip invalid pixels
                    continue
                if (
                    id[i] != thisModuleId
                ):  # find the first pixel in a different module
                    msize = min(msize, Int32(i))
                    break

            hist.psws = 0
            for j in range(Hist.totbins()):
                hist.off[j] = 0

            debug_assert(
                (msize == numElements)
                or ((msize < numElements) and (id[msize] != thisModuleId))
            )

            # limit to maxPixInModule

            if (
                msize - firstPixel.cast[DType.int32]()
                > maxPixInModule.cast[DType.int32]()
            ):
                print(
                    "too many pixels in module ",
                    thisModuleId,
                    ": ",
                    msize - firstPixel.cast[DType.int32](),
                    " > ",
                    maxPixInModule,
                    sep="",
                )
                msize = (maxPixInModule + firstPixel).cast[DType.int32]()

            debug_assert(
                msize - firstPixel.cast[DType.int32]()
                <= maxPixInModule.cast[DType.int32]()
            )

            for i in range(Int(first), Int(msize)):
                if id[i] == GPUClusteringConstants.InvId:  # skip invalid pixels
                    continue
                hist.count(y[i])

            hist.finalize()

            for i in range(Int(first), Int(msize)):
                if id[i] == GPUClusteringConstants.InvId:  # skip invalid pixels
                    continue
                hist.fill(y[i], UInt16(i - Int(firstPixel)))

            # Serial connected-component labelling with a union-find:
            #   - the root of each set is always its pixel with the smallest index, so the final labels are the same
            #     as those produced by the iterative min-propagation used by the parallel (GPU) implementation;
            #   - the neighbours are merged as soon as they are found, without storing them.
            @always_inline
            @parameter
            def findRoot(var i: Int) -> Int:
                while Int(clusterId[i]) != i:
                    # path halving
                    clusterId[i] = clusterId[Int(clusterId[i])]
                    i = Int(clusterId[i])
                return i

            var nh = Int(hist.size())
            for j in range(nh):
                var i = Int(hist.bins[j]) + Int(firstPixel)
                debug_assert(id[i] != GPUClusteringConstants.InvId)
                debug_assert(id[i] == thisModuleId)  # same module
                var be = Hist.bin(y[i] + 1).cast[DType.uint32]()
                var e = Int(hist.off[Int(be) + 1])
                for q in range(j + 1, e):
                    var m = Int(hist.bins[q]) + Int(firstPixel)
                    debug_assert(m != i)
                    debug_assert(Int(y[m]) - Int(y[i]) >= 0)
                    debug_assert(Int(y[m]) - Int(y[i]) <= 1)
                    if abs(Int(x[m]) - Int(x[i])) > 1:
                        continue
                    var ri = findRoot(i)
                    var rm = findRoot(m)
                    if ri < rm:
                        clusterId[rm] = Int32(ri)
                    elif rm < ri:
                        clusterId[ri] = Int32(rm)

            # point every pixel directly to the root of its cluster
            for j in range(nh):
                var i = Int(hist.bins[j]) + Int(firstPixel)
                clusterId[i] = Int32(findRoot(i))

            var foundClusters: UInt32 = 0

            # find the number of different clusters, identified by a pixels with clus[i] == i;
            # mark these pixels with a negative id.
            for i in range(Int(first), Int(msize)):
                if id[i] == GPUClusteringConstants.InvId:  # skip invalid pixels
                    continue
                if Int(clusterId[i]) == i:
                    var old = foundClusters
                    if foundClusters < 0xFFFFFFFF:
                        foundClusters += 1
                    clusterId[i] = -((old + 1).cast[DType.int32]())

            # propagate the negative id to all the pixels in the cluster.
            for i in range(Int(first), Int(msize)):
                if id[i] == GPUClusteringConstants.InvId:  # skip invalid pixels
                    continue
                if clusterId[i] >= 0:
                    # mark each pixel in a cluster with the same id as the first one
                    # (read before write: one Span cannot be borrowed both ways)
                    var seed = clusterId[Int(clusterId[i])]
                    clusterId[i] = seed

            # adjust the cluster id to be a positive value starting from 0
            for i in range(Int(first), Int(msize)):
                if id[i] == GPUClusteringConstants.InvId:  # skip invalid pixels
                    clusterId[i] = -9999
                    continue
                clusterId[i] = -clusterId[i] - 1

            nClustersInModule[thisModuleId] = foundClusters
            moduleId[module] = thisModuleId.cast[DType.uint32]()

    @staticmethod
    def clusterChargeCut(
        id: Span[
            mut=True, UInt16, _
        ],  # module id of each pixel (modified if bad cluster)
        adc: Span[UInt16, _],  # charge of each pixel
        moduleStart: Span[
            UInt32, _
        ],  # index of the first pixel of each module
        nClustersInModule: Span[
            mut=True, UInt32, _
        ],  # modified: number of clusters found in each module
        moduleId: Span[UInt32, _],  # module id of each module
        clusterId: Span[
            mut=True, Int32, _
        ],  # modified: cluster id of each pixel
        numElements: UInt32,
    ):
        var charge = InlineArray[
            Int32, Int(GPUClusteringConstants.MaxNumClustersPerModules)
        ](fill=0)
        var ok = InlineArray[
            UInt8, Int(GPUClusteringConstants.MaxNumClustersPerModules)
        ](fill=0)
        var newclusId = InlineArray[
            UInt16, Int(GPUClusteringConstants.MaxNumClustersPerModules)
        ](fill=0)

        var endModule = moduleStart[0]
        for module in range(endModule):
            var firstPixel = moduleStart[module + 1]
            var thisModuleId = id[firstPixel]
            debug_assert(
                thisModuleId.cast[DType.uint32]()
                < GPUClusteringConstants.MaxNumModules
            )
            debug_assert(thisModuleId.cast[DType.uint32]() == moduleId[module])

            var nclus = nClustersInModule[thisModuleId]
            if nclus == 0:
                continue

            if (
                nclus
                > GPUClusteringConstants.MaxNumClustersPerModules.cast[
                    DType.uint32
                ]()
            ):
                print(
                    "Warning too many clusters in module ",
                    thisModuleId,
                    " in block ",
                    0,
                    ": ",
                    nclus,
                    " > ",
                    GPUClusteringConstants.MaxNumClustersPerModules,
                    sep="",
                )

            var first = firstPixel

            if (
                nclus
                > GPUClusteringConstants.MaxNumClustersPerModules.cast[
                    DType.uint32
                ]()
            ):
                # remove excess
                for i in range(Int(first), Int(numElements)):
                    if id[i] == GPUClusteringConstants.InvId:
                        continue  # not valid
                    if id[i] != thisModuleId:
                        break  # end of module
                    if (
                        clusterId[i]
                        >= GPUClusteringConstants.MaxNumClustersPerModules
                    ):
                        id[i] = GPUClusteringConstants.InvId
                        clusterId[i] = GPUClusteringConstants.InvId.cast[
                            DType.int32
                        ]()
                nclus = GPUClusteringConstants.MaxNumClustersPerModules.cast[
                    DType.uint32
                ]()

            debug_assert(
                nclus
                <= GPUClusteringConstants.MaxNumClustersPerModules.cast[
                    DType.uint32
                ]()
            )
            for i in range(nclus):
                charge[i] = 0

            for i in range(Int(first), Int(numElements)):
                if id[i] == GPUClusteringConstants.InvId:
                    continue  # not valid
                if id[i] != thisModuleId:
                    break  # end of module
                charge[clusterId[i]] += adc[i].cast[DType.int32]()

            var chargeCut: Int32 = 2000 if thisModuleId < 96 else 4000
            for i in range(nclus):
                newclusId[i] = 1 if charge[i] > chargeCut else 0
                ok[i] = 1 if charge[i] > chargeCut else 0

            # renumber

            blockPrefixScan(Span(newclusId), nclus)

            debug_assert(nclus >= newclusId[nclus - 1].cast[DType.uint32]())

            if nclus == newclusId[nclus - 1].cast[DType.uint32]():
                continue

            nClustersInModule[thisModuleId] = newclusId[nclus - 1].cast[
                DType.uint32
            ]()

            # mark bad cluster again
            for i in range(nclus):
                if ok[i] == 0:
                    newclusId[i] = GPUClusteringConstants.InvId + 1

            # reassign id
            for i in range(Int(first), Int(numElements)):
                if id[i] == GPUClusteringConstants.InvId:
                    continue  # not valid
                if id[i] != thisModuleId:
                    break  # end of module
                clusterId[i] = newclusId[clusterId[i]].cast[DType.int32]() - 1
                if (
                    clusterId[i]
                    == GPUClusteringConstants.InvId.cast[DType.int32]()
                ):
                    id[i] = GPUClusteringConstants.InvId
