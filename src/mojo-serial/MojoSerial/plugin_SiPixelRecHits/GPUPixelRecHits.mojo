from std.collections import Span
from std.math import sqrt
from std.utils.numerics import max_finite

from MojoSerial.DataFormats.BeamSpotPOD import BeamSpotPOD
from MojoSerial.DataFormats.ApproxAtan2 import ApproxAtan2
from MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DHeterogeneous,
)
from MojoSerial.CondFormats.PixelCPEFast import PixelCPEFast
from MojoSerial.Geometry.Phase1PixelTopology import AverageGeometry

import MojoSerial.CUDADataFormats.SiPixelDigisSoA as SiPixelDigisSoA
import MojoSerial.CUDADataFormats.SiPixelClustersSoA as SiPixelClustersSoA
import MojoSerial.CondFormats.PixelCPEforGPU as PixelCPEforGPU


def getHits(
    cpeParams: PixelCPEFast,
    bs: BeamSpotPOD,
    digis: SiPixelDigisSoA.SiPixelDigisSoA,
    numElements: UInt32,
    clusters: SiPixelClustersSoA.SiPixelClustersSoA,
    mut hits: TrackingRecHit2DHeterogeneous,
):
    # FIXME
    # the compiler seems NOT to optimize loads from views (even in a simple test case)
    # The whole gimnastic here of copying or not is a pure heuristic exercise that seems to produce the fastest code with the above signature
    # not using views (passing a gazzilion of array pointers) seems to produce the fastest code (but it is harder to mantain)

    # copy average geometry corrected by beamspot . FIXME (move it somewhere else???)

    ref ag = cpeParams.averageGeometry()
    ref agc = hits.averageGeometry()
    for il in range(AverageGeometry.numberOfLaddersInBarrel):
        agc.ladderZ[il] = ag.ladderZ[il] - bs.z
        agc.ladderX[il] = ag.ladderX[il] - bs.x
        agc.ladderY[il] = ag.ladderY[il] - bs.y
        agc.ladderR[il] = sqrt(
            agc.ladderX[il] * agc.ladderX[il]
            + agc.ladderY[il] * agc.ladderY[il]
        )
        agc.ladderMinZ[il] = ag.ladderMinZ[il] - bs.z
        agc.ladderMaxZ[il] = ag.ladderMaxZ[il] - bs.z

    agc.endCapZ[0] = ag.endCapZ[0] - bs.z
    agc.endCapZ[1] = ag.endCapZ[1] - bs.z

    # Columns are bound once here rather than reached through the per-hit
    # accessors: an accessor call reloads the List indirection on every store,
    # which the backend cannot hoist. Field-path borrows, so they compose; a
    # `ref self` accessor would take the whole struct.
    var xLocal = Span(hits.m_xl_d)
    var yLocal = Span(hits.m_yl_d)
    var xerrLocal = Span(hits.m_xerr_d)
    var yerrLocal = Span(hits.m_yerr_d)
    var xGlobal = Span(hits.m_xg_d)
    var yGlobal = Span(hits.m_yg_d)
    var zGlobal = Span(hits.m_zg_d)
    var rGlobal = Span(hits.m_rg_d)
    var iphi = Span(hits.m_iphi_d)
    var charge = Span(hits.m_charge_d)
    var clusterSizeX = Span(hits.m_xsize_d)
    var clusterSizeY = Span(hits.m_ysize_d)
    var detectorIndex = Span(hits.m_detInd_d)

    # to be moved in common namespace...
    comptime InvId: UInt16 = 9999  # must be > MaxNumModules
    comptime MaxHitsInIter = Int(PixelCPEforGPU.MaxHitsInIter)

    comptime ClusParams = PixelCPEforGPU.ClusParams

    # as usual one block per module
    var clusParams = ClusParams()

    var firstModule: Int = 0
    var endModule = Int(clusters.moduleStart(0))
    for module in range(firstModule, endModule):
        var me = Int(clusters.moduleId(module))
        var nclus = Int(clusters.clusInModule(me))

        if nclus == 0:
            continue

        var endClus = nclus
        for startClus in range(0, endClus, MaxHitsInIter):
            var first = Int(clusters.moduleStart(1 + module))

            var nClusInIter: Int = min(MaxHitsInIter, endClus - startClus)
            var lastClus: Int = startClus + nClusInIter

            debug_assert(nClusInIter <= nclus)
            debug_assert(nClusInIter > 0)
            debug_assert(lastClus <= nclus)

            debug_assert(
                nclus > MaxHitsInIter
                or (
                    0 == startClus
                    and nClusInIter == nclus
                    and lastClus == nclus
                )
            )

            # init
            for ic in range(nClusInIter):
                clusParams.minRow[ic] = max_finite[DType.uint32]()
                clusParams.maxRow[ic] = 0
                clusParams.minCol[ic] = max_finite[DType.uint32]()
                clusParams.maxCol[ic] = 0
                clusParams.charge[ic] = 0
                clusParams.Q_f_X[ic] = 0
                clusParams.Q_l_X[ic] = 0
                clusParams.Q_f_Y[ic] = 0
                clusParams.Q_l_Y[ic] = 0

            # one thead per "digi"

            for i in range(first, Int(numElements)):
                var id = digis.moduleInd(i)
                if id == InvId:
                    continue  # not valid
                if Int(id) != me:
                    break  # end of module

                var cl = Int(digis.clus(i))
                if cl < startClus or cl >= lastClus:
                    continue
                var x = UInt32(digis.xx(i))
                var y = UInt32(digis.yy(i))
                cl -= startClus

                debug_assert(cl >= 0)
                debug_assert(cl < MaxHitsInIter)

                clusParams.minRow[cl] = min(clusParams.minRow[cl], x)
                clusParams.maxRow[cl] = max(clusParams.maxRow[cl], x)
                clusParams.minCol[cl] = min(clusParams.minCol[cl], y)
                clusParams.maxCol[cl] = max(clusParams.maxCol[cl], y)

            # pixmx is not available in the binary dumps
            var pixmx = max_finite[DType.uint16]()
            for i in range(first, Int(numElements)):
                var id = digis.moduleInd(i)
                if id == InvId:
                    continue  # not valid
                if Int(id) != me:
                    break  # end of module

                var cl = Int(digis.clus(i))
                if cl < startClus or cl >= lastClus:
                    continue

                cl -= startClus
                debug_assert(cl >= 0)
                debug_assert(cl < MaxHitsInIter)

                var x = UInt32(digis.xx(i))
                var y = UInt32(digis.yy(i))
                var ch = Int32(min(digis.adc(i), pixmx))

                clusParams.charge[cl] += ch
                if clusParams.minRow[cl] == x:
                    clusParams.Q_f_X[cl] += ch
                if clusParams.maxRow[cl] == x:
                    clusParams.Q_l_X[cl] += ch
                if clusParams.minCol[cl] == y:
                    clusParams.Q_f_Y[cl] += ch
                if clusParams.maxCol[cl] == y:
                    clusParams.Q_l_Y[cl] += ch

            # next one cluster per thread...

            first = Int(clusters.clusModuleStart(me)) + startClus

            for ic in range(nClusInIter):
                var h = first + ic  # output index in global memory

                # this cannot happen anymore
                if h >= Int(TrackingRecHit2DHeterogeneous.maxHits()):
                    break  # overflow...

                debug_assert(h < Int(hits.nHits()))
                debug_assert(h < Int(clusters.clusModuleStart(me + 1)))

                PixelCPEforGPU.position(
                    cpeParams.commonParams(),
                    cpeParams.detParams(me),
                    clusParams,
                    UInt32(ic),
                )
                PixelCPEforGPU.errorFromDB(
                    cpeParams.commonParams(),
                    cpeParams.detParams(me),
                    clusParams,
                    UInt32(ic),
                )

                # store it

                charge[h] = clusParams.charge[ic]

                detectorIndex[h] = UInt16(me)

                var xl: Float32
                var yl: Float32

                xl = clusParams.xpos[ic]
                xLocal[h] = xl

                yl = clusParams.ypos[ic]
                yLocal[h] = yl

                clusterSizeX[h] = clusParams.xsize[ic]
                clusterSizeY[h] = clusParams.ysize[ic]

                xerrLocal[h] = clusParams.xerr[ic] * clusParams.xerr[ic]
                yerrLocal[h] = clusParams.yerr[ic] * clusParams.yerr[ic]

                # keep it local for computations

                var xg: Float32 = 0
                var yg: Float32 = 0
                var zg: Float32 = 0

                # to global and compute phi...
                cpeParams.detParams(me).frame.toGlobal(xl, yl, xg, yg, zg)
                # here correct for the beamspot...
                xg -= bs.x
                yg -= bs.y
                zg -= bs.z

                xGlobal[h] = xg
                yGlobal[h] = yg
                zGlobal[h] = zg

                rGlobal[h] = sqrt(xg * xg + yg * yg)
                iphi[h] = ApproxAtan2.unsafe_atan2s[7](yg, xg)
