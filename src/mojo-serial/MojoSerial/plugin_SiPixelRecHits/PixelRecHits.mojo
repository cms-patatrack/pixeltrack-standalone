from MojoSerial.DataFormats.BeamSpotPOD import BeamSpotPOD
from MojoSerial.CUDADataFormats.SiPixelClustersSoA import SiPixelClustersSoA
from MojoSerial.CUDADataFormats.SiPixelDigisSoA import SiPixelDigisSoA
from std.collections import Span

from MojoSerial.CondFormats.PixelCPEFast import PixelCPEFast
from MojoSerial.MojoBridge.DTypes import Typeable
from MojoSerial.plugin_SiPixelRecHits.GPUPixelRecHits import getHits
from MojoSerial.CUDACore.HistoContainer import fillManyFromVector


import MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous as TrackingRecHit2DHeterogeneous


def setHitsLayerStart(
    hitsModuleStart: Span[UInt32, _],
    cpeParams: PixelCPEFast,
    hitsLayerStart: Span[mut=True, UInt32, _],
):
    debug_assert(hitsModuleStart[0] == 0)

    for i in range(11):
        hitsLayerStart[i] = hitsModuleStart[
            Int(cpeParams.layerGeometry().layerStart[i])
        ]


@fieldwise_init
struct PixelRecHitGPUKernel(Defaultable, Typeable):
    def makeHits(
        self,
        ref digis_d: SiPixelDigisSoA,
        ref clusters_d: SiPixelClustersSoA,
        ref bs_d: BeamSpotPOD,
        cpeParams: PixelCPEFast,
    ) -> TrackingRecHit2DHeterogeneous.TrackingRecHit2DCPU:
        var nHits = clusters_d.nClusters()
        var hits_d = TrackingRecHit2DHeterogeneous.TrackingRecHit2DCPU(
            nHits, clusters_d.clusModuleStart()
        )

        if digis_d.nModules():  # protect from empty events
            getHits(
                cpeParams,
                bs_d,
                digis_d,
                digis_d.nDigis(),
                clusters_d,
                hits_d,
            )

        # assuming full warp of threads is better than a smaller number...
        if nHits:
            setHitsLayerStart(
                clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart()
            )

        if nHits:
            fillManyFromVector(
                hits_d.phiBinner(),
                10,
                hits_d.iphi(),
                hits_d.hitsLayerStart(),
                nHits,
            )

        return hits_d^

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "PixelRecHitGPUKernel"
