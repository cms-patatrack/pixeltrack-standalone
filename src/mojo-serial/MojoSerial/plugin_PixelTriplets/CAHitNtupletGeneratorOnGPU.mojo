
import MojoSerial.plugin_PixelTriplets.CAConstants as CAConstants
from MojoSerial.plugin_PixelTriplets.CAHitNtupletGeneratorKernels import (
    CAHitNtupletGeneratorKernelsCPU,
    Counters as KernelCounters,
    Params as KernelParams,
    QualityCuts as KernelQualityCuts,
    Stream,
)
from MojoSerial.plugin_PixelTriplets.HelixFitOnGPU import HelixFitOnGPU
from MojoSerial.CUDADataFormats.PixelTrackHeterogeneous import (
    PixelTrack as pixelTrack,
    PixelTrackHeterogeneous,
)
from MojoSerial.CUDADataFormats.TrackingRecHit2DHeterogeneous import (
    TrackingRecHit2DCPU,
)
from MojoSerial.CondFormats.PixelCPEFast import PixelCPEFast
from MojoSerial.Framework.Event import Event
from MojoSerial.Framework.EventSetup import EventSetup
from MojoSerial.Framework.ProductRegistry import ProductRegistry


struct CAHitNtupletGeneratorOnGPU:
    comptime Quality = pixelTrack.Quality
    comptime OutputSoA = pixelTrack.TrackSoA
    comptime HitContainer = pixelTrack.HitContainer
    comptime Tuple = Self.HitContainer

    comptime QualityCuts = KernelQualityCuts
    comptime Params = KernelParams
    comptime Counters = KernelCounters

    var m_params: Self.Params
    # Counters is 88 B (11 x UInt64); inline, since the box only added an
    # allocation and an indirection.
    var m_counters: Self.Counters

    # C++: CAHitNtupletGeneratorOnGPU::CAHitNtupletGeneratorOnGPU (CAHitNtupletGeneratorOnGPU.cc)
    def __init__(out self, mut reg: ProductRegistry):
        self.m_params = Self.Params(
            False,  # onGPU
            3,  # minHitsPerNtuplet,
            458752,  # maxNumberOfDoublets
            False,  # useRiemannFit
            True,  # fit5as4,
            True,  # includeJumpingForwardDoublets
            True,  # earlyFishbone
            False,  # lateFishbone
            True,  # idealConditions
            False,  # fillStatistics
            True,  # doClusterCut
            True,  # doZ0Cut
            True,  # doPtCut
            0.899999976158,  # ptmin
            0.00200000009499,  # CAThetaCutBarrel
            0.00300000002608,  # CAThetaCutForward
            0.0328407224959,  # hardCurvCut
            0.15000000596,  # dcaCutInnerTriplet
            0.25,  # dcaCutOuterTriplet
        )
        self.m_counters = Self.Counters()

    # C++: CAHitNtupletGeneratorOnGPU::~CAHitNtupletGeneratorOnGPU (CAHitNtupletGeneratorOnGPU.cc)
    def __deinit__(deinit self):
        if self.m_params.doStats:
            CAHitNtupletGeneratorKernelsCPU.print_counters(self.m_counters)

    # C++: CAHitNtupletGeneratorOnGPU::makeTuples (CAHitNtupletGeneratorOnGPU.cc)
    # cpeParams threaded in: C++ reads it off the hits view (§11). C++ is
    # `const` only because m_counters is a raw pointer; an owned field is not.
    def make_tuples(
        mut self,
        hits_d: TrackingRecHit2DCPU,
        cpeParams: PixelCPEFast,
        bfield: Float32,
    ) raises -> PixelTrackHeterogeneous:
        var tracks = PixelTrackHeterogeneous(Self.OutputSoA())

        var kernels = CAHitNtupletGeneratorKernelsCPU(self.m_params)
        kernels.allocate_on_gpu(Stream())

        kernels.build_doublets(hits_d, Stream())
        kernels.launch_kernels(hits_d, tracks[], Stream(), self.m_counters)
        # in principle needed only if Hits not "available"
        kernels.fill_hit_det_indices(hits_d, tracks[], Stream())

        if hits_d.nHits() == 0:
            return tracks^

        # now fit
        var fitter = HelixFitOnGPU(bfield, self.m_params.fit5as4)

        if self.m_params.useRiemannFit:
            fitter.launchRiemannKernelsOnCPU(
                hits_d,
                cpeParams,
                kernels.tuple_multiplicity(),
                tracks[],
                hits_d.nHits(),
                CAConstants.maxNumberOfQuadruplets(),
            )
        else:
            fitter.launchBrokenLineKernelsOnCPU(
                hits_d,
                cpeParams,
                kernels.tuple_multiplicity(),
                tracks[],
                hits_d.nHits(),
                CAConstants.maxNumberOfQuadruplets(),
            )

        kernels.classify_tuples(hits_d, tracks[], Stream(), self.m_counters)

        return tracks^
