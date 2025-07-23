using .CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h: n_hits
using .cAHitNtupletGenerator: Counters, Params, CAHitNTupletGeneratorKernels, build_doublets, launch_kernels, resetCAHitNTupletGeneratorKernels, fill_hit_det_indices, classify_tuples
using .Tracks: TrackSOA, hit_indices
using .RecoPixelVertexing_PixelTrackFitting_interface_BrokenLine_h
using TaskLocalValues
using .BrokenLineFitOnGPU: launchBrokenLineKernelsOnCPU
using .RecoPixelVertexing_PixelTrackFitting_plugins_HelixFitOnGPU_h: HelixFitOnGPU, allocate_on_gpu!

const CACHED_KERNELS = TaskLocalValue(() -> CAHitNTupletGeneratorKernels(Params(
    false,             # onGPU
    3,                 # minHitsPerNtuplet,
    458752,            # maxNumberOfDoublets
    false,             # useRiemannFit
    true,              # fit5as4,
    true,              #includeJumpingForwardDoublets
    true,              # earlyFishbone
    false,             # lateFishbone
    true,              # idealConditions
    false,             # fillStatistics
    true,              # doClusterCut
    true,              # doZ0Cut
    true,              # doPtCut
    0.899999976158,    # ptmin
    0.00200000009499,  # CAThetaCutBarrel
    0.00300000002608,  # CAThetaCutForward
    0.0328407224959,   # hardCurvCut
    0.15000000596,     # dcaCutInnerTriplet
    0.25,              # dcaCutOuterTriplet
    cAHitNtupletGenerator.cuts
)))

struct CAHitNtupletGeneratorOnGPU
    m_params::Params
    m_counters::Counters
    function CAHitNtupletGeneratorOnGPU()
        new(
            Params(
                false,             # onGPU
                3,                 # minHitsPerNtuplet,
                458752,            # maxNumberOfDoublets
                false,             # useRiemannFit
                true,              # fit5as4,
                true,              #includeJumpingForwardDoublets
                true,              # earlyFishbone
                false,             # lateFishbone
                true,              # idealConditions
                false,             # fillStatistics
                true,              # doClusterCut
                true,              # doZ0Cut
                true,              # doPtCut
                0.899999976158,    # ptmin
                0.00200000009499,  # CAThetaCutBarrel
                0.00300000002608,  # CAThetaCutForward
                0.0328407224959,   # hardCurvCut
                0.15000000596,     # dcaCutInnerTriplet
                0.25,              # dcaCutOuterTriplet
                cAHitNtupletGenerator.cuts
            ),
            Counters()
        )
    end
end


function make_tuples(self::CAHitNtupletGeneratorOnGPU, hits_d::TrackingRecHit2DHeterogeneous, b_field::AbstractFloat)
    # Create PixelTrackHeterogeneous
    tracks = TrackSOA()
    soa = tracks
    @assert !isnothing(soa)
    #kernels = CAHitNTupletGeneratorKernels(self.m_params) # m 
    kernels = CACHED_KERNELS[]
    resetCAHitNTupletGeneratorKernels(kernels)
    kernels.counters = self.m_counters

    build_doublets(kernels, hits_d) # doesnt modify n_hits 124.197 to 172.237
    launch_kernels(kernels, hits_d, tracks)
    fill_hit_det_indices(hist_view(hits_d), tracks)
    # # if(! const bool useRiemannFit_;)

     # HelixFitOnGPU fitter(bfield, m_params.fit5as4_);
    # # fitter.allocateOnGPU(&(soa->hitIndices), kernels.tupleMultiplicity(), soa);

    fitter = HelixFitOnGPU(Float32(b_field), self.m_params.fit_5_as_4)

    # # fitter.tuple_multiplicity_d is nothing 
    allocate_on_gpu!(fitter, hit_indices(soa), kernels.device_tuple_multiplicity, soa)

    # # for i in 1:15
    # #     println("i: ", i)
    # #     println(fitter.tuple_multiplicity_d.bins[fitter.tuple_multiplicity_d.off[3]+i])
    # # end

   launchBrokenLineKernelsOnCPU(fitter, hist_view(hits_d), n_hits(hits_d), UInt32(24 * 1024))
    
   classify_tuples(kernels,hits_d,tracks,kernels.device_the_cell_tracks)
    return tracks
end
