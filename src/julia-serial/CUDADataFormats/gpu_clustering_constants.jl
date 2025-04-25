module CUDADataFormatsSiPixelClusterInterfaceGPUClusteringConstants

module pixelGPUConstants
export MAX_NUM_MODULES , MAX_NUM_CLUSTERS , MAX_HITS_IN_MODULE, MAX_HITS_IN_ITER, MAX_NUMBER_OF_HITS
    const INV_ID::UInt16 = 9999 # must be > MaxNumModules
    const MAX_NUM_CLUSTERS_PER_MODULES::Int32 = 1024
    const MAX_HITS_IN_MODULE::UInt32 = 1024 # as above
    const MAX_NUM_MODULES::UInt32 = 2000
    # if isdefined(Main, :GPU_SMALL_EVENTS)
    #     const MAX_NUMBER_OF_HITS::UInt32 = UInt32(24 * 1024)
    # else
    #     const MAX_NUMBER_OF_HITS::UInt32 = UInt32(48 * 1024) # data at pileup 50 has 18300 +/- 3500 hits; 40000 is around 6 sigma away
    # end
    const MAX_NUMBER_OF_HITS::UInt32 = UInt32(48 * 1024)
    const MAX_NUM_CLUSTERS = MAX_NUMBER_OF_HITS

    function MAX_HITS_IN_ITER()
        return 160;
    end
end # module pixelGPUConstants
using .pixelGPUConstants
export MAX_NUM_MODULES, MAX_NUM_CLUSTERS, MAX_HITS_IN_MODULE, MAX_HITS_IN_ITER, MAX_NUMBER_OF_HITS

end # module CUDADataFormatsSiPixelClusterInterfaceGPUClusteringConstants