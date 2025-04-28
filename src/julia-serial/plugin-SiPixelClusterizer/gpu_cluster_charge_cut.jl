module gpuClusterCharge

    include("../CUDACore/cuda_assert.jl")
    # using .gpuConfig
    include("../CUDACore/prefix_scan.jl")
    using .prefix_scan:block_prefix_scan
    include("../CUDADataFormats/gpu_clustering_constants.jl")
    using .CUDADataFormatsSiPixelClusterInterfaceGPUClusteringConstants.pixelGPUConstants:INV_ID, MAX_NUM_CLUSTERS_PER_MODULES, MAX_NUM_MODULES
    using Printf
    using TaskLocalValues
    const CACHED_CHARGE = TaskLocalValue(()-> fill(Int32(0),MAX_NUM_CLUSTERS_PER_MODULES))
    const CACHED_OK = TaskLocalValue(()-> Vector{UInt8}(undef, MAX_NUM_CLUSTERS_PER_MODULES))
    const CACHED_NEWCLUSID = TaskLocalValue(()-> Vector{UInt16}(undef, MAX_NUM_CLUSTERS_PER_MODULES))
    function cluster_charge_cut(id, adc, moduleStart, nClustersInModule, moduleId, clusterId, numElements)
        charge = fill(Int32(0),MAX_NUM_CLUSTERS_PER_MODULES) # m
        # charge = CACHED_CHARGE[]
        ok = Vector{UInt8}(undef, MAX_NUM_CLUSTERS_PER_MODULES) # m
        # ok = CACHED_OK[]
        newclusId = Vector{UInt16}(undef, MAX_NUM_CLUSTERS_PER_MODULES) # m 
        # newclusId = CACHED_NEWCLUSID[]
        firstModule = 1
        endModule = moduleStart[1]
        for mod ∈ firstModule:endModule
            firstPixel = moduleStart[1 + mod]
            thisModuleId = id[firstPixel]
            @assert thisModuleId < MAX_NUM_MODULES
            @assert thisModuleId == moduleId[mod]
            nClus = nClustersInModule[thisModuleId+1]
            if nClus == 0
                continue
            end
            if nClus > MAX_NUM_CLUSTERS_PER_MODULES
                @printf("Warning too many clusters in module %d in block %d: %d > %d\n",
               thisModuleId,
               0,
               nClus,
               MaxNumClustersPerModules)
            end
            
            first = firstPixel

            if nClus > MAX_NUM_CLUSTERS_PER_MODULES
                for i in first:numElements
                    if id[i] == INV_ID | id[i] == -INV_ID
                        continue
                    end
                    if id[i] != thisModuleId
                        break
                    end
                    if clusterId[i] > MAX_NUM_CLUSTERS_PER_MODULES
                        id[i] = INV_ID 
                        clusterId[i] = INV_ID 
                    end
                end
                nClus = MAX_NUM_CLUSTERS_PER_MODULES
            end

            @assert nClus <= MAX_NUM_CLUSTERS_PER_MODULES

            fill!(charge,0)

            for i in first:numElements
                if id[i] == INV_ID | id[i] == -INV_ID 
                    continue
                end
                if id[i] != thisModuleId
                    break
                end
                charge[clusterId[i]] += adc[i]
            end

            chargeCut = thisModuleId < 96 ? 2000 : 4000 # L1 : 2000 , other layers : 4000
            for i in 1:nClus
                newclusId[i] = ok[i] = charge[i] > chargeCut ? 1 : 0
            end
            block_prefix_scan(newclusId, nClus)
            @assert nClus >= newclusId[nClus]

            if nClus == newclusId[nClus]
                continue
            end

            nClustersInModule[thisModuleId+1] =  newclusId[nClus]

            for i in 1:nClus
                if ok[i] == 0 
                    newclusId[i] = INV_ID 
                end
            end

            for i in first:numElements
                if id[i] == INV_ID
                    continue
                end
                if id[i] != thisModuleId
                    break
                end
                clusterId[i] = newclusId[clusterId[i]]
                if clusterId[i] == INV_ID || clusterId[i] == -INV_ID 
                    id[i] = INV_ID 
                end
            end
        end
    end
end