module gpuClustering

export MAX_NUM_MODULES, count_modules, find_clus

using Printf
using ..CUDADataFormatsSiPixelClusterInterfaceGPUClusteringConstants.pixelGPUConstants

#using ..CUDADataFormatsSiPixelClusterInterfaceGPUClusteringConstants.pixelGPUConstants

using ..histogram: HisToContainer, zero, count!, finalize!, size, bin, val, begin_h, end_h, fill!, type_I

using ..Geometry_TrackerGeometryBuilder_phase1PixelTopology_h.phase1PixelTopology: num_cols_in_module
using TaskLocalValues
const CACHED_HIST = TaskLocalValue(() -> HisToContainer{Int16, 418, 4000, 9, UInt16, 1}()) # ? num_cols_in_module + 2 max_pix_in_module ? ? ?
const CACHED_NN = TaskLocalValue(()-> Matrix{UInt16}(undef,160*416, 10))
const CACHED_NNN = TaskLocalValue(()-> zeros(UInt8,160*416))
#using ..histogram: HisToContainer, zero, count, finalize, size, bin, val, begin_h, end_h

#using ..gpuConfig

#using ..heterogeneousCoreCUDAUtilitiesInterfaceCudaCompat.cms.cudacompat 

###
# using ..pixelGPUConstants
if isdefined(Main, :GPU_SMALL_EVENTS)
    const max_hits_in_iter() = 64
else
    const max_hits_in_iter() = 160 # optimized for real data PU 50
end
const MAX_NUM_MODULES::UInt32 = 2000
const MAX_NUM_CLUSTERS_PER_MODULES::Int32 = 1024
const MAX_HITS_IN_MODULE::UInt32 = 1024 # as above
const MAX_NUM_CLUSTERS::UInt32 = pixelGPUConstants.MAX_NUMBER_OF_HITS
const INV_ID::UInt16 = 9999 # must be > MaxNumModules
###

"""
* @brief Counts modules and assigns starting indices for each module in the data.
*
* This function iterates through an array of pixel IDs and identifies 
* module boundaries. It utilizes an atomic operation to update an output array 
* (`module_start`) that stores the starting of the pixel index within a module for each module within the `id` array.
*
* @param id A constant array of type `UInt16` containing module IDs.
* @param module_start An output array of type `UInt32` where the starting index of each module will be stored.
* @param cluster_id An output array of type `UInt32` where each element is initially set to its own index (potentially used for cluster identification later).
* @param num_elements The number of elements (digis) in the `id` array.
* InvId refers to Invalid pixel 
* Changes since julia is indexed at 1
* Note: Each word in the main wordfedappender array is given a clusterid
"""
function count_modules(id::Vector{T}, module_start::Vector{J}, cluster_id::Vector{K}, num_elements::Integer) where {T <: Integer, J <: Integer, K <: Signed}
    first = 1
    for i ∈ first:num_elements
        cluster_id[i] = i
        if id[i] == INV_ID 
            continue
        end
        j = i - 1
        while j >= 1 && id[j] == INV_ID # find the first value to the left that is valid
            j -= 1
        end
        if j < 1 || id[j] != id[i]
            module_start[1] = min(module_start[1] + 1, MAX_NUM_MODULES)
            loc = module_start[1] + 1
            if loc <= length(module_start)
                module_start[loc] = i
            else
                println("Warning: Exceeded the bounds of module_start array. loc = ",loc)
                break
            end
        end
    end
end

"""
* @brief Finds and labels clusters of pixels within a module based on their coordinates and IDs.
*
* This function identifies clusters of pixels within a module by iterating through
* pixel IDs (`id`), and coordinates (`x`, `y`). It uses atomic operations for concurrency
* management to update `cluster_id`, which stores the cluster ID for each pixel.
*
* @param id Array of UInt16, representing Module IDs.
* @param x Array of UInt16, representing pixel x-coordinates.
* @param y Array of UInt16, representing pixel y-coordinates.
* @param module_start Array of UInt32, specifying start indices for modules and pixel boundaries.
* @param n_clusters_in_module UInt32, output array to store the number of clusters found per module.
* @param moduleId UInt32, output array to store module IDs.
* @param cluster_id UInt32, array to store cluster IDs for each pixel.
* @param num_elements Int, the number of elements (digis)
*
* @remarks InvId refers to an invalid pixel ID.
"""
function find_clus(id, x, y, module_start, n_clusters_in_module, moduleId, cluster_id, num_elements)
        
    # julia is 1 indexed
    first_module = 1
    end_module = module_start[1]
    #Hist{T, N, M, K, U} = HisToContainer{T, N, M, K, U} # was on line 120 question why did it cause a lot of memory allocation
    for mod ∈ first_module:end_module # Go over all modules
        first_pixel = module_start[mod + 1] # access index of starting pixel within module
        this_module_id = id[first_pixel] # get module id
        @assert this_module_id < MAX_NUM_MODULES
        first = first_pixel
        msize = num_elements
        
        for i ∈ first:num_elements
            if id[i] == INV_ID 
                continue
            end
            if id[i] != this_module_id
                msize = min(msize, i)
                break
            end
        end
        
        max_pix_in_module = 4000
        nbins = num_cols_in_module + 2
        
        
        
        hist = CACHED_HIST[]
        #hist = HisToContainer{Int16, 418, 4000, 9, UInt16, 1}()
        zero(hist)
        
        @assert msize == num_elements || (msize < num_elements && id[msize] != this_module_id)
        if msize - first_pixel > max_pix_in_module
            @printf("too many pixels in module %d: %d > %d\n", this_module_id, msize - first_pixel, max_pix_in_module)
            msize = max_pix_in_module + first_pixel
        end
        @assert msize - first_pixel <= max_pix_in_module
        if(msize == num_elements && id[msize] == this_module_id)
            msize+=1
        end
        # fill histo
        
        for i in first:msize-1
            if id[i] == INV_ID
                continue
            end
            count!(hist, Int16(y[i]))
        end
        
        finalize!(hist)
        for i in first:msize-1
            if id[i] == INV_ID
                continue 
            end
            fill!(hist, Int16(y[i]), type_I(hist)((i - first_pixel))) # m
        end
        
        # println(hist)
        
        max_iter = size(hist) # number of digis added to hist
        # max_iter = 160 * 416
        max_neighbours = 10
        
        # nearest neighbour 
        nn = Matrix{UInt16}(undef,max_iter, max_neighbours) # m
        nnn = zeros(UInt8, max_iter) # m
        # nn = CACHED_NN[]
        # nnn = CACHED_NNN[]
        fill!(nnn,UInt8(0))
        # fill NN
        # testing = 0 
        for (j, k) in zip(0:size(hist)-1, 1:size(hist)) # j is the index of the digi within the hist
            @assert k <= max_iter
            p = begin_h(hist) + j
            i = val(hist,p) + first_pixel # index of 32bit word (digi)
            @assert id[i] != INV_ID
            @assert id[i] == this_module_id
            be = bin(hist, Int16(y[i]) + Int16(1)) 
            e = end_h(hist, be)
            p += 1
            @assert nnn[k] == 0 
            while p < e
                m = val(hist, p) + first_pixel # index of 32bit word (digi)
                @assert m != i
                @assert y[m] - 0 - y[i] >= 0
                if y[m] - y[i] > 1
                    break
                end
                if abs(x[m] - x[i]) > 1
                    p += 1
                    continue
                end
                # if this_module_id == 510 && i == 15187
                #     testing = k 
                # end
                nnn[k] += 1
                l = nnn[k]
                @assert l <= max_neighbours
                nn[k, l] = val(hist, p)
                p += 1
            end
        end
        
        more = true
        n_loops = 0
        while more
            if n_loops % 2 == 1
                for j ∈ 0:size(hist)-1
                    p = begin_h(hist) + j
                    i = val(hist, p) + first_pixel
                    m = cluster_id[i]
                    while m != cluster_id[m]
                        m = cluster_id[m]
                    end
                    cluster_id[i] = m
                end
            else
                more = false
                
                for (j, k) ∈ zip(0:size(hist)-1, 1:size(hist))
                    p = begin_h(hist) + j 
                    i::Int = val(hist, p) + first_pixel
                    for kk ∈ 1:nnn[k]
                        l = nn[k, kk]
                        m = l + first_pixel
                        @assert m != i
                        old = cluster_id[m]
                        cluster_id[m] = min(cluster_id[m], cluster_id[i])
                        if old != cluster_id[i]
                            more = true
                        end
                        cluster_id[i] = min(cluster_id[i], old)
                    end
                end
            end
            n_loops += 1
        end
        
        found_clusters = 0
        
        for i ∈ first:msize-1
            if id[i] == INV_ID
                continue
            end
            if cluster_id[i] == i
                old = found_clusters
                found_clusters += 1
                cluster_id[i] = -(old + 1)
            end
        end
    
        for i ∈ first:msize-1
            if id[i] == INV_ID 
                continue
            end
            if cluster_id[i] > 0
                cluster_id[i] = cluster_id[cluster_id[i]]
            end
        end
    
        for i ∈ first:msize-1
            if id[i] == INV_ID 
                cluster_id[i] = -9999
                continue
            end
            cluster_id[i] = -cluster_id[i] 
        end
        n_clusters_in_module[this_module_id+1] = found_clusters
        moduleId[mod] = this_module_id
    end
  
end


end
