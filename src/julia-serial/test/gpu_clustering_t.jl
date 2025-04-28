include("../plugin-SiPixelClusterizer/gpu_clustering.jl")
using .gpuClustering:find_clus, count_modules

include("../plugin-SiPixelClusterizer/gpu_cluster_charge_cut.jl")
using .gpuClusterCharge:cluster_charge_cut

using DataStructures:SortedSet


INV_ID = 9999

const max_num_modules = 2000  # Assuming MaxNumModules is predefined
num_elements = 256 * 2000 # digis (pixels)

# these in reality are already on GPU
h_id = Vector{UInt16}(undef, num_elements) #module ids
h_x = Vector{Int16}(undef, num_elements) 
h_y = Vector{Int16}(undef, num_elements)
h_adc = Vector{Int16}(undef, num_elements)
h_clus = Vector{Int}(undef, num_elements) # cluster id of each digi

h_moduleStart = Vector{Int64}(undef, max_num_modules + 1) # extra stores at first index the number of modules
h_clusInModule = fill(0, max_num_modules) # stores the number of clusters in each module
h_moduleId = Vector{UInt32}(undef, max_num_modules) # module ids in order of the wordfedappender


function generateClusters(kn)
    n = 0
    ncl = 0

    addBigNoise = (1 == kn % 2)
    y = [5, 7, 9, 1, 3, 0, 4, 8, 2, 6]

    if addBigNoise # if odd
        max_pixels = 1000
        id = 666
        for x in 0:3:139 # skipping 80 to 159 rows
            for yy in 0:3:399 # skipping 400 to 415 columns
                n += 1
                ncl += 1
                h_id[n] = id
                h_x[n] = x
                h_y[n] = yy
                h_adc[n] = 1000
                if max_pixels <= ncl
                    break
                end
            end
            if max_pixels <= ncl
                break
            end
        end
    end
    # isolated (10,10)
    id = 42
    x = 10
    ncl += 1
    n += 1
    h_id[n] = id
    h_x[n] = x
    h_y[n] = x
    h_adc[n] = kn == 0 ? 100 : 5000
    # first column (10, 0)
    ncl += 1
    n += 1
    h_id[n] = id
    h_x[n] = x
    h_y[n] = 0
    h_adc[n] = 5000
    # first columns (90,2) (90,1) adjacent added one cluster
    ncl += 1
    n += 1
    h_id[n] = id
    h_x[n] = x + 80
    h_y[n] = 2
    h_adc[n] = 5000
    # (90,1)
    n += 1
    h_id[n] = id
    h_x[n] = x + 80
    h_y[n] = 1
    h_adc[n] = 5000
    
    
    # last column (10, 415)
    ncl += 1
    n += 1
    h_id[n] = id
    h_x[n] = x
    h_y[n] = 415
    h_adc[n] = 5000
    # last columns (90, 415) , (90, 414) adjacent pixels one cluster
    ncl += 1
    n += 1
    h_id[n] = id
    h_x[n] = x + 80
    h_y[n] = 415
    h_adc[n] = 2500
    n += 1
    # (90, 414)
    h_id[n] = id
    h_x[n] = x + 80
    h_y[n] = 414
    h_adc[n] = 2500
    # diagonal
    ncl += 1
    for x in 20:24
        n += 1
        h_id[n] = id
        h_x[n] = x
        h_y[n] = x
        h_adc[n] = 1000
    end

    ncl += 1
    # reversed
    for x in 45:-1:41
        n += 1
        h_id[n] = id
        h_x[n] = x
        h_y[n] = x
        h_adc[n] = 1000
    end
    
    ncl += 1
    n += 1
    h_id[n] = INV_ID  # error
    # messy
    xx = [21, 25, 23, 24, 22] # (21,41) , (25,45) , (23,43) , (22 , 42) , (24 , 44)
    for k in 1:5
        n += 1
        h_id[n] = id
        h_x[n] = xx[k]
        h_y[n] = 20 + xx[k]
        h_adc[n] = 1000
    end
    # holes
    ncl += 1
    for k in 1:5
        n += 1
        h_id[n] = id
        h_x[n] = xx[k]
        h_y[n] = 100 # (21,100) (25, 100) (23, 100) (24, 100) (22, 100)
        h_adc[n] = kn == 2 ? 100 : 1000
        if xx[k] % 2 == 0 # (22,101) (24,101)
            n += 1
            h_id[n] = id
            h_x[n] = xx[k]
            h_y[n] = 101
            h_adc[n] = 1000
        end
    end
    
    # id == 0 (make sure it works!)
    id = 0
    x = 10
    ncl += 1
    n += 1
    h_id[n] = id
    h_x[n] = x
    h_y[n] = x
    h_adc[n] = 5000
    
    # above ids used 0, 666, 42
    
    # all odd id between 11 and 1800
    for id in 11:2:1800 # module ids go from module 11 to 1800
        if (id ÷ 20) % 2 != 0
            n += 1 
            h_id[n] = INV_ID
        end
        for x in 0:4:36
            ncl += 1
            if (id ÷ 10) % 2 != 0 # if tens digit id was odd do this
                for k in 1:10
                    n += 1
                    h_id[n] = id
                    h_x[n] = x
                    h_y[n] = x + y[k]
                    h_adc[n] = 100
                    n += 1
                    h_id[n] = id
                    h_x[n] = x + 1
                    h_y[n] = x + y[k] + 2
                    h_adc[n] = 1000
                end
            else
                for k in 10:-1:1
                    n += 1
                    h_id[n] = id
                    h_x[n] = x
                    h_y[n] = x + y[k]
                    h_adc[n] = kn == 2 ? 10 : 1000
                    if y[k] == 3
                        continue  # hole
                    end
                    if id == 51
                        n += 1
                        h_id[n] = INV_ID
                        n += 1
                        h_id[n] = INV_ID
                    end
                    n += 1
                    h_id[n] = id
                    h_x[n] = x + 1
                    h_y[n] = x + y[k] + 2
                    h_adc[n] = kn == 2 ? 10 : 1000
                end
            end
        end
    end
    # println("x array: ",h_x)
    # println("y array: ",h_y)
    return n, ncl
end

# function plot_clusters(n, loop_num, module_id)
#     indices = findall(x -> x == module_id, h_id[1:n])
#     x_values = h_x[indices]
#     y_values = h_y[indices]
#     cluster_ids = h_clus[indices]

#     unique_clusters = unique(cluster_ids)

#     n_clusters = length(unique_clusters)
#     colors = distinguishable_colors(n_clusters)

#     color_dict = Dict(zip(unique_clusters, colors))

#     cluster_colors = [color_dict[id] for id in cluster_ids]

#     cluster_labels = string.(unique_clusters)

#     scatter(x_values, y_values, marker_z = cluster_ids, color = cluster_colors,
#             label = cluster_labels, legend=:topright, title="Graph for loop $loop_num",
#             xlabel="x", ylabel="y")
    
#     ylims!(0, maximum(y_values) + 50)
    
#     display(current())
# end



for kkk in 0:4
    n = 0
    ncl = 0
    n,ncl = generateClusters(kkk)
  
    println("Loop ", kkk)
    println("created ", n, " digis in ", ncl, " clusters")
    @assert n <= num_elements #512000
    nModules = 0
    h_moduleStart[1] = nModules
    count_modules(h_id, h_moduleStart, h_clus, n)
    find_clus(h_id, h_x, h_y, h_moduleStart, h_clusInModule, h_moduleId, h_clus, n)
    nModules = h_moduleStart[1]  
    nclus = h_clusInModule
    max_num_clusters = maximum(h_clusInModule)
    module_with_max_clusters = 0

    for i in 0:max_num_modules-1
        if h_clusInModule[i+1] == max_num_clusters
            module_with_max_clusters = i
            println("Number of clusters in Module ",i, ": ",max_num_clusters)
            break
        end
    end

    # println("before charge cut found ", sum(nclus), " clusters")
    for i in max_num_modules-1:-1:0
        if nclus[i+1] > 0
            # println("last module is ", i, ' ', nclus[i+1])
            break
        end
    end
    println("Actual Number of Clusters: ", ncl, " Number of clusters from function: ", sum(nclus))
    @assert ncl == sum(nclus)
    cluster_charge_cut(h_id, h_adc, h_moduleStart,nclus, h_moduleId, h_clus, n)
    

    # if h_clusInModule[module_with_max_clusters + 1] > 0
    #     plot_clusters(n, kkk, module_with_max_clusters)
    # else
    #     println("No clusters found for module id $module_with_max_clusters in loop $kkk")
    # end


    # println("found ", nModules, " Modules active")
    clids = SortedSet{UInt}()
    for i in 1:n
        @assert h_id[i] != 666  # only noise
        if h_id[i] == INV_ID 
            continue
        end
        @assert h_clus[i] >= 1
        @assert h_clus[i] <= nclus[h_id[i]+1]
        push!(clids, h_id[i] * 1000 + h_clus[i])
    end
    
    # verify no hole in numbering
    p_cl_id = first(clids)
    p_mod_id = p_cl_id ÷ 1000
    @assert 1 == p_cl_id % 1000
    for curr_cl_id ∈ Iterators.drop(clids, 1)
        curr_mod_id = curr_cl_id ÷ 1000
        nc = curr_cl_id % 1000
        pnc = p_cl_id % 1000

        if p_mod_id != curr_mod_id
            @assert 1 == curr_cl_id % 1000
            @assert nclus[p_mod_id + 1] == p_cl_id % 1000
            p_mod_id = curr_mod_id
            p_cl_id = curr_cl_id
            continue
        end
        p_cl_id = curr_cl_id
        @assert nc == pnc + 1
    end
    
    # println("found ", sum(nclus), ' ', length(clids), " clusters")
    for i in max_num_modules-1:-1:0  
        if nclus[i+1] > 0
            # println("last module is ", i, ' ', nclus[i+1])
            break
        end
    end
end
