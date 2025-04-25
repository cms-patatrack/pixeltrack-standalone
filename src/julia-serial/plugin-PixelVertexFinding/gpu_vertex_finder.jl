module gpuVertexFinder
using ..VertexSOA: MAX_TRACKS, MAX_VTX, ZVertexSoA
using ..Tracks:n_hits_track, stride_track, zip
using ..Patatrack:Quality,loose
using ..histogram:HisToContainer, capacity, count!, finalize!, fill!, for_each_in_bins, size
using StaticArrays
using ..Patatrack:TrackSOA
# A seed is a track that serves as the starting point for forming a cluster.
mutable struct WorkSpace
    # Fields
    n_tracks::UInt32               # Number of "selected tracks"
    index_track::Vector{UInt16}        # Index of original track
    zt::Vector{Float32}         # Input track z at beam spot (bs)
    ezt2::Vector{Float32}       # Input error^2 on zt
    ptt2::Vector{Float32}       # Input pt^2
    izt::Vector{UInt8}          # Discretized z-position of input tracks
    iv::Vector{Int32}           # Vertex index for each associated track
    nv_intermediate::UInt32      # Number of vertices after splitting, pruning, etc.

    # Constructor
    function WorkSpace()
        new(
            UInt32(0),              # ntrks initialized to 0
            fill(UInt16(0), MAX_TRACKS), # itrk array
            zeros(Float32, MAX_TRACKS),  # zt array
            zeros(Float32, MAX_TRACKS),  # ezt2 array
            zeros(Float32, MAX_TRACKS),  # ptt2 array
            fill(UInt8(0), MAX_TRACKS),  # izt array
            fill(Int32(-1), MAX_TRACKS), # iv array (-1 for unassociated)
            UInt32(0)              # nvIntermediate initialized to 0
        )
    end
end

struct Producer 
    one_kernel::Bool
    use_density::Bool
    use_db_scan::Bool
    use_iterative::Bool
    min_T::Int32  #min number of neighbours to be "core"
    eps::Float32 #max absolute distance to cluster
    err_max::Float32 #max error to be "seed" 
    chi2_max::Float32 #max normalized distance to cluster

    function Producer(one_kernel::Bool,use_density::Bool,use_db_scan::Bool,use_iterative::Bool,i_min_T::Integer,i_eps::AbstractFloat,i_err_max,i_chi2_max)
        new(one_kernel,use_density,use_db_scan,use_iterative,i_min_T,i_eps,i_err_max,i_chi2_max)
    end
    
end
using Printf
function load_tracks(tracks,ws,pt_min)
    quality = tracks.m_quality
    fit = tracks.stateAtBS
    for idx ∈ 1:stride_track(tracks)
        n_hits = n_hits_track(tracks,idx)

        if n_hits == 0
            break
        end
        if n_hits < 4 # no triplets 
            continue 
        end
        
        if quality[idx] != loose
            continue
        end
        pt = tracks.pt[idx]

        if pt < pt_min 
            continue
        end
        ws.n_tracks += 1
        it = ws.n_tracks
        ws.index_track[it] = idx
        ws.zt[it] = zip(tracks,idx)
        ws.ezt2[it] = fit.covariance[idx,15]
        ws.ptt2[it] = pt*pt
        # @printf("%.6f",ws.ptt2[it])
        # print(" ")
    end

end

function cluster_tracks_by_density(vertices,ws,min_T,eps,err_max,chi2_max)
    verbose::Bool = false 

    err2_max = err_max^2
    n_tracks = ws.n_tracks
    zt = ws.zt
    ezt2 = ws.ezt2
    nv_final = vertices.nv_final
    nv_intermediate = ws.nv_intermediate
    izt = ws.izt
    nn = vertices.ndof
    iv = ws.iv
    hist = HisToContainer{UInt8, 256, 16000,8,UInt16}()
    fill!(hist.off,0)
    @assert n_tracks <= capacity(hist)
    for i ∈ 1:n_tracks
        @assert i <= MAX_TRACKS
        INT8_MIN = typemin(Int8)
        INT8_MAX = typemax(Int8)
        iz::Int32 = trunc(zt[i] * 10.)
        iz = min(max(iz,INT8_MIN),INT8_MAX)
        izt[i] = iz - INT8_MIN
        @assert izt[i] >= 0 
        @assert izt[i] < 256
        count!(hist,izt[i])
        iv[i] = i # associate each track with a vertex index
    end

    finalize!(hist)
    @assert size(hist) == n_tracks

    for i ∈ 1:n_tracks
        fill!(hist,izt[i],UInt16(i))
    end
    # print(Int.(hist.off[1:256]))
    #count neighbors
    for i ∈ 1:n_tracks
        if ezt2[i] > err2_max # if the uncertainty in the z position of the vertex is too high ignore it
            continue
        end 
        loop = (j) -> begin
            if i == j
                return
            end
            dist = abs(zt[i] - zt[j])
            # println("i: ",i," j: ",j)
            # println(dist)
            # println(izt[j])
            # sleep(2)
            if dist > eps
                return
            end
            if dist^2 > chi2_max * (ezt2[i] + ezt2[j])
                return
            end
            nn[i] += 1
        end
        # println(izt[i])
        for_each_in_bins(hist,izt[i],1,loop)
    end

    for i in 1:n_tracks
        mdist = eps
        loop = (j) -> begin
            if nn[j] < nn[i]
                return
            end
            if nn[j] == nn[i] && zt[j] >= zt[i]
                return  # if equal, use natural order
            end
            dist = abs(zt[i] - zt[j])
            if dist > mdist
                return
            end
            if dist^2 > chi2_max * (ezt2[i] + ezt2[j])
                return  # break natural order
            end
            mdist = dist
            iv[i] = j  # assign to cluster (should be unique)
        end
        for_each_in_bins(hist,izt[i],1,loop)
    end

    for i ∈ 1:n_tracks # consolidate graph (percolate index of seed)
        m = iv[i]
        while m != iv[m]
            m = iv[m]
        end
        iv[i] = m 
    end

    found_clusters = 0

    # find the number of different clusters, identified by a tracks with clus[i] == i and density larger than threshold;
    # mark these tracks with a negative id.

    for i ∈ 1:n_tracks
        if iv[i] == i 
            if nn[i] >= min_T
                found_clusters += 1
                iv[i] = -found_clusters
            else
                iv[i] = -9998
            end

        end
    end
    @assert found_clusters < MAX_VTX
    # propagate the negative id to all the tracks in the cluster.
    for i ∈ 1:n_tracks
        if iv[i] >= 0 
            # mark each track in a cluster with the same id as the first one
            iv[i] = iv[iv[i]]
        end
    end

    #adjust the cluster id to be a positive value starting from 1
    for i ∈ 1:n_tracks
        iv[i] = -iv[i]
        # print(iv[i], " ")
    end
    ws.nv_intermediate = vertices.nv_final = found_clusters
end

function make(self::Producer,tk_soa::TrackSOA,pt_min::AbstractFloat)::ZVertexSoA
    vertices = ZVertexSoA()
    ws = WorkSpace()
    load_tracks(tk_soa,ws,pt_min)
    cluster_tracks_by_density(vertices,ws,self.min_T,self.eps,self.err_max,self.chi2_max)
    # if(self.use_density)
    #     cluster_tracks_by_density(vertices,ws,self.min_T,self.eps,self.err_max,self.chi2_max)
    # elseif(self.use_db_scan)
    #     cluster_tracks_db_scan(vertices,ws,self.min_T,self.eps,self.err_max,self.chi2_max)
    # elseif(self.use_iterative)
    #     cluster_tracks_iterative(vertices,ws,self.min_T,self.eps,self.err_max,self.chi2_max)
    # end
    fit_vertices(vertices,ws,50.)
    split_vertices(vertices,ws,9.f0)
    fit_vertices(vertices,ws,5000.)
    sort_by_pt2(vertices,ws)
    return vertices

end

function fit_vertices(vertices::ZVertexSoA,ws::WorkSpace,chi2_max)
    nt = ws.n_tracks
    zt = ws.zt
    ezt2 = ws.ezt2
    zv = vertices.zv
    wv = vertices.wv
    chi2 = vertices.chi2
    nv_final = vertices.nv_final
    nv_intermediate = ws.nv_intermediate
    nn = vertices.ndof
    iv = ws.iv
    
    @assert nv_final <= nv_intermediate
    nv_final = nv_intermediate
    vertices.nv_final = nv_intermediate
    found_clusters = nv_final
    for i ∈ 1:found_clusters
        zv[i] = 0
        wv[i] = 0 
        chi2[i] = 0 
    end
    noise = 0 
    for i ∈ 1:nt
        if iv[i] > 9990
            noise += 1
            continue
        end
        @assert iv[i] >= 1
        # print(Int(found_clusters))
        @assert iv[i] <= Int(found_clusters)
        w = 1 / ezt2[i]
        zv[iv[i]] += zt[i]*w
        wv[iv[i]] += w
    end
    for i ∈ 1:found_clusters
        @assert wv[i] > 0 
        zv[i] /= wv[i]
        nn[i] = -1 
    end
    # compute chi2
    for i ∈ 1:nt 
        if iv[i] > 9990
            continue
        end
        c2 = zv[iv[i]] - zt[i]
        c2 *= c2 / ezt2[i]
        if c2 > chi2_max
            iv[i] = 9999
            continue
        end
        chi2[iv[i]] += c2 
        nn[iv[i]] += 1
    end
    for i ∈ 1:found_clusters
        if nn[i] > 0 
            wv[i] *= (nn[i] / chi2[i])
        end
    end
end

function split_vertices(vertices::ZVertexSoA,ws::WorkSpace,chi2_max)
    nt = ws.n_tracks # total number of loaded tracks
    zt = ws.zt  # zt of each track
    ezt2 = ws.ezt2 # variance in zt of each track
    zv = vertices.zv # z coordinate of each vertex
    wv = vertices.wv # weight associated to each vertex
    chi2 = vertices.chi2 
    nv_final = vertices.nv_final # final number of vertices
    nn = vertices.ndof # number of nearest neighbors to each track
    iv = ws.iv  # vertex association to each track
    for kv ∈ 1:nv_final
        if nn[kv] < 4 
            continue
        end
        if chi2[kv] < chi2_max * nn[kv]
            continue
        end
        
        MAX_TK = 512
        @assert nn[kv] < MAX_TK
        it = Vector{UInt32}(undef,MAX_TK) # track index
        zz = Vector{Float32}(undef,MAX_TK) # z pos
        new_v = Vector{UInt8}(undef,MAX_TK) # 0 or 1 
        ww = Vector{Float32}(undef,MAX_TK) # z weight which is 1 / variance
        nq = 0 # number of tracks for this vertex

        for k ∈ 1:nt 
            if iv[k] == kv
                nq += 1 
                zz[nq] = zt[k] - zv[kv]
                new_v[nq] = zz[nq] < 0 ? 1 : 2 
                ww[nq] = 1 / ezt2[k]
                it[nq] = k 
            end
        end
        # the new vertices
        z_new = @MArray [0f0,0f0]
        w_new = @MArray [0f0,0f0]
        @assert (nq == (nn[kv] + 1) )
        max_iter = 20
        more = true 
        while(more)
            more = false 
            z_new[1] = 0 
            z_new[2] = 0 
            w_new[1] = 0
            w_new[2] = 0 
            for k ∈ 1:nq
                i = new_v[k]
                z_new[i] += zz[k] * ww[k]
                w_new[i] += ww[k]  
            end
            z_new[1] /= w_new[1]
            z_new[2] /= w_new[2]

            for k ∈ 1:nq
                d0 = abs(z_new[1] - zz[k])
                d1 = abs(z_new[2] - zz[k])
                newer = d0 < d1 ? 1 : 2
                more |= newer != new_v[k]
                new_v[k] = newer
            end
            max_iter -= 1
            if max_iter <= 0 
                more = false 
            end
        end
        if w_new[1] == 0 || w_new[2] == 0 
            continue 
        end
        dist_2 = (z_new[1] - z_new[2]) ^ 2
        chi2_dist = dist_2 / ((1/w_new[1]) + (1/w_new[2]))
        if chi2_dist < 4 
            continue 
        end
        ws.nv_intermediate +=1
        new_vertex = ws.nv_intermediate 
        for k ∈ 1:nq
            if new_v[k] == 2
                iv[it[k]] = new_vertex
            end
        end
    end # loop on vertices
end

# using Printf
function print_vertex_validation(zvsoa::ZVertexSoA, filename::String)
    # Open the file for writing
    open(filename, "a") do io
        # Write a header line
        println(io, "VertexIndex\tz\tw\tchi2\tndof")
        # Determine the number of vertices to print (assuming nv_final indicates valid entries)
        nv = min(Int(zvsoa.nv_final), length(zvsoa.zv))
        # Loop over valid vertices and write each vertex's data on one line
        for i in 1:nv
            # Format: vertex index, z-position, weight, chi², and ndof
            @printf(io,"%d\t%.6f\t%.6f\t%.6f\t%d\n", i, zvsoa.zv[i], zvsoa.wv[i], zvsoa.chi2[i], zvsoa.ndof[i])
        end
    end
end



function sort_by_pt2(vertices::ZVertexSoA,ws::WorkSpace)
    n_tracks = ws.n_tracks
    ptt2 = ws.ptt2
    nv_final = vertices.nv_final
    iv = ws.iv
    ptv2 = vertices.ptv2
    sort_ind = vertices.sortInd

    if nv_final < 1 
        return
    end

    for i ∈ 1:n_tracks
        vertices.idv[ws.index_track[i]] = iv[i]
    end
    # for i ∈ 1:nv_final
    #     ptv2[i] = 0
    # end
    for i ∈ 1:n_tracks
        if iv[i] > 9990
            continue
        end
        # print(iv[i]-1," ")
        # print(ptt2[i]," ")
        ptv2[iv[i]] += ptt2[i]
    end
    # println()
    # println(ptv2[1:nv_final])
    # println("cccc ",nv_final)
    if nv_final == 1
        sort_ind[1] = 1 
        return
    end
    # print(ptv2[1:nv_final])
    sort_ind = collect(1:nv_final)
    sort_ind = sort(sort_ind, lt = (i, j) -> ptv2[i] < ptv2[j])
    # open("vertexValidationJulia.txt","a") do file 
    #     for i ∈ 1:nv_final
    #         print(file,sort_ind[i]," ")
            
    #     end
    #     print(file,'\n')
    # end
    # print(sort_ind.-1)
    # print_vertex_validation(vertices,"testingVertices.txt")
end




end