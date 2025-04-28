module gpuPixelDoublets
function test(x)
    print(x)
    while(true)
        sleep(0.1)
    end
end
    using StaticArrays
    include("gpu_fishbone.jl")
    using ..caConstants
    using ..gpuCACELL
    using ..CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
    using ..CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h:detector_index,z_global,cluster_size_y,i_phi,r_global
    using ..histogram:hist_off,bin,begin_h,end_h,val,n_bins
    using ..Patatrack:reset!,extend!
    using Printf
    export n_pairs
    export get_doublets_from_histo
    n_pairs = 13 + 2 + 4 
    const layer_pairs = SArray{Tuple{2*n_pairs}}(
        0, 1, 0, 4, 0, 7,              # BPIX1 (3)
        1, 2, 1, 4, 1, 7,              # BPIX2 (5)
        4, 5, 7, 8,                    # FPIX1 (8)
        2, 3, 2, 4, 2, 7, 5, 6, 8, 9,  # BPIX3 & FPIX2 (13)
        0, 2, 1, 3,                    # Jumping Barrel (15)
        0, 5, 0, 8,                    # Jumping Forward (BPIX1,FPIX2)
        4, 6, 7, 9                     # Jumping Forward (19)
    )
    const phi0p05::Int16 = 522 #round(521.52189...) = phi2short(0.05);
    const phi0p06::Int16 = 626 #round(625.82270...) = phi2short(0.06);
    const phi0p07::Int16 = 730 #round(730.12648...) = phi2short(0.07);
    const phi_cuts = SArray{Tuple{n_pairs}}(
        phi0p05,
        phi0p07,
        phi0p07,
        phi0p05,
        phi0p06,
        phi0p06,
        phi0p05,
        phi0p05,
        phi0p06,
        phi0p06,
        phi0p06,
        phi0p05,
        phi0p05,
        phi0p05,
        phi0p05,
        phi0p05,
        phi0p05,
        phi0p05,
        phi0p05
    )

    const min_z = SArray{Tuple{n_pairs}}(-20., 0., -30., -22., 10., -30., -70., -70., -22., 15., -30, -70., -70., -20., -22., 0, -30., -70., -70.)
    const max_z = SArray{Tuple{n_pairs}}(20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.)
    const max_r = SArray{Tuple{n_pairs}}(20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.)

    """
        Assuming Zero impact parameter, ensures that the calculated value for pt satisfies the lower bound
    """
    function pt_cut(j, idphi, r2t4, ri, hh)
        ro = r_global(hist_view(hh), j)
        dϕ = idphi * ((2 * π) / (1 << 16))
        return dϕ^2 * (r2t4 - ri * ro) > (ro - ri)^2
    end
    """
    Determines if the z0 cut condition is satisfied based on the line in the rz plane formed by the inner and outer hits.
    """
    function z0_cut_off(j, i, ri, zi, max_r, z0_cut, hh, pair_layer_id)
        zo = z_global(hist_view(hh), j)
        ro = r_global(hist_view(hh), j)
        dr = ro - ri
        return dr > max_r[pair_layer_id] || dr < 0 || abs(zi * ro - ri * zo) > z0_cut * dr
    end
    """
            Delta y of inner and outer hit must not be bigger than some upper bound. Depends on layer pair.
            Here, if both hits are on barrrels, or if the inner hit is on a barrel, and the outer hit is on a disk.
    """
    function z_size_cut(j, hh, outer, inner, me_s, zi, ri, max_dy_size_12, max_dy_size,  dz_dr_fact::Float32, max_dy_pred)
        only_barrel = outer < 4
        so = cluster_size_y(hist_view(hh), j)           # Cluster size in y for the outer hit
        dy = (inner == 0) ? max_dy_size_12 : max_dy_size
        zo = z_global(hist_view(hh), j)                 # Z-position for the outer hit
        ro = r_global(hist_view(hh), j)                 # Radius for the outer hit
    
        return only_barrel ? (me_s > 0 && so > 0 && abs(so - me_s) > dy) :
                             (inner < 4) && (me_s > 0) && abs(me_s - Int(trunc(abs((zi - zo) / (ri - ro)) * dz_dr_fact + 0.5))) > max_dy_pred
    end

    
    function init_doublets(is_outer_hit_of_cell::OuterHitOfCellVector,n_hits::Integer,cell_neighbors::CellNeighborsVector,
                           cell_tracks::CellTracksVector)
        # @assert(!isempty(is_outer_hit_of_cell))
        first = 1
        # for i ∈ first:n_hits
        #     reset!(is_outer_hit_of_cell[i])
        # end
        i = extend!(cell_neighbors)
        @assert(i == 1)
        # reset!(cell_neighbors[1])
        i = extend!(cell_tracks)
        @assert(i == 1)
        # reset!(cell_tracks[1])
        # extend!(cell_neighbors)
        # extend!(cell_tracks)
    end
    function get_doublets_from_histo(cells::Vector{GPUCACell},n_cells,cell_neighbors::CellNeighborsVector,cell_tracks::CellTracksVector,
                                     hhp::TrackingRecHit2DHeterogeneous,is_outer_hit_of_cell::OuterHitOfCellVector,n_actual_pairs::Integer,
                                     ideal_cond::Bool,do_cluster_cut::Bool,do_z0_cut::Bool,do_pt_cut::Bool,max_num_of_doublets::Integer)
        doublets_from_histo(layer_pairs,n_actual_pairs,cells,n_cells,cell_neighbors,
        cell_tracks,hhp,is_outer_hit_of_cell,phi_cuts,min_z,max_z,max_r,ideal_cond,
        do_cluster_cut,do_z0_cut,do_pt_cut,max_num_of_doublets)
    end
    """
    layer_pairs: vector containing encoding of layer pairs
    n_pairs: number of actual layer pairs to go over
    cells: Vector of GPUCACell representing Doublets
    n_cells: set to 0 initially should increase while processing
    is_outer_hit_of_cell: a vector of vectors for every outer hit which holds indices to all doublets that it participates in
    """
    function doublets_from_histo(layer_pairs::SArray{Tuple{layer_pairs_2}},n_pairs::Integer,cells::Vector{GPUCACell},n_cells,cell_neighbors::CellNeighborsVector,
                                 cell_tracks::CellTracksVector,hh::TrackingRecHit2DHeterogeneous,is_outer_hit_of_cell::OuterHitOfCellVector,phi_cuts:: SArray{Tuple{n_layer_pairs}},
                                 min_z::SArray{Tuple{n_layer_pairs}},max_z::SArray{Tuple{n_layer_pairs}},max_r::SArray{Tuple{n_layer_pairs}},ideal_cond::Bool,do_cluster_cut::Bool,do_z0_cut::Bool,
                                 do_pt_cut::Bool,max_num_of_doublets::Integer) where {n_layer_pairs, layer_pairs_2}
        """
        Used for filtering doublets based on the y-size comparision of the hit[1]s
        """
        ###
        min_y_size_B1 = 36
        min_y_size_B2 = 28
        max_dy_size_12 = 28
        max_dy_size = 20
        max_dy_pred = 20
        dz_dr_fact::Float32 = 8 * 0.0285 / 0.015 # from dz/dr to "DY"
        ### used when do_cluster_cut is set to true
        
        is_outer_ladder = ideal_cond
        hist = phi_binner(hh)
        offsets = hits_layer_start(hh)
        @assert(!isempty(offsets))
        """
        lambda function layer_size: returns the total number of hits for a particular layer : li
        """
        layer_size = let offsets = offsets 
            li -> offsets[li+1] - offsets[li]
        end

        n_pairs_max = MAX_NUM_OF_LAYER_PAIRS
        """
        Ensure that the type of the pair (layer i -> layer j) is within the maximum bound MAX_NUM_OF_LAYER_PAIRS
        defined in cluster constants
        """
        @assert(n_pairs <= n_pairs_max)
        """
        used for determining for indexed hits which pair_layer_id to associate it with. Example:
            If the index of a hit i was >= inner_layer_cumulative_size[pair_layer_id] but < inner_layer_cumulative_size[pair_layer_id+1],
            then we associate it with that specific pair_layer_id
        """
        
        inner_layer_cumulative_size = MArray{Tuple{n_pairs_max},UInt32}(undef)
        n_tot = 0
        inner_layer_cumulative_size[1] = layer_size(layer_pairs[1]+1)
        
        """
        To fill the inner_layer_cumulative_size array, we need to search for the number of hits for the inner layer of the pair labeled with i.
        The layerPairs array contains all pairs p1, p2, p3, p4, ... Each pair contains two integers the first consisting of the inner layer, the 
        second consisting of the outer layer. So layerPairs contains p1.first, p1.second, p2.first, p2.second, p3.first, p3.second etc...
        To find the inner layer index of the i'th pair, we look at index 2*i - 1. We add 1 because julia indexing starts at 1 and we have them encoded starting from 0 within layer_pairs
        """
        for i ∈ 2:n_pairs
            inner_layer_cumulative_size[i] = inner_layer_cumulative_size[i-1] + layer_size(layer_pairs[2*i-1]+1)
        end
        
        """
        n_total : The total number of inner hits
        """
        n_tot = inner_layer_cumulative_size[n_pairs]
        
        idy = 0
        first = 0 
        stride = 1
        pair_layer_id = 1 
        """
        Go over all inner hits by indexing them from 0 to the total number of inner hits - 1
        """
        
        for j ∈ idy:n_tot-1
            """
            update the pair_layer_id to ensure the correct type of pair the hit j belongs to.
            I don't think a while loop is needed here. A regular if statement would suffice.
            """
            while(j >= inner_layer_cumulative_size[pair_layer_id])
                pair_layer_id+=1
            end
            @assert(pair_layer_id <= n_pairs)
            @assert(j < inner_layer_cumulative_size[pair_layer_id])
            @assert(pair_layer_id == 1 || j >= inner_layer_cumulative_size[pair_layer_id-1])
            
            """
            find the inner and outer indices for the layers 
            """
            inner = layer_pairs[2*pair_layer_id-1]
            outer = layer_pairs[2*pair_layer_id]
            @assert(outer > inner)
            h_off = hist_off(Hist,outer)
            """
            i is the index of the inner hit within the hits struct
            inner + 2 would never overflow becauses we are considering a hit on an inner layer
            """
            i = (1 == pair_layer_id) ? j : j - inner_layer_cumulative_size[pair_layer_id-1] # bounded i between 0 and total number of hits on inner layer
            i += offsets[inner+1] # +1 because indexed from 1 in julia
            @assert(i >= offsets[inner + 1]) # +1 because of indexing in julia
            @assert(i < offsets[inner + 2])
            i+=1
            mi = detector_index(hist_view(hh),i)
            if(mi > 2000)
                continue 
            end
            me_z = z_global(hist_view(hh),i)
            if (me_z < min_z[pair_layer_id] || me_z > max_z[pair_layer_id])
                continue;
            end
            me_s = -1
            """
            Modules on barrels are found on ladders which mount 8 modules each.
            They are alternating. Here, the size of a hit on the y-direction must satisfy a lower bound.
            inner hits on outer ladders on BPIX1 and BPIX2 are tested to meet the lower bound
            """
            if do_cluster_cut
                if inner == 0 
                    @assert(mi < 96)
                end
                is_outer_ladder = ideal_cond ? true : ((mi ÷ 8 ) % 2) == 0

                #Always test me_s > 0 
                me_s = (inner > 0 || is_outer_ladder) ? cluster_size_y(hist_view(hh),i) : -1
                
                if (inner == 0 && outer > 3) && (me_s > 0 && me_s < min_y_size_B1)# B1 and F1
                    continue
                end

                if (inner == 1 && outer > 3) && (me_s > 0 && me_s < min_y_size_B2)# B2 and F1
                    continue
                end
            end

            me_p = i_phi(hist_view(hh),i)
            me_r = r_global(hist_view(hh),i)
            
            z0_cut = 12.f0 # cm
            hard_pt_cut = 0.5 # GeV
            min_radius = hard_pt_cut * 87.78 # cm ( 1 GeV track has 1 GeV/c / (e * 3.8 T) ~ 87 cm radius in a 3.8 T field)
            min_radius_2T4 = 4. * min_radius * min_radius
            
            # """
            # Assuming Zero impact parameter, ensures that the calculated value for pt satisfies the lower bound
            # """
            # pt_cut = let r2t4 = min_radius_2T4 , ri = me_r, hh = hh
            #     (j,idphi) -> begin
            #         ro = r_global(hist_view(hh),j)
            #         dϕ = idphi * ((2*π)/(1<<16))
            #         return dϕ^2 * (r2t4 - ri*ro) > (ro - ri)^2
            #     end
            # end
            # """
            # Ensure that the value Z0 the line in the rz plane formed by inner and outer hit is < z0_cut = 12 cm
            # """
            # z0_cut_off = let ri = me_r , zi = me_z, max_r = max_r, z0_cut = z0_cut, hh = hh, pair_layer_id = pair_layer_id
            #     (j,i) -> begin
            #     zo = z_global(hist_view(hh),j)
            #     ro = r_global(hist_view(hh),j)
            #     dr = ro - ri
            #     # if i == 2870 && j == 5793
            #     #         @printf("%.16f\n",abs(zi*ro - ri*zo))
            #     #         @printf("%.16f\n",z0_cut * dr)
            #     #         println(typeof(dr))
            #     #         println(typeof(z0_cut))
            #     #     end
            #     return dr > max_r[pair_layer_id] || dr < 0 || abs(zi*ro - ri*zo) > z0_cut * dr
            #     end
            # end
            # """
            # Delta y of inner and outer hit must not be bigger than some upper bound. Depends on layer pair.
            # Here, if both hits are on barrrels, or if the inner hit is on a barrel, and the outer hit is on a disk.
            # """
            # z_size_cut = let hh = hh , outer = outer, inner = inner, max_dy_size_12 = max_dy_size_12, max_dy_size = max_dy_size, me_s = me_s, zi = me_z, ri = me_r, dz_dr_fact::Float32 = dz_dr_fact::Float32, max_dy_pred = max_dy_pred
            #     (j) -> begin
            #         only_barrel = outer < 4
            #         so = cluster_size_y(hist_view(hh),j)
            #         dy = (inner == 0 ) ? max_dy_size_12 : max_dy_size
            #         zo = z_global(hist_view(hh),j)
            #         ro = r_global(hist_view(hh),j)
            #         return only_barrel ? (me_s > 0 && so > 0 && abs(so - me_s) > dy) : (inner < 4 ) && (me_s > 0 ) && abs(me_s - Int(trunc(abs((zi - zo)/(ri - ro))*dz_dr_fact::Float32 + 0.5))) > max_dy_pred
            #     end
            # end
            """
            Consider hits that are delta phi away from inner hit.
            """
            
            i_phi_cut = phi_cuts[pair_layer_id]

            kl = bin(hist,Int16(me_p-i_phi_cut))
            kh = bin(hist,Int16(me_p+i_phi_cut))
            kh += 1
            if(kh == (n_bins(hist)+1))
                kh = 1 
            end
            current_bin = kl
            
            while(current_bin != kh)
                
                p = begin_h(hist,current_bin+h_off)# first index of current_bin
                e = end_h(hist,current_bin+h_off) # first index of current_bin+1
                p += first
                for p ∈ p:e-1
                    
                    oi = val(hist,p)
                    @assert(oi > offsets[outer+1])
                    @assert(oi <= offsets[outer+2])
                    
                    mo = detector_index(hist_view(hh),oi)
                    if (mo > 2000)
                        continue
                    end
                    
                    if (do_z0_cut && z0_cut_off(oi,i,me_r,me_z,max_r,z0_cut,hh,pair_layer_id))
                        continue
                    end
                    mo_p = i_phi(hist_view(hh),oi)
                    i_dphi = abs(mo_p - me_p)
                    if i_dphi > i_phi_cut
                        continue
                    end
                    if do_cluster_cut && z_size_cut(oi,hh,outer,inner,me_s,me_z,me_r,max_dy_size_12,max_dy_size,dz_dr_fact,max_dy_pred)
                        continue
                    end

                    if do_pt_cut && pt_cut(oi,i_dphi,min_radius_2T4,me_r,hh)
                        continue
                    end
                    
                    n_cells[1]+=UInt32(1)
                    
                    if(n_cells[1] > max_num_of_doublets)
                        n_cells[1] -=UInt32(1)
                        break
                    end
                    cells[n_cells[1]] = GPUCACell(cell_neighbors,cell_tracks,hist_view(hh),pair_layer_id,n_cells[1],i,oi)
                    push!(is_outer_hit_of_cell,oi,n_cells[1])
                end
                current_bin = current_bin+1
                if(current_bin == (n_bins(hist)+1))
                    current_bin = 1 
                end
            end

            
        end 

    end
end