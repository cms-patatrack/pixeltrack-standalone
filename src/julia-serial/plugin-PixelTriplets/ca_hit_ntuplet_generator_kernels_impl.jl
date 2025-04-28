# using ..CUDACore.hist_to_container
# using ..gpuCACELL: GPUCACell, get_inner_hit_id, get_inner_r, get_inner_z, get_outer_r, get_outer_z, get_inner_det_index, are_aligned, dca_cut, add_outer_neighbor
# using ..ca_constants: OuterHitOfCell, CellNeighborsVector
# using Main:data
module kernelsImplementation
using ..CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h: TrackingRecHit2DSOAView, n_hits, detector_index
using ..gpuCACELL: GPUCACell, get_inner_hit_id, get_inner_r, get_inner_z, get_outer_r, get_outer_z, get_inner_det_index, are_aligned, dca_cut, add_outer_neighbor, find_ntuplets
using ..caConstants
using ..Patatrack: data
using DataStructures
using Printf
using ..Patatrack: CircleEq, compute, dca0, curvature
using ..Patatrack: Quality, dup, bad, loose
using ..histogram: n_bins, size, count_direct, fill_direct, tot_bins, begin_h, end_h
using ..Tracks: tip, zip
using Setfield:@set
function maxNumber()
    return 32 * 1024
end

S = maxNumber()

# HitContainer = HisToContainer{UInt32, S, 5 * S, sizeof(UInt32), 8 , UInt16, 1}

function kernel_fill_hit_indices(tuples, hh, hit_det_index)
    first = 1
    ntot = tot_bins(tuples)
    sz = size(tuples) - 1 # because that is the index of the last hit put in hist.bins. hist.off holds the starting index of the track in hist.bins so i had to add the last bins by 1
    for idx in first:ntot
        hit_det_index.off[idx] = tuples.off[idx]
    end
    num_hits = n_hits(hh)
    #   println(num_hits)
    for idx in first:sz
        @assert tuples.bins[idx] <= num_hits
        hit_det_index.bins[idx] = detector_index(hh, tuples.bins[idx])
    end
end
function test(x)
    print(x)
    while (true)
        sleep(0.1)
    end
end
function kernel_connect(hhp::TrackingRecHit2DSOAView, cells::Vector{GPUCACell}, n_cells, cell_neighbors::CellNeighborsVector, is_outer_hit_of_cell::OuterHitOfCellVector, hard_curv_cut::AbstractFloat, pt_min::AbstractFloat, ca_theta_cut_barrel::AbstractFloat, ca_theta_cut_forward::AbstractFloat, dca_cut_inner_triplet::AbstractFloat, dca_cut_outer_triplet::AbstractFloat) #=apc1::AtomicPairCounter, apc2::AtomicPairCounter,=#
    first_cell_index = 1
    first = 1
    apc1 = 0
    apc2 = 0
    # eq = CircleEq{Float32}()
    # print(n_cells[1])
    """
    Loops over all Doublets
    Gets the inner hit of the doublet
    Get all doublets where the inner hit is found as an outer hit 
    Now loop over all neighboring doublets and check whether the three points are aligned within the r z plane
    Also make sure that the curvature formed by the three points on cross sectional plane (circle) satisify a certain lower bound
    """
    for idx ∈ first_cell_index:n_cells[1]
        cell_index = idx
        this_cell = cells[cell_index]

        inner_hit_id = get_inner_hit_id(this_cell)
        outer_hit_id = this_cell.the_outer_hit_id


        number_of_possible_neighbors = length(is_outer_hit_of_cell,inner_hit_id)
        vi = is_outer_hit_of_cell[:,inner_hit_id]

        last_bpix1_det_index::UInt32 = 96
        last_barrel_det_index::UInt32 = 1184

        ri = get_inner_r(this_cell, hhp)
        zi = get_inner_z(this_cell, hhp)

        ro = get_outer_r(this_cell, hhp)
        zo = get_outer_z(this_cell, hhp)

        is_barrel = get_inner_det_index(this_cell, hhp) < last_barrel_det_index

        for j ∈ first:number_of_possible_neighbors
            other_cell_index = vi[j]
            other_cell = cells[other_cell_index]
            inner_other_cell_hit = other_cell.the_inner_hit_id
            outer_other_cell_hit = other_cell.the_outer_hit_id
            r1 = get_inner_r(other_cell, hhp)
            z1 = get_inner_z(other_cell, hhp)


            aligned = are_aligned(r1, z1, ri, zi, ro, zo, pt_min, is_barrel ? ca_theta_cut_barrel : ca_theta_cut_forward)
            cut = dca_cut(this_cell, other_cell, hhp, get_inner_det_index(other_cell, hhp) < last_bpix1_det_index ? dca_cut_inner_triplet : dca_cut_outer_triplet, hard_curv_cut, 0)
            if aligned && cut
                add_outer_neighbor(cells,other_cell_index,cell_index, cell_neighbors)
                # triplet_info = @sprintf("%d %d\n",cell_index-1,other_cell_index-1)
                # open("tripletsTestingJulia.txt","a") do file
                #     write(file,triplet_info)
                # end
                if this_cell.the_used != 1
                    this_cell = cells[cell_index]
                    cells[cell_index] = @set this_cell.the_used |= 1
                end
                if other_cell.the_used != 1
                    other_cell = cells[other_cell_index]
                    cells[other_cell_index] = @set other_cell.the_used |= 1
                end
                # this_cell.the_used |= 1
                # other_cell.the_used |= 1
            end
            # if inner_other_cell_hit == 9415 && outer_other_cell_hit == 10291
            #     println(other_cell.the_doublet_id)
            # end
        end
    end
end

function kernel_find_ntuplets(hits, cells, n_cells, cell_tracks, found_ntuplets, hit_tuple_counter, quality, min_hits_per_ntuplet,cell_neighbors)
    stack = Stack{UInt32}()
    for idx ∈ 1:n_cells[1]
        this_cell = cells[idx]
        if this_cell.the_doublet_id < 0
            continue # cut by early fishbone
        end
        p_id = this_cell.the_layer_pair_id
        p_id -= 1
        do_it = min_hits_per_ntuplet > 3 ? p_id < 3 : p_id < 8 || p_id > 12

        if do_it
            find_ntuplets(this_cell, Val{6}(), cells, cell_tracks, found_ntuplets, hit_tuple_counter, quality, stack, min_hits_per_ntuplet, p_id < 3,cell_neighbors)
            @assert isempty(stack)
        end
    end
end

function kernel_marked_used(hits, cells, n_cells)
    for idx ∈ 1:n_cells
        this_cell = cells[idx]
        if this_cell.the_tracks != 1
            cells[idx] = @set this_cell.the_used = 2
        end
    end
end

function kernel_early_duplicate_remover(cells, n_cells, found_ntuplets, quality,tracks::CellTracksVector)
    duplicate = dup
    @assert(n_cells != 0)
    for idx ∈ 1:n_cells
        this_cell = cells[idx]
        cell_tracks = tracks[:,this_cell.the_tracks]
        if length(cell_tracks) < 2
            continue
        end
        max_num_hits = 0

        for it ∈ cell_tracks
            n_h = size(found_ntuplets, it)
            max_num_hits = max(max_num_hits, n_h)
        end

        for it ∈ cell_tracks
            n_h = size(found_ntuplets, it)
            if n_h != max_num_hits
                quality[it] = duplicate
            end
        end
    end
end

function kernel_check_overflow(found_ntuplets, tuple_multiplicity, hit_tuple_counter, cells, n_cells, cell_neighbors,
    cell_tracks, is_outer_hit_of_cell, n_hits, max_number_of_doublets, counters)


end

function kernel_countMultiplicity(found_ntuplets, quality, tuple_multiplicity)
    nt = n_bins(found_ntuplets)

    for it ∈ 1:nt
        n_hits = size(found_ntuplets, it)
        if n_hits < 3
            continue
        end
        if quality[it] == dup
            continue
        end
        @assert(quality[it] == bad)
        if n_hits > 5
            @printf "Wrong mult %d %d\n" it n_hits
        end
        @assert(n_hits < 8)
        count_direct(tuple_multiplicity, n_hits)
    end
end

function kernel_fillMultiplicity(found_ntuplets, quality, tuple_multiplicity)
    nt = n_bins(found_ntuplets)
    for it ∈ 1:nt
        n_hits = size(found_ntuplets, it)
        if n_hits < 3
            continue
        end
        if quality[it] == dup
            continue
        end
        @assert(quality[it] == bad)
        if n_hits > 5
            @printf "Wrong mult %d %d\n" it n_hits
        end
        @assert(n_hits < 8)
        fill_direct(tuple_multiplicity, n_hits, UInt16(it))
    end
end

function kernel_classify_tracks(tuples,tracks,cuts,quality)
    first = 1
    nt = n_bins(tuples)
    for it ∈ first:nt
        n_hits = size(tuples,it)
        if(n_hits == 0)
            break
        end
        if(quality[it] == dup)
            continue
        end
        @assert(quality[it] == bad)
        if(n_hits < 3)
            continue
        end
        is_NAN = false
        for i ∈ 1:5
            is_NAN |= isnan(tracks.stateAtBS.state[it,i])
        end
        if(is_NAN) # leave it bad
            continue
        end
        pt = min(tracks.pt[it],cuts.chi2_max_pt)
        chi2_cut = cuts.chi2_scale * (cuts.chi2_coeff[1] + pt * (cuts.chi2_coeff[2] + pt * (cuts.chi2_coeff[3] + pt * cuts.chi2_coeff[4])))

        if(3f0 * tracks.chi2[it] >= chi2_cut)
            continue
        end
        region = (n_hits > 3) ? cuts.quadruplet : cuts.triplet
        is_ok = (abs(tip(tracks,it)) < region.max_tip) && (tracks.pt[it] > region.min_pt) && (abs(zip(tracks,it)) < region.max_zip)
        if is_ok
            quality[it] = loose
        end
    end
end


function kernel_fast_duplicate_remover(cells,n_cells,found_Ntuplets,tracks,the_cell_tracks::CellTracksVector)
    @assert(n_cells != 0)
    first = 1
    nt = n_cells[1]
    
    for idx ∈ first:nt
        this_cell = cells[idx]
        cell_tracks = the_cell_tracks[:,this_cell.the_tracks]
        if length(cell_tracks) < 2
            continue
        end
        mc = 10000f0
        im = 60000
        score = it -> abs(tip(tracks,it))
        for it_index ∈ 1:length(cell_tracks)
            it = cell_tracks[it_index]
            if tracks.m_quality[it]  == loose && score(it) < mc
                mc = score(it)
                im = it
            end
        end
        for it_index ∈ 1:length(cell_tracks)
            it = cell_tracks[it_index]
            if tracks.m_quality[it] != bad && it != im
                tracks.m_quality[it] = dup
            end
        end

    end
end

function kernel_count_hit_in_tracks(tuples,quality,hit_to_tuple)
    first = 1
    n_tot = n_bins(tuples)
    for idx ∈ first:n_tot
        if size(tuples,idx) == 0
            break
        end
        if quality[idx] != loose
            continue
        end
        for h ∈ begin_h(tuples,idx)-1:end_h(tuples,idx)-2
            count_direct(hit_to_tuple,UInt32(tuples.bins[h]))
        end
    end
end

function kernel_fill_hit_in_tracks(tuples,quality,hit_to_tuple)
    first = 1 
    n_tot = n_bins(tuples)
    for idx ∈ first:n_tot
        if size(tuples,idx) == 0
            break
        end
        if quality[idx] != loose
            continue
        end
        for h ∈ begin_h(tuples,idx)-1:end_h(tuples,idx)-2
            fill_direct(hit_to_tuple,UInt32(tuples.bins[h]),UInt16(idx))
        end
    end
end

function kernel_triplet_cleaner(hhp,p_tuples,p_tracks,quality,phi_to_tuple)
    hit_to_tuple = phi_to_tuple
    found_Ntuplets = p_tuples
    tracks = p_tracks
    n_tot = n_bins(hit_to_tuple)
    first = 1
    for idx ∈ first:n_tot
        if size(hit_to_tuple,idx) < 2
            continue 
        end
        mc = 10000f0
        im = 60000
        maxNh = 0
        # find maxNh track with max num hits
        for it ∈ begin_h(hit_to_tuple,idx):end_h(hit_to_tuple,idx)-1
            it = hit_to_tuple.bins[it]
            nh = size(found_Ntuplets,it)
            maxNh = max(maxNh,nh)
        end
        # kill all tracks shorter than maxNh
        for it ∈ begin_h(hit_to_tuple,idx):end_h(hit_to_tuple,idx)-1
            it = hit_to_tuple.bins[it]
            nh = size(found_Ntuplets,it)
            if maxNh != nh
                quality[it] = dup
            end
        end

        if maxNh > 3
            continue
        end
        # For triples, choose best tip !
        for it ∈ begin_h(hit_to_tuple,idx):end_h(hit_to_tuple,idx)-1
            it = hit_to_tuple.bins[it]
            if quality[it] != bad && abs(tip(tracks,it)) < mc
                mc = abs(tip(tracks,it))
                im = it
            end
        end
        # mark duplicates
        for it ∈ begin_h(hit_to_tuple,idx):end_h(hit_to_tuple,idx)-1
            it = hit_to_tuple.bins[it]
            if quality[it] != bad && it != im
                quality[it] = dup
            end
        end


    end 
end

end