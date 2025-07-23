module cAHitNtupletGenerator
# using ..CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
using ..CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h: n_hits, TrackingRecHit2DHeterogeneous, hist_view
using ..CUDADataFormatsSiPixelClusterInterfaceGPUClusteringConstants
using StaticArrays: MArray, MVector
using ..caConstants
using ..gpuCACELL
using ..gpuPixelDoublets: init_doublets, n_pairs, get_doublets_from_histo, fish_bone
using ..kernelsImplementation: kernel_connect, kernel_find_ntuplets, kernel_marked_used, kernel_early_duplicate_remover, kernel_countMultiplicity, kernel_fillMultiplicity, kernel_fill_hit_indices, kernel_classify_tracks, kernel_count_hit_in_tracks, kernel_fill_hit_in_tracks, kernel_triplet_cleaner, kernel_fast_duplicate_remover
using ..histogram: zero, bulk_finalize_fill, n_bins, finalize!
using ..Patatrack: reset!
export Params, Counters
#using Main::kernel_fill_hit_indices
struct Counters
    n_events::UInt64
    n_hits::UInt64
    n_cells::UInt64
    n_tuples::UInt64
    n_fit_tracks::UInt64
    n_good_tracks::UInt64
    n_used_hits::UInt64
    n_dup_hits::UInt64
    n_killed_cells::UInt64
    n_empty_cells::UInt64
    n_zero_track_cells::UInt64
    Counters() = new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
end
# const HitsView = TrackingRecHit2DSOAView
const HitsOnCPU = TrackingRecHit2DHeterogeneous

struct Region
    max_tip::Float32 # cm
    min_pt::Float32 # Gev
    max_zip::Float32 # cm
end
struct QualityCuts
    # chi2 cut = chi2Scale * (chi2Coeff[0] + pT/GeV * (chi2Coeff[1] + pT/GeV * (chi2Coeff[2] + pT/GeV * chi2Coeff[3])))
    chi2_coeff::MArray{Tuple{4},Float32}
    chi2_max_pt::Float32
    chi2_scale::Float32
    triplet::Region
    quadruplet::Region
end

struct Params
    on_gpu::Bool
    min_hits_per_ntuplet::UInt32
    max_num_of_doublets::UInt32
    use_riemann_fit::Bool
    fit_5_as_4::Bool
    include_jumping_forward_doublets::Bool
    early_fish_bone::Bool
    late_fish_bone::Bool
    ideal_conditions::Bool
    do_stats::Bool
    do_cluster_cut::Bool
    do_z0_cut::Bool
    do_pt_cut::Bool
    pt_min::Float32
    ca_theta_cut_barrel::Float32
    ca_theta_cut_forward::Float32
    hard_curv_cut::Float32
    dca_cut_inner_triplet::Float32
    dca_cut_outer_triplet::Float32
    cuts::QualityCuts
    function Params(on_gpu::Bool, min_hits_per_ntuplet::Integer, max_num_of_doublets::Integer, use_riemann_fit::Bool,
        fit_5_as_4::Bool, include_jumping_forward_doublets::Bool, early_fish_bone::Bool, late_fish_bone::Bool,
        ideal_conditions::Bool, do_stats::Bool, do_cluster_cut::Bool, do_z0_cut::Bool, do_pt_cut::Bool,
        pt_min::AbstractFloat, ca_theta_cut_barrel::AbstractFloat, ca_theta_cut_forward::AbstractFloat, hard_curv_cut::AbstractFloat,
        dca_cut_inner_triplet::AbstractFloat, dca_cut_outer_triplet::AbstractFloat, cuts::QualityCuts)
        new(on_gpu, min_hits_per_ntuplet, max_num_of_doublets, use_riemann_fit,
            fit_5_as_4, include_jumping_forward_doublets, early_fish_bone, late_fish_bone,
            ideal_conditions, do_stats, do_cluster_cut, do_z0_cut, do_pt_cut,
            pt_min, ca_theta_cut_barrel, ca_theta_cut_forward, hard_curv_cut,
            dca_cut_inner_triplet, dca_cut_outer_triplet, cuts)
    end
end

cuts = QualityCuts(MArray{Tuple{4},Float32}((0.68177776, 0.74609577, -0.08035491, 0.00315399)),# polynomial coefficients for the pT-dependent chi2 cut
    10.0,                                                                        # max pT used to determine the chi2 cut
    30.0,                                                                        # chi2 scale factor: 30 for Broken line Fit, 45 for Riemann Fit
    Region(0.3, # |Tip| < 0.3 cm                                                # Regional cuts for Triplets
        0.5, # pT > 0.5 GeV
        12.0), # |Zip| < 12.0 cm
    Region(0.5, # |Tip| < 0.5 cm                                                # Reginal cuts for quadruplets    
        0.3, # pT > 0.3 GeV
        12.0)) # |Zip| < 12.0 cm


mutable struct CAHitNTupletGeneratorKernels
    #cell_storage::Vector{UInt8}
    device_the_cell_neighbors::CellNeighborsVector
    # device_the_cell_neighbors_container::Vector{CellNeighbors}
    device_the_cell_tracks::CellTracksVector
    # device_the_cell_tracks_container::Vector{CellTracks}
    device_the_cells::Vector{GPUCACell}
    device_is_outer_hit_of_cell::OuterHitOfCellVector
    device_n_cells::MVector{1,UInt32}
    device_hit_tuple_counter::MVector{2,UInt32}
    device_hit_to_tuple::HitToTuple
    device_tuple_multiplicity::TupleMultiplicity
    m_params::Params
    counters::Counters
    function CAHitNTupletGeneratorKernels(params::Params)
        # is_outer_hit_of_cell = [OuterHitOfCell() for _ ∈ 1:15e4]
        is_outer_hit_of_cell = OuterHitOfCellVector(0,0)
        # the_cell_neighbors_container = [CellNeighbors() for _ ∈ 1:MAX_NUM_OF_ACTIVE_DOUBLETS]# Vector of CellNeighbors
        the_cell_neighbors_container = CellNeighborsVector(36,MAX_NUM_OF_ACTIVE_DOUBLETS)
        # the_cell_tracks_container = [CellTracks() for _ ∈ 1:MAX_NUM_OF_ACTIVE_DOUBLETS]# Vector of CellTracks
        the_cell_tracks_container = CellTracksVector(48,MAX_NUM_OF_ACTIVE_DOUBLETS)
        the_cells = Vector{GPUCACell}(undef, params.max_num_of_doublets)
        device_hit_to_tuple = HitToTuple()
        device_tuple_multiplicity = TupleMultiplicity()
        zero(device_hit_to_tuple)
        zero(device_tuple_multiplicity)
        new(the_cell_neighbors_container,
            the_cell_tracks_container, the_cells, is_outer_hit_of_cell,
            MVector{1,UInt32}(0), MVector{2,UInt32}(0, 0), device_hit_to_tuple, device_tuple_multiplicity, params, Counters())
    end
end
function resetCAHitNTupletGeneratorKernels(self)
    # for idx ∈ 1:MAX_NUM_OF_ACTIVE_DOUBLETS
    #     reset!(self.device_the_cell_neighbors[idx])
    #     reset!(self.device_the_cell_tracks[idx])
    #     if idx <= 15e4
    #         reset!(self.device_is_outer_hit_of_cell[idx])
    #     end
    # end
    reset!(self.device_the_cell_neighbors)
    reset!(self.device_the_cell_tracks)
    self.device_n_cells[1] = 0
    self.device_hit_tuple_counter[1] = 0
    self.device_hit_tuple_counter[2] = 0
    zero(self.device_hit_to_tuple)
    zero(self.device_tuple_multiplicity)
end
function fill_hit_det_indices(hv, tracks_d)
    kernel_fill_hit_indices(tracks_d.hit_indices, hv, tracks_d.det_indices)
end

function build_doublets(self::CAHitNTupletGeneratorKernels, hh::HitsOnCPU)
    current_n_hits = n_hits(hh)
    # println("Building Doublets out of ", current_n_hits, " Hits")
    # cell_storage
    # @timev self.device_is_outer_hit_of_cell = [OuterHitOfCell() for _ ∈ 1:max(1,current_n_hits)]
    self.device_is_outer_hit_of_cell = OuterHitOfCellVector(MAX_CELLS_PER_HIT,current_n_hits)
    init_doublets(self.device_is_outer_hit_of_cell, current_n_hits, self.device_the_cell_neighbors, self.device_the_cell_tracks)
    if (current_n_hits == 0)
        return
    end

    n_actual_pairs = n_pairs

    if (!self.m_params.include_jumping_forward_doublets)
        n_actual_pairs = 15
    end

    if (self.m_params.min_hits_per_ntuplet > 3)
        n_actual_pairs = 13
    end

    @assert(n_actual_pairs <= n_pairs)
    get_doublets_from_histo(self.device_the_cells, self.device_n_cells, self.device_the_cell_neighbors, self.device_the_cell_tracks, hh,
        self.device_is_outer_hit_of_cell, n_actual_pairs, self.m_params.ideal_conditions, self.m_params.do_cluster_cut,
        self.m_params.do_z0_cut, self.m_params.do_pt_cut, self.m_params.max_num_of_doublets)
end

function launch_kernels(self::CAHitNTupletGeneratorKernels, hh, tracks_d)
    tuples_d = tracks_d.hit_indices
    quality_d = tracks_d.m_quality
    zero(tuples_d)
    num_hits = n_hits(hh)
    @assert(num_hits <= MAX_NUMBER_OF_HITS)
    kernel_connect(hist_view(hh), self.device_the_cells, self.device_n_cells, self.device_the_cell_neighbors,
        self.device_is_outer_hit_of_cell, self.m_params.hard_curv_cut, self.m_params.pt_min,
        self.m_params.ca_theta_cut_barrel, self.m_params.ca_theta_cut_forward, self.m_params.dca_cut_inner_triplet,
        self.m_params.dca_cut_outer_triplet)
    if num_hits > 1 && self.m_params.early_fish_bone
        fish_bone(hist_view(hh), self.device_the_cells, self.device_n_cells, self.device_is_outer_hit_of_cell, num_hits, false)
    end
    kernel_find_ntuplets(hist_view(hh), self.device_the_cells, self.device_n_cells, self.device_the_cell_tracks, tuples_d, self.device_hit_tuple_counter, quality_d, self.m_params.min_hits_per_ntuplet,self.device_the_cell_neighbors)
    if self.m_params.do_stats
        kernel_mark_used((hist_view(hh), self.device_the_cells, self.device_n_cells))
    end
    bulk_finalize_fill(tuples_d, self.device_hit_tuple_counter)
    kernel_early_duplicate_remover(self.device_the_cells, self.device_n_cells[1], tuples_d, quality_d,self.device_the_cell_tracks)

    kernel_countMultiplicity(tuples_d, quality_d, self.device_tuple_multiplicity)
    finalize!(self.device_tuple_multiplicity)

    # self.device_tuple_multiplicity.bins[self.device_tuple_multiplicity.off[3]+i] changes here!
    kernel_fillMultiplicity(tuples_d, quality_d, self.device_tuple_multiplicity)

    if num_hits > 1 && self.m_params.late_fish_bone
        fish_bone(hist_view(hh), self.device_the_cells, self.device_n_cells[1], self.device_is_outer_hit_of_cell, num_hits, true)
    end

    if (self.m_params.do_stats)
        kernel_check_overflow(tuples_d, self.device_tuple_multiplicity, self.device_hit_to_tuple, self.device_the_cells, self.device_n_cells,
            self.device_the_cell_neighbors, self.device_the_cell_tracks, self.device_is_outer_hit_of_cell, n_hits,
            self.m_params.max_num_of_doublets, self.counters)
    end
    # for cell ∈ self.device_is_outer_hit_of_cell[2733]
    #     if self.device_the_cells[cell].the_inner_hit_id == 650
    #         print(self.device_the_cells[cell].the_outer_neighbors)
    #     end
    # end
    # checking_hist = IOBuffer()
    # for it ∈ 1:self.device_hit_tuple_counter[1]
    #     print(checking_hist,it)
    #     for index ∈ tuples_d.off[it]:tuples_d.off[it+1]-1
    #         print(checking_hist," ",tuples_d.bins[index]," ")
    #     end
    #     print(checking_hist,"\n")
    # end
    # buff = String(take!(checking_hist))
    # open("hist_tesitng.txt","w") do file
    #     print(file,buff)
    # end
end

function classify_tuples(self::CAHitNTupletGeneratorKernels, hh::HitsOnCPU, tracks_d,the_cell_tracks::CellTracksVector)
    tuples_d = tracks_d.hit_indices
    quality_d = tracks_d.m_quality
    kernel_classify_tracks(tuples_d, tracks_d, self.m_params.cuts, quality_d)
    kernel_fast_duplicate_remover(self.device_the_cells, self.device_n_cells, tuples_d, tracks_d,the_cell_tracks)
    kernel_count_hit_in_tracks(tuples_d, quality_d, self.device_hit_to_tuple)
    finalize!(self.device_hit_to_tuple)
    kernel_fill_hit_in_tracks(tuples_d, quality_d, self.device_hit_to_tuple)
    kernel_triplet_cleaner(hist_view(hh), tuples_d, tracks_d, quality_d, self.device_hit_to_tuple)
end

end
