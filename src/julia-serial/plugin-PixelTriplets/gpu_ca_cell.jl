module gpuCACELL
    using InteractiveUtils
    using StaticArrays
    using ..caConstants
    using ..CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
    using ..Patatrack:VecArray
    using ..Patatrack:SimpleVector
    using ..Patatrack:empty,extend!,reset!
    using Printf
    const ptr_as_int = UInt64
    const Hits = TrackingRecHit2DSOAView
    const TmpTuple = VecArray{UInt32,6}
    using ..CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h:z_global,r_global
    using ..Patatrack:CircleEq, compute, dca0, curvature
    using ..histogram:bulk_fill
    using Setfield:@set
    import ..Patatrack:Quality,bad
    export GPUCACell
    export get_outer_x,get_outer_y,get_outer_z,get_inner_x,get_inner_y,get_inner_z,get_inner_det_index
    # using Main:CircleEq
    # using Main:curvature
    # using Main:dca0%
    # using Main:extend!
    # using Main:reset
    # using Main:push!
    # using Main:empty

    struct GPUCACell
        the_outer_neighbors::UInt32
        the_tracks::UInt32
        the_doublet_id::Int32
        the_layer_pair_id::Int16
        the_used::UInt16
        the_inner_z::Float32
        the_inner_r::Float32
        the_inner_hit_id::hindex_type
        the_outer_hit_id::hindex_type
    end
    function GPUCACell(cell_neighbors::CellNeighborsVector,cell_tracks::CellTracksVector,hh::Hits,layer_pair_id::Integer,doublet_id::Integer,
        inner_hit_id::Integer,outer_hit_id::Integer)
        z_global_inner = z_global(hh,inner_hit_id)
        r_global_inner = r_global(hh,inner_hit_id)
        # z_global_inner_str = @sprintf("%.7g", z_global_inner)
        # r_global_inner_str = @sprintf("%.7g", r_global_inner)
        # Construct the string to append to the file
        # doublet_id -=1
        # layer_pair_id -=1
        # inner_hit_id -= 1
        # outer_hit_id -= 1

        # output_string = @sprintf("doublet_id: %d, layer_pair_id: %d, z_global_inner: %s, r_global_inner: %s, inner_hit_id: %d, outer_hit_id: %d\n",
        # doublet_id, layer_pair_id, z_global_inner_str, r_global_inner_str, inner_hit_id, outer_hit_id)
        # doublet_id +=1
        # layer_pair_id +=1
        # inner_hit_id += 1
        # outer_hit_id += 1
        # Open the file in append mode and write the output_string
        
        #write(file, output_string)
        
        GPUCACell(UInt32(1),UInt32(1),doublet_id,layer_pair_id,0,z_global_inner,r_global_inner,inner_hit_id,outer_hit_id)
    end
    print_cell(self::GPUCACell) = @printf("printing cell: %d, on layerPair: %d, innerHitId: %d, outerHitId: %d \n",
           theDoubletId,
           theLayerPairId,
           theInnerHitId,
           theOuterHitId)
    
    function init(self::GPUCACell,cell_neighbors::CellNeighborsVector,cell_tracks::CellTracksVector,hh::Hits,layer_pair_id::Integer,doublet_id::Integer,
                  inner_hit_id::Integer,outer_hit_id::Integer)
        self.the_inner_hit_id = inner_hit_id
        self.the_outer_hit_id = outer_hit_id
        self.the_doublet_id = doublet_id
        self.the_layer_pair_id = layer_pair_id
        self.the_used = 0 
        self.the_inner_r = r_global(hh,inner_hit_id)
        self.the_inner_z = z_global(hh,inner_hit_id)
        self.the_outer_neighbors = cell_neighbors[1]
        self.the_tracks = cell_tracks[1]
        #@assert()
        #@assert()
    end
    


function get_inner_hit_id(self::GPUCACell)
    return self.the_inner_hit_id
end

function get_inner_x(self::GPUCACell, hh::TrackingRecHit2DSOAView)
    return x_global(hh, self.the_inner_hit_id)
end

function get_inner_y(self::GPUCACell, hh::TrackingRecHit2DSOAView)
    return y_global(hh, self.the_inner_hit_id)
end

function get_outer_x(self::GPUCACell, hh::TrackingRecHit2DSOAView)
    return x_global(hh, self.the_outer_hit_id)
end

function get_outer_y(self::GPUCACell, hh::TrackingRecHit2DSOAView)
    return y_global(hh, self.the_outer_hit_id)
end

function get_inner_r(self::GPUCACell, hh::TrackingRecHit2DSOAView)
    return self.the_inner_r
end

function get_inner_z(self::GPUCACell, hh::TrackingRecHit2DSOAView)
    return self.the_inner_z
end

function get_outer_r(self::GPUCACell, hh::TrackingRecHit2DSOAView)
    return r_global(hh, self.the_outer_hit_id)
end

function get_outer_z(self::GPUCACell, hh::TrackingRecHit2DSOAView)
    return z_global(hh, self.the_outer_hit_id)
end

function get_inner_det_index(self::GPUCACell, hh::TrackingRecHit2DSOAView)
    return detector_index(hh, self.the_inner_hit_id)
end

function get_outer_det_index(self::GPUCACell, hh::TrackingRecHit2DSOAView)
    return detector_index(hh, self.the_outer_hit_id)
end

function are_aligned(r1, z1, ri, zi, ro, zo, pt_min, theta_cut)
    radius_diff = abs(r1 - ro)
    distance_13_squared = radius_diff * radius_diff + (z1 - zo) * (z1 - zo)
    p_min = pt_min * √(distance_13_squared)
    tan_12_13_half_mul_distance_13_squared = abs(z1 * (ri - ro) + zi * (ro - r1) + zo * (r1 - ri))
    return tan_12_13_half_mul_distance_13_squared * p_min <= theta_cut * distance_13_squared * radius_diff
end

function dca_cut(cell::GPUCACell, other_cell::GPUCACell, hh::TrackingRecHit2DSOAView, region_origin_radius_plus_tolerance::AbstractFloat, max_curv::AbstractFloat,eq2)
    x1 = get_inner_x(other_cell, hh)
    y1 = get_inner_y(other_cell, hh)

    x2 = get_inner_x(cell, hh)
    y2 = get_inner_y(cell, hh)

    x3 = get_outer_x(cell, hh)
    y3 = get_outer_y(cell, hh)
    eq = CircleEq{Float32}(x1, y1, x2, y2, x3, y3)
    # compute(eq,x1, y1, x2, y2, x3, y3)
    curvature_c = curvature(eq)
    if curvature_c > max_curv
        return false
    end
    return abs(dca0(eq)) < region_origin_radius_plus_tolerance * abs(curvature_c)
end

function outer_neighbors(self::GPUCACell)
    return self.the_outer_neighbors
end
"""
check if oughter_neighbor vector for the other_doublet if it is empty
if its empty, assign for it a neighbors slot within cell_neighbors by extending cell_neighbors.
Finally push the second doublet index t to the oughter_neighbor forming the triplet
"""
function add_outer_neighbor(cells::Vector{GPUCACell},cell_index::Integer, t::Integer, cell_neighbors::CellNeighborsVector)
    neighbors_index = cells[cell_index].the_outer_neighbors
    if neighbors_index == 1
        i = extend!(cell_neighbors)
        if i > 1
            # reset!(cell_neighbors[i])
            temp_cell = cells[cell_index]
            cells[cell_index] = @set temp_cell.the_outer_neighbors = i
        else
            return -1
        end
    end
    return push!(cell_neighbors,cells[cell_index].the_outer_neighbors,UInt32(t))
end

function add_track(cells::Vector{GPUCACell},cell_index::Integer, t::Integer, cell_tracks::CellTracksVector)
    if cells[cell_index].the_tracks == 1
        i = extend!(cell_tracks)
        if i > 1 
            # reset!(cell_tracks[i])
            temp_cell = cells[cell_index]
            cells[cell_index] = @set temp_cell.the_tracks = i 
        else
            return -1
        end
    end
    return push!(cell_tracks,cells[cell_index].the_tracks,UInt16(t))
end
"""
Recursively identifies ntuplets (sequences of hits) in a set of cells. This function traverses through neighboring cells to construct valid ntuplets, ensuring constraints on depth, minimum hits, and memory management.

# Arguments:
- `self`: The current cell being processed.
- `::Val{DEPTH}`: Compile-time constant representing the current depth of recursion. Helps optimize performance by reducing runtime overhead.
- `cells`: An Vector of all cells, each representing a doublet with relevant attributes (e.g., `doublet_id`, `inner_hit_id`, `outer_hit_id`, etc.).
- `cell_tracks`: Data structure tracking associations between cells and tracks (updated when valid ntuplets are found).
- `found_ntuplets`: A histogram or storage structure where valid ntuplets are recorded.
- `apc`: A parameter used for appending or managing data in `found_ntuplets`.
- `quality`: An array storing the quality score of identified ntuplets.
- `temp_ntuplet`: A temporary stack used to construct ntuplets during recursion.
- `min_hits_per_ntuplet`: The minimum number of hits required for a sequence to qualify as a valid ntuplet.
- `start_at_0`: 

# Behavior:
1. **Add Current Cell to Temporary Ntuplet:**
   - Pushes the current cell's `doublet_id` to `temp_ntuplet` for processing.
   - Asserts the maximum length of `temp_ntuplet` to 4 (max hits is 5)
2. **Iterate Through Outer Neighbors:**
   - For each valid neighbor of the current cell, recursively calls `find_ntuplets` to extend the ntuplet.
   - Skips invalid neighbors (e.g., killed by preprocessing or marked by a negative `doublet_id`).
3. **Handle Terminal Nodes:**
   - If the current cell has no valid neighbors (`last == true`), checks if the ntuplet satisfies the minimum hits condition.
   - Constructs an array of hits and records the ntuplet in `found_ntuplets` if valid.
4. **Update Tracks and Quality:**
   - Updates track information for all cells in the ntuplet and assigns a default quality score (`0`).
5. **Backtrack for Recursion:**
   - Removes the current cell from `temp_ntuplet` to prepare for the next recursive path.
   - Asserts the consistency of the temporary ntuplet size.

# Notes:
- This function uses recursion with compile-time constants (`Val{DEPTH}`) for efficiency.
- Memory allocations are optimized using structures like `@MArray` to reduce runtime overhead.
- Assertions (`@assert`) and overflow checks ensure robustness and prevent invalid states.

# Returns:
The function does not explicitly return a value but updates `found_ntuplets`, `cell_tracks`, and `quality` as a side effect of processing.
cells is a vector of GPUCACell, cell_tracks is device_the_cell_tracks, 
""" 
function find_ntuplets(self,::Val{DEPTH},cells,cell_tracks,found_ntuplets,apc,quality,temp_ntuplet,min_hits_per_ntuplet,start_at_0,cell_neighbors) where DEPTH
    push!(temp_ntuplet,self.the_doublet_id)
    @assert length(temp_ntuplet) <= 4
    last = true 
    neighbors = cell_neighbors[:,self.the_outer_neighbors]
    for i ∈ 1:length(neighbors)
        other_cell = neighbors[i]
        if cells[other_cell].the_doublet_id < 0 
            # if other_cell== 65098
            #     print("YES")
            # end
            continue # killed by early_fishbone
        end
        last = false
        # print(cells[other_cell].the_inner_hit_id)
        @assert(cells[other_cell].the_inner_hit_id != self.the_inner_hit_id)
        find_ntuplets(cells[other_cell],Val{DEPTH-1}(),cells,cell_tracks,found_ntuplets,apc,quality,temp_ntuplet,min_hits_per_ntuplet,start_at_0,cell_neighbors)
    end
    if last
        if length(temp_ntuplet) >= min_hits_per_ntuplet - 1 # number of min doublets is min_hits - 1 
           hits = @MArray fill(UInt16(0),6)
           n_h = length(temp_ntuplet)
            for c ∈ temp_ntuplet
                hits[n_h] = cells[c].the_inner_hit_id
                n_h -= 1
            end
            hits[length(temp_ntuplet)+1] = self.the_outer_hit_id
            it = bulk_fill(found_ntuplets,apc,hits,length(temp_ntuplet)+1)
            if it > 0 # no overflow of histogram
                for c ∈ temp_ntuplet
                    add_track(cells,c,it,cell_tracks)
                end
            quality[it] = bad
            end
        end
    end
    pop!(temp_ntuplet)
    @assert length(temp_ntuplet) < 4
end

end