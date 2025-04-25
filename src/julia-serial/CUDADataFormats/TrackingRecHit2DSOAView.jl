module CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h

using ..histogram: HisToContainer
using ..CUDADataFormatsSiPixelClusterInterfaceGPUClusteringConstants: MAX_NUM_CLUSTERS
using ..Geometry_TrackerGeometryBuilder_phase1PixelTopology_h.phase1PixelTopology: AverageGeometry
using ..SOA_h
using ..PixelGPU_h
const Hist = HisToContainer{Int16,128,MAX_NUM_CLUSTERS,8 * sizeof(UInt16),UInt16,10}
export max_hits, TrackingRecHit2DSOAView, average_geometry, cpe_params, CommonParams, DetParams, LayerGeometry, ClusParamsT, n_hits, x_global, y_global, z_global, set_x_global, set_y_global, set_z_global, charge, detector_index, x_local, y_local, cluster_size_x, cluster_size_y, xerr_local, yerr_local, hits_layer_start, r_global, i_phi
export Hist, phi_binner
"""
    Struct representing the 2D Structure of Arrays view of tracking hits.

    ## Fields
    - `m_xl::Vector{Float64}`: Local x-coordinates of hits.
    - `m_yl::Vector{Float64}`: Local y-coordinates of hits.
    - `m_xerr::Vector{Float64}`: Errors in local x-coordinates.
    - `m_yerr::Vector{Float64}`: Errors in local y-coordinates.
    - `m_xg::Vector{Float64}`: Global x-coordinates of hits.
    - `m_yg::Vector{Float64}`: Global y-coordinates of hits.
    - `m_zg::Vector{Float64}`: Global z-coordinates of hits.
    - `m_rg::Vector{Float64}`: Global r-coordinates of hits.
    - `m_iphi::Vector{UInt16}`: Indices of hits in phi.
    - `m_charge::Vector{UInt32}`: Charges of hits.
    - `m_xsize::Vector{UInt16}`: Sizes of hits in x direction.
    - `m_ysize::Vector{UInt16}`: Sizes of hits in y direction.
    - `m_det_ind::Vector{UInt16}`: Detector indices of hits.
    - `m_average_geometry::AverageGeometry`: Average geometry data.
    - `m_cpe_params::ParamsOnGPU`: Parameters for CPE (charge propagation estimation).
    - `m_hits_module_start::Vector{Integer}`: Start indices of modules for hits.
    - `m_hits_layer_start::Vector{Integer}`: Start indices of layers for hits.
    - `m_hist::HisToContainer`: Histogram container for hits.
    - `m_nHits::UInt32`: Number of hits.
"""
mutable struct TrackingRecHit2DSOAView
    m_xl::Vector{Float32}
    m_yl::Vector{Float32}
    m_xerr::Vector{Float32}
    m_yerr::Vector{Float32}
    m_xg::Vector{Float32}
    m_yg::Vector{Float32}
    m_zg::Vector{Float32}
    m_rg::Vector{Float32}
    m_iphi::Vector{Int16}
    m_charge::Vector{UInt32}
    m_xsize::Vector{Int16}
    m_ysize::Vector{Int16}
    m_det_ind::Vector{Int16}
    m_average_geometry::AverageGeometry
    m_cpe_params::ParamsOnGPU
    m_hits_module_start::Vector{Integer}
    m_hits_layer_start::Vector{Integer}
    m_hist::HisToContainer
    m_nHits::UInt32


    function TrackingRecHit2DSOAView()
        empty__float_vector = Vector{Float64}()
        empty_int_vector = Vector{Integer}()

        return new(empty__float_vector,
            empty__float_vector,
            empty__float_vector,
            empty__float_vector,
            empty__float_vector,
            empty__float_vector,
            empty__float_vector,
            empty__float_vector,
            empty_int_vector,
            empty_int_vector,
            empty_int_vector,
            empty_int_vector,
            empty_int_vector,
            AverageGeometry(),
            ParamsOnGPU(),
            Vector{Integer}(),
            Vector{Integer}(),
            HisToContainer{0,0,0,0,UInt32}(), #added dummy values 
            zero(UInt32))
    end



    function TrackingRecHit2DSOAView(
        m_xl::Vector{Float64},
        m_yl::Vector{Float64},
        m_xerr::Vector{Float64},
        m_yerr::Vector{Float64},
        m_xg::Vector{Float64},
        m_yg::Vector{Float64},
        m_zg::Vector{Float64},
        m_rg::Vector{Float64},
        m_iphi::Vector{Int16},
        m_charge::Vector{UInt32},
        m_xsize::Vector{Int16},
        m_ysize::Vector{Int16},
        m_det_ind::Vector{Int16},
        m_average_geometry::AverageGeometry,
        m_cpe_params::ParamsOnGPU,
        m_hits_module_start::Vector{UInt32},
        m_hits_layer_start::Vector{Integer},
        m_hist::HisToContainer,
        m_nHits::UInt32
    )

        return new(
            m_xl,
            m_yl,
            m_xerr,
            m_yerr,
            m_xg,
            m_yg,
            m_zg,
            m_rg,
            m_iphi,
            m_charge,
            m_xsize,
            m_ysize,
            m_det_ind,
            m_average_geometry,
            m_cpe_params,
            m_hits_module_start,
            m_hits_layer_start,
            m_hist,
            m_nHits
        )
    end

end
"""
    max_hits()

    Function to get the maximum number of clusters allowed.

    # Arguments
    - None

    # Returns
    - `UInt32`: The maximum number of clusters (`MAX_NUM_CLUSTERS`).
"""
function max_hits()
    return UInt32(48 * 1024)
end

"""
    n_hits(self::TrackingRecHit2DSOAView)::UInt32

    Function to get the total number of hits stored in a `TrackingRecHit2DSOAView` instance.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView` whose number of hits is to be retrieved.

    # Returns
    - `UInt32`: The total number of hits (`m_nHits`) in the `TrackingRecHit2DSOAView` instance.
"""
@inline function n_hits(self::TrackingRecHit2DSOAView)::UInt32
    return self.m_nHits
end

"""
    x_local(self::TrackingRecHit2DSOAView, i::Int)::Float64

    Function to get the local x-coordinate of a hit at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `Float64`: The local x-coordinate (`m_xl[i]`) of the hit at index `i`.
"""
@inline function x_local(self::TrackingRecHit2DSOAView, i::UInt32)::Float32
    return self.m_xl[i]
end

@inline function x_local(self::TrackingRecHit2DSOAView, i::UInt32, k::Float32)::Float32
    self.m_xl[i] = k
end

"""
    y_local(self::TrackingRecHit2DSOAView, i::Int)::Float64

    Function to get the local y-coordinate of a hit at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `Float64`: The local y-coordinate (`m_yl[i]`) of the hit at index `i`.
"""
@inline function y_local(self::TrackingRecHit2DSOAView, i::UInt32)::Float32
    return self.m_yl[i]
end

@inline function y_local(self::TrackingRecHit2DSOAView, i::UInt32, k::Float32)::Float32
    self.m_yl[i] = k
end

"""
    xerr_local(self::TrackingRecHit2DSOAView, i::Int)::Float64

    Function to get the error in the local x-coordinate of a hit at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `Float64`: The error in the local x-coordinate (`m_xerr[i]`) of the hit at index `i`.
"""
@inline function xerr_local(self::TrackingRecHit2DSOAView, i::UInt32)::Float64
    return self.m_xerr[i]
end

@inline function xerr_local(self::TrackingRecHit2DSOAView, i::UInt32, k::Float32)::Float64
    self.m_xerr[i] = k
end
"""
    yerr_local(self::TrackingRecHit2DSOAView, i::Int)::Float64

    Function to get the error in the local y-coordinate of a hit at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `Float64`: The error in the local y-coordinate (`m_yerr[i]`) of the hit at index `i`.
"""
@inline function yerr_local(self::TrackingRecHit2DSOAView, i::UInt32)
    return self.m_yerr[i]
end

@inline function yerr_local(self::TrackingRecHit2DSOAView, i::UInt32, k::Float32)
    self.m_yerr[i] = k
end

"""
    x_global(self::TrackingRecHit2DSOAView, i::Int)::Float64

    Function to get the global x-coordinate of a hit at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `Float64`: The global x-coordinate (`m_xg[i]`) of the hit at index `i`.
"""
@inline function x_global(self::TrackingRecHit2DSOAView, i)
    return self.m_xg[i]
end

"""
    y_global(self::TrackingRecHit2DSOAView, i::Int)::Float64

    Function to get the global y-coordinate of a hit at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `Float64`: The global y-coordinate (`m_yg[i]`) of the hit at index `i`.
"""
@inline function y_global(self::TrackingRecHit2DSOAView, i)
    return self.m_yg[i]
end

"""
    z_global(self::TrackingRecHit2DSOAView, i::Int)::Float64

    Function to get the global z-coordinate of a hit at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `Float64`: The global z-coordinate (`m_zg[i]`) of the hit at index `i`.
"""
@inline function z_global(self::TrackingRecHit2DSOAView, i)
    return self.m_zg[i]
end

@inline function set_x_global(self::TrackingRecHit2DSOAView, i, val)
    self.m_xg[i] = val
end

@inline function set_y_global(self::TrackingRecHit2DSOAView, i, val)
    self.m_yg[i] = val
end

@inline function set_z_global(self::TrackingRecHit2DSOAView, i, val)
    self.m_zg[i] = val
end



"""
    r_global(self::TrackingRecHit2DSOAView, i::Int)::Float64

    Function to get the global r-coordinate (radius) of a hit at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `Float64`: The global r-coordinate (`m_rg[i]`) of the hit at index `i`.
"""
@inline function r_global(self::TrackingRecHit2DSOAView, i::Integer)
    return self.m_rg[i]
end

@inline function r_global(self::TrackingRecHit2DSOAView, i::UInt32, l::Float32)
    self.m_rg[i] = l
end

"""
    i_phi(self::TrackingRecHit2DSOAView, i::Int)::UInt16

    Function to get the phi index of a hit at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `UInt16`: The phi index (`m_iphi[i]`) of the hit at index `i`.
"""
@inline function i_phi(self::TrackingRecHit2DSOAView, i::Integer)
    return self.m_iphi[i]
end
@inline function i_phi(self::TrackingRecHit2DSOAView)
    return self.m_iphi
end

@inline function i_phi(self::TrackingRecHit2DSOAView, i::Integer, k::Int16)
    self.m_iphi[i] = k
end

"""
    charge(self::TrackingRecHit2DSOAView, i::Int)::UInt32

    Function to get the charge of a hit at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `UInt32`: The charge (`m_charge[i]`) of the hit at index `i`.
"""
@inline function charge(self::TrackingRecHit2DSOAView, i::Int32)::UInt32
    return self.m_charge[i]
end

@inline function charge(self::TrackingRecHit2DSOAView, i::UInt32, k::Int32)::UInt32
    self.m_charge[i] = k
end

"""
    cluster_size_x(self::TrackingRecHit2DSOAView, i::Int)::UInt16

    Function to get the size of a hit cluster in the x direction at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `UInt16`: The size of the hit cluster in the x direction (`m_xsize[i]`) at index `i`.
"""
@inline function cluster_size_x(self::TrackingRecHit2DSOAView, i::UInt32)::Int16
    return self.m_xsize[i]
end
@inline function cluster_size_x(self::TrackingRecHit2DSOAView, i::UInt32, k::Int16)::Int16
    self.m_xsize[i] = k
end

"""
    cluster_size_y(self::TrackingRecHit2DSOAView, i::Int)::UInt16

    Function to get the size of a hit cluster in the y direction at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `UInt16`: The size of the hit cluster in the y direction (`m_ysize[i]`) at index `i`.
"""
@inline function cluster_size_y(self::TrackingRecHit2DSOAView, i::Integer)
    return self.m_ysize[i]
end

@inline function cluster_size_y(self::TrackingRecHit2DSOAView, i::UInt32, k::Int16)::Int16
    self.m_ysize[i] = k
end
"""
    detector_index(self::TrackingRecHit2DSOAView, i::Int)::UInt16

    Function to get the detector index of a hit at index `i`.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.
    - `i::Int`: The index of the hit.

    # Returns
    - `UInt16`: The detector index (`m_det_ind[i]`) of the hit at index `i`.
"""
@inline function detector_index(self::TrackingRecHit2DSOAView, i::Integer)
    return self.m_det_ind[i]
end

@inline function detector_index(self::TrackingRecHit2DSOAView, i::UInt32, k::UInt32)::UInt16
    self.m_det_ind[i] = k
end

"""
    cpe_params(self::TrackingRecHit2DSOAView)::ParamsOnGPU

    Function to get the CPE (Cluster Parameter Estimation) parameters.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.

    # Returns
    - `ParamsOnGPU`: The CPE parameters (`m_cpe_params`) for GPU usage.
"""
@inline function cpe_params(self::TrackingRecHit2DSOAView)::ParamsOnGPU
    return self.m_cpe_params
end

"""
    hits_module_start(self::TrackingRecHit2DSOAView)::UInt32

    Function to get the start indices of hits in each module.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.

    # Returns
    - `UInt32`: The start indices of hits in each module (`m_hits_module_start`).
"""
@inline function hits_module_start(self::TrackingRecHit2DSOAView)
    return self.m_hits_module_start
end

"""
    hits_layer_start(self::TrackingRecHit2DSOAView)::UInt32

    Function to get the start indices of hits in each layer.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.

    # Returns
    - `UInt32`: The start indices of hits in each layer (`m_hits_layer_start`).
"""
@inline function hits_layer_start(self::TrackingRecHit2DSOAView)
    return self.m_hits_layer_start
end

"""
    phi_binner(self::TrackingRecHit2DSOAView)::HisToContainer

    Function to get the histogram binning information.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.

    # Returns
    - `HisToContainer`: The histogram binning information (`m_hist`).
"""
@inline function phi_binner(self::TrackingRecHit2DSOAView)::HisToContainer
    return self.m_hist
end

"""
    average_geometry(self::TrackingRecHit2DSOAView)::AverageGeometry

    Function to get the average geometry data.

    # Arguments
    - `self::TrackingRecHit2DSOAView`: The instance of `TrackingRecHit2DSOAView`.

    # Returns
    - `AverageGeometry`: The average geometry data (`m_average_geometry`).
"""
@inline function average_geometry(self::TrackingRecHit2DSOAView)
    return self.m_average_geometry
end

end
# # test case
# using .CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h:TrackingRecHit2DSOAView,ParamsOnGPU,max_hits,n_hits,x_local,y_local,xerr_local,yerr_local,x_global,y_global,z_global,r_global,i_phi,charge,cluster_size_x,cluster_size_y,detector_index,cpe_params,hits_module_start,hits_layer_start,phi_binner,average_geometry
# using .Geometry_TrackerGeometryBuilder_phase1PixelTopology_h.phase1PixelTopology:AverageGeometry
# using .histogram:HisToContainer
# using .CUDADataFormatsSiPixelClusterInterfaceGPUClusteringConstants:MAX_NUM_CLUSTERS

# avg_geom = AverageGeometry()
# params_gpu = ParamsOnGPU()
# 

# number_Hits = 100
# tracking_view = TrackingRecHit2DSOAView(
#     rand(Float64, number_Hits),  # m_xl
#     rand(Float64, number_Hits),  # m_yl
#     rand(Float64, number_Hits),  # m_xerr
#     rand(Float64, number_Hits),  # m_yerr
#     rand(Float64, number_Hits),  # m_xg
#     rand(Float64, number_Hits),  # m_yg
#     rand(Float64, number_Hits),  # m_zg
#     rand(Float64, number_Hits),  # m_rg
#     rand(UInt16, number_Hits),   # m_iphi
#     rand(UInt32, number_Hits),   # m_charge
#     rand(UInt16, number_Hits),   # m_xsize
#     rand(UInt16, number_Hits),   # m_ysize
#     rand(UInt16, number_Hits),   # m_det_ind
#     avg_geom,              # m_average_geometry
#     params_gpu,            # m_cpe_params
#     Vector{Integer}(rand(Int64, number_Hits)),   # m_hits_module_start
#     Vector{Integer}(rand(Int64, number_Hits)),   # m_hits_layer_start
#     Hist,                  # m_hist
#     UInt32(number_Hits)    # m_nHits
# )

# println("Max Hits: ", max_hits())
# println("Number of Hits: ", n_hits(tracking_view))

# for i in 1:number_Hits
#     println("Hit $i: ")
#     println("  xLocal: ", x_local(tracking_view, i))
#     println("  yLocal: ", y_local(tracking_view, i))
#     println("  xerrLocal: ", xerr_local(tracking_view, i))
#     println("  yerrLocal: ", yerr_local(tracking_view, i))
#     println("  xGlobal: ", x_global(tracking_view, i))
#     println("  yGlobal: ", y_global(tracking_view, i))
#     println("  zGlobal: ", z_global(tracking_view, i))
#     println("  rGlobal: ", r_global(tracking_view, i))
#     println("  iphi: ", i_phi(tracking_view, i))
#     println("  charge: ", charge(tracking_view, i))
#     println("  clusterSizeX: ", cluster_size_x(tracking_view, i))
#     println("  clusterSizeY: ", cluster_size_y(tracking_view, i))
#     println("  detectorIndex: ", detector_index(tracking_view, i))
# end

# println("CPE Params: ", cpe_params(tracking_view))
# println("error here 1")
# println("Hits Module Start: ", hits_module_start(tracking_view))
# println("Hits Layer Start: ", hits_layer_start(tracking_view))
# println("Phi Binner: ", phi_binner(tracking_view))
# println("Average Geometry: ", average_geometry(tracking_view))