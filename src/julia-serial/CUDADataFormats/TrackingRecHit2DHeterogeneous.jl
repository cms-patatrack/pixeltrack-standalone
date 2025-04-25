module CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

export TrackingRecHit2DHeterogeneous, hist_view, ParamsOnGPU, hits_layer_start, phi_binner, iphi, Hist
export n_hits

# Import necessary types and functions from other modules
using ..CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h: TrackingRecHit2DSOAView, ParamsOnGPU
using ..histogram: HisToContainer
using ..Geometry_TrackerGeometryBuilder_phase1PixelTopology_h.phase1PixelTopology: AverageGeometry
using ..CUDADataFormatsSiPixelClusterInterfaceGPUClusteringConstants
"""
    Struct representing the heterogeneous data for 2D tracking hits.

    ## Fields
    - `n16::UInt32`: Number of 16-bit data elements per hit.
    - `n32::UInt32`: Number of 32-bit data elements per hit.
    - `m_store16::Union{Nothing, Vector{Vector{UInt16}}}`: Optional storage for 16-bit data, initialized as Nothing or a vector of vectors.
    - `m_store32::Union{Nothing, Vector{Vector{Float64}}}`: Optional storage for 32-bit data, initialized as Nothing or a vector of vectors.
    - `m_HistStore::Union{Nothing, Vector{HisToContainer}}`: Optional storage for histogram data, initialized as Nothing or a vector of histogram containers.
    - `m_AverageGeometryStore::Union{Nothing, Vector{AverageGeometry}}`: Optional storage for average geometry data, initialized as Nothing or a vector of AverageGeometry.
    - `m_view::Union{Nothing, Vector{TrackingRecHit2DSOAView}}`: Optional storage for 2D structure of arrays view, initialized as Nothing or a vector of TrackingRecHit2DSOAView.
    - `m_nHits::UInt32`: Number of hits.
    - `m_hitsModuleStart::Vector{UInt32}`: Start indices for hits in modules.
    - `m_hist::Union{Nothing, HisToContainer}`: Optional histogram container for hits, initialized as Nothing or a HisToContainer.
    - `m_hitsLayerStart::Union{Nothing, Vector{UInt32}}`: Optional start indices for hits in layers, initialized as Nothing or a vector of UInt32.
    - `m_iphi::Union{Nothing, Vector{UInt16}}`: Optional vector of indices in phi, initialized as Nothing or a vector of UInt16.
"""
    const Hist = HisToContainer{Int16, 128, MAX_NUM_CLUSTERS, 8 * sizeof(UInt16), UInt16, 10}
mutable struct TrackingRecHit2DHeterogeneous
    n16::UInt32
    n32::UInt32
    m_store16::Union{Nothing, Vector{Vector{Int16}}}
    m_store32::Union{Nothing, Vector{Vector{Float64}}}
    m_HistStore::HisToContainer
    m_AverageGeometryStore::AverageGeometry
    m_view::TrackingRecHit2DSOAView
    m_nHits::UInt32
    m_hitsModuleStart::Vector{UInt32}
    m_hist::Hist
    m_hitsLayerStart::Union{Nothing, Vector{UInt32}}
    m_iphi::Union{Nothing, Vector{Int16}}

    """
        Constructor for TrackingRecHit2DHeterogeneous.

        ## Arguments
        - `nHits::Integer`: Number of hits.
        - `cpe_params::ParamsOnGPU`: Parameters for charge propagation estimation (CPE).
        - `hitsModuleStart::Vector{Integer}`: Start indices of modules for hits.
        - `Hist::HisToContainer`: Histogram container for hits.

        ## Fields Initialized
        - `n16`: Set to 4.
        - `n32`: Set to 9.
        - `m_store16`: Initialized as a vector of vectors of UInt16.
        - `m_store32`: Initialized as a vector of vectors of Float64.
        - `m_HistStore`: Set to `Hist`.
        - `m_AverageGeometryStore`: Initialized with a vector containing one `AverageGeometry` object.
        - `m_view`: Initialized with one `TrackingRecHit2DSOAView` object.
    """
    function TrackingRecHit2DHeterogeneous(nHits::Integer, cpe_params::ParamsOnGPU, hitsModuleStart::Vector{UInt32})
        n16 = 4
        n32 = 9
    
        if nHits == 0
            return new(n16, n32, Vector{Vector{Int16}}()
            , Vector{Vector{Float64}}(), HisToContainer{0,0,0,0,UInt32}(), AverageGeometry(), TrackingRecHit2DSOAView(), nHits, hitsModuleStart, Hist(), Vector{UInt32}(), Vector{Int16}()) #added dummy values for HisToContainer
        end
    
        # Initialize storage vectors
        m_store16 = [Vector{Int16}(undef, nHits) for _ in 1:n16]
        m_store32 = [Vector{Float64}(undef, nHits) for _ in 1:n32]
        m_store32_UInt32 = [Vector{UInt32}(undef, nHits) for _ in 1:n32]
        append!(m_store32, [Vector{Float64}(undef, 11)])
        append!(m_store32_UInt32, [Vector{UInt32}(undef, 11)])

        # Initialize AverageGeometry and Histogram store
        m_AverageGeometryStore = AverageGeometry()
        m_HistStore = Hist()
    
        # Define local functions to access storage
        function get16(i)
            return  m_store16[i + 1]
        end
        function get32(i) 
            return m_store32[i + 1] 
        end
        function get32_uint(i)
            return m_store32_UInt32[i + 1] 
        end
    
        # Initialize hits_layer_start and m_iphi
        hits_layer_start = Vector{Integer}(get32_uint(n32))
        m_iphi =  get16(0)
    
        # Create and initialize the TrackingRecHit2DSOAView object
        m_view = TrackingRecHit2DSOAView(
            get32(0),
            get32(1),
            get32(2),
            get32(3),
            get32(4),
            get32(5),
            get32(6),
            get32(7),
            m_iphi,
            get32_uint(8),
            get16(2),
            get16(3),
            get16(1),
            m_AverageGeometryStore,
            cpe_params,
            hitsModuleStart,
            hits_layer_start,
            m_HistStore,
            UInt32(nHits)
        )    
        

        # Return a new instance of TrackingRecHit2DHeterogeneous
        return new(n16, n32, m_store16, m_store32, m_HistStore, m_AverageGeometryStore, m_view, nHits, hitsModuleStart, m_HistStore, hits_layer_start, m_iphi)
    end
end

"""
    Accessor function for retrieving the view from TrackingRecHit2DHeterogeneous.

    ## Arguments
    - `hit::TrackingRecHit2DHeterogeneous`: The tracking hit object.

    ## Returns
    - `hit.m_view`: The stored view of type `Vector{TrackingRecHit2DSOAView}`.
"""
hist_view(hit::TrackingRecHit2DHeterogeneous) = hit.m_view

"""
    Accessor function for retrieving the number of hits.

    ## Arguments
    - `hit::TrackingRecHit2DHeterogeneous`: The tracking hit object.

    ## Returns
    - `hit.m_nHits`: The number of hits as `UInt32`.
"""
n_hits(hit::TrackingRecHit2DHeterogeneous) = hit.m_nHits

"""
    Accessor function for retrieving the start indices of hits in modules.

    ## Arguments
    - `hit::TrackingRecHit2DHeterogeneous`: The tracking hit object.

    ## Returns
    - `hit.m_hitsModuleStart`: The vector of start indices for hits in modules.
"""
hits_module_start(hit::TrackingRecHit2DHeterogeneous) = hit.m_hitsModuleStart

"""
    Accessor function for retrieving the start indices of hits in layers.

    ## Arguments
    - `hit::TrackingRecHit2DHeterogeneous`: The tracking hit object.

    ## Returns
    - `hit.m_hitsLayerStart`: The vector of start indices for hits in layers.
"""
hits_layer_start(hit::TrackingRecHit2DHeterogeneous) = hit.m_hitsLayerStart

"""
    Accessor function for retrieving the histogram container.

    ## Arguments
    - `hit::TrackingRecHit2DHeterogeneous`: The tracking hit object.

    ## Returns
    - `hit.m_hist`: The histogram container (`HisToContainer`).
"""
phi_binner(hit::TrackingRecHit2DHeterogeneous) = hit.m_hist

"""
    Accessor function for retrieving the phi indices.

    ## Arguments
    - `hit::TrackingRecHit2DHeterogeneous`: The tracking hit object.

    ## Returns
    - `hit.m_iphi`: The vector of phi indices.
"""
iphi(hit::TrackingRecHit2DHeterogeneous) = hit.m_iphi

function test_tracking_rec_hit()
    hitsModuleStart = Integer[1, 2, 3] 
    cpe_params = ParamsOnGPU()
    Hist = HisToContainer{Int16, 128, MAX_NUM_CLUSTERS, 8 * sizeof(UInt16), UInt16, 10}()
    hits = TrackingRecHit2DHeterogeneous(3, cpe_params, hitsModuleStart, Hist)
    
    println("Number of hits: ", n_hits(hits))
    println("Hits module start: ", hits_module_start(hits))
    
    if !isempty(view(hits))
        println("First view element: ", view(hits)[1])
    end
end
# test_tracking_rec_hit()

end
