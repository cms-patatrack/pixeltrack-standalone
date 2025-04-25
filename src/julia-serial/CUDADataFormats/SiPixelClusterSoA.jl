module CUDADataFormatsSiPixelClusterInterfaceSiPixelClustersSoA
export SiPixelClustersSoA, nClusters, clus_module_start, clusterView, DeviceConstView, module_start, setNClusters!, module_id, clus_in_module
    """
    Struct to represent a constant view of the device data.
    """
    struct DeviceConstView
        module_start::Vector{UInt32}       # Pointer to module start data
        clus_in_module::Vector{UInt32}      # Pointer to clusters in module data
        module_id::Vector{UInt32}          # Pointer to module ID data
        clus_module_start::Vector{UInt32}   # Pointer to clusters module start data
    end

    """
    Function to get the start of a module from the view.
    Inputs:
    - view::DeviceConstView: The device view containing the data pointers.
    - i::Int: Index of the module.
    Outputs:
    - UInt32: The start index of the specified module.
    """
    @inline function module_start(view::DeviceConstView, i::Int)::UInt32
        return view.module_start[i]
    end

    """
    Function to get the number of clusters in a module from the view.
    Inputs:
    - view::DeviceConstView: The device view containing the data pointers.
    - i::Int: Index of the module.
    Outputs:
    - UInt32: The number of clusters in the specified module.
    """
    @inline function clus_in_module(view::DeviceConstView, i::UInt32)::UInt32
        return view.clus_in_module[i]
    end

    """
    Function to get the module id from the view.
    Inputs:
    - view::DeviceConstView: The device view containing the data pointers.
    - i::Int: Index of the module.
    Outputs:
    - UInt32: The module ID of the specified module.
    """
    @inline function module_id(view::DeviceConstView, i::Int)::UInt32
        return view.module_id[i]
    end

    """
    Function to get the start of a cluster module from the view.
    Inputs:
    - view::DeviceConstView: The device view containing the data pointers.
    - i::Int: Index of the cluster module.
    Outputs:
    - UInt32: The start index of the specified cluster module.
    """
    @inline function clus_module_start(view::DeviceConstView, i::UInt32)::UInt32
        return view.clus_module_start[i]
    end

    """
    Struct to hold the cluster data in a CUDA-compatible structure.
    """
    mutable struct SiPixelClustersSoA
        module_start_d::Vector{UInt32}       # Pointer to the module start data
        clus_in_module_d::Vector{UInt32}      # Pointer to the number of clusters in each module
        module_id_d::Vector{UInt32}          # Pointer to the module ID data
        clus_module_start_d::Vector{UInt32}   # Pointer to the start index of clusters in each module
        
        view_d::DeviceConstView             # Device view containing the data pointers
        nClusters_h::UInt32                 # Number of clusters (stored on host)

    end

    """
    Constructor for SiPixelClustersSoA.
    Inputs:
    - maxClusters::Int: Maximum number of clusters.
    Outputs:
    - SiPixelClustersSoA: A new instance with allocated data arrays and initialized device view.
    """
    function SiPixelClustersSoA(maxClusters)
        # Allocate memory for the data arrays.
        module_start_d = Vector{UInt32}(undef,maxClusters + 1)
        clus_in_module_d = Vector{UInt32}(undef,maxClusters)
        module_id_d = Vector{UInt32}(undef,maxClusters)
        clus_module_start_d = Vector{UInt32}(undef,maxClusters + 1)

        view_d = DeviceConstView(module_start_d, clus_in_module_d, module_id_d, clus_module_start_d)
    
        return SiPixelClustersSoA(module_start_d, clus_in_module_d, module_id_d, clus_module_start_d, view_d, 0)
    end

    """
    Function to get the device view pointer from a SiPixelClustersSoA instance.
    Inputs:
    - self::SiPixelClustersSoA: The instance of SiPixelClustersSoA.
    Outputs:
    - DeviceConstView: The pointer to the device view.
    """
    function clusterView(self::SiPixelClustersSoA)::DeviceConstView
        return self.view_d  
    end

    """
    Function to get the module start pointer from a SiPixelClustersSoA instance.
    Inputs:
    - self::SiPixelClustersSoA: The instance of SiPixelClustersSoA.
    Outputs:
    - pointer(UInt32): The pointer to the module start data.
    """
    function module_start(self::SiPixelClustersSoA)::Vector{UInt32}
        return self.module_start_d
    end

    """
    Function to get the clusters in module pointer from a SiPixelClustersSoA instance.
    Inputs:
    - self::SiPixelClustersSoA: The instance of SiPixelClustersSoA.
    Outputs:
    - pointer(UInt32): The pointer to the clusters in module data.
    """
    function clus_in_module(self::SiPixelClustersSoA)::Vector{UInt32}
        return self.clus_in_module_d
    end

    """
    Function to get the module id pointer from a SiPixelClustersSoA instance.
    Inputs:
    - self::SiPixelClustersSoA: The instance of SiPixelClustersSoA.
    Outputs:
    - pointer(UInt32): The pointer to the module id data.
    """
    function module_id(self::SiPixelClustersSoA)::Vector{UInt32}
        return self.module_id_d
    end

    """
    Function to get the clusters module start pointer from a SiPixelClustersSoA instance.
    Inputs:
    - self::SiPixelClustersSoA: The instance of SiPixelClustersSoA.
    Outputs:
    - pointer(UInt32): The pointer to the clusters module start data.
    """
    function clus_module_start(self::SiPixelClustersSoA)::Vector{UInt32}
        return self.clus_module_start_d
    end

    """
    Function to set the number of clusters in a SiPixelClustersSoA instance.
    Inputs:
    - self::SiPixelClustersSoA: The instance of SiPixelClustersSoA.
    - nClusters::UInt32: The number of clusters to set.
    Outputs:
    - None
    """
    function setNClusters!(self::SiPixelClustersSoA, nClusters::UInt32)
        self.nClusters_h = nClusters  # Set the number of clusters.
    end

    """
    Function to get the number of clusters from a SiPixelClustersSoA instance.
    Inputs:
    - self::SiPixelClustersSoA: The instance of SiPixelClustersSoA.
    Outputs:
    - UInt32: The number of clusters.
    """
    function nClusters(self::SiPixelClustersSoA)::UInt32
        return self.nClusters_h 
    end

end # module