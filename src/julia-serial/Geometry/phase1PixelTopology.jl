module Geometry_TrackerGeometryBuilder_phase1PixelTopology_h
export number_of_module_in_barrel, AverageGeometry, find_max_module_stride, local_x, local_y, is_big_pix_y, is_big_pix_x, number_of_ladders_in_barrel
module phase1PixelTopology
export AverageGeometry, number_of_module_in_barrel, number_of_layers, layer_index_size, find_max_module_stride, local_x, local_y, is_big_pix_y, is_big_pix_x, number_of_ladders_in_barrel, last_row_in_module, last_col_in_module, x_offset, y_offset
    # Constants defining the dimensions of ROCs and modules
    const num_rows_in_ROC = 80
    const num_cols_in_ROC = 52
    const last_row_in_roc = num_rows_in_ROC - 1
    const last_col_in_roc = num_cols_in_ROC - 1

    const num_rows_in_module = 2 * num_rows_in_ROC
    const num_cols_in_module = 8 * num_cols_in_ROC
    const last_row_in_module = num_rows_in_module - 1
    const last_col_in_module = num_cols_in_module - 1

    const x_offset::Int16 = Int16(-81)
    const y_offset::Int16 = Int16(-54 * 4)

    const num_pixs_in_module = num_rows_in_module * num_cols_in_module

    const number_of_modules = 1856
    const number_of_layers = UInt32(10)

    # Starting indices for each layer
    const layer_start = [
        0,
        96,
        320,
        672,   # barrel
        1184,
        1296,
        1408,  # positive endcap
        1520,
        1632,
        1744,  # negative endcap
        number_of_modules
    ]

    # Names of the layers
    const layer_name = [
        "BL1",
        "BL2",
        "BL3",
        "BL4",   # barrel
        "E+1",
        "E+2",
        "E+3",  # positive endcap
        "E-1",
        "E-2",
        "E-3"   # negative endcap
    ]
    #  2, 3, 2, 4, 2, 7, 5, 6, 8, 9,
    const number_of_module_in_barrel = 1184
    const number_of_ladders_in_barrel = number_of_module_in_barrel ÷ 8

    """
    Function to find the maximum module stride that divides all layer start indices.

    ## Returns
    - `Int`: The maximum module stride.
    """
    function find_max_module_stride()
            n = 2
     all_divisible = true
            while all_divisible

                for i in 2:11
                    if layer_start[i] % n != 0
                        all_divisible = false
                        break
                    end
                end
                if !all_divisible
                    break
                end
                n *= 2
            end
            return n ÷ 2
        end



    const max_module_stride = find_max_module_stride()

    """
    Function to find the layer index for a given detector ID.

    ## Arguments
    - `det_id::UInt32`: The detector ID.

    ## Returns
    - `Int`: The layer index.
    """
    function find_layer(det_id::UInt32)
        for i in 0:11
            if det_id < layer_start[i + 1]
                return i
            end
        end
        return 11
    end

    """
    Function to find the layer index for a given compact detector ID.

    ## Arguments
    - `det_id::UInt32`: The compact detector ID.

    ## Returns
    - `Int`: The layer index.
    """
    function find_layer_from_compact(det_id)
        det_id *= max_module_stride
        for i in 0:11
            if det_id < layer_start[i + 1]
                return i
            end
        end
        return 11
    end

    const layer_index_size::UInt32 = number_of_modules ÷ max_module_stride

    # FIXME can do broadcasting
    const layer = find_layer_from_compact.(0:layer_index_size-1)

    """
    Function to validate the layer index.

    ## Returns
    - `Bool`: `true` if the layer index is valid, `false` otherwise.
    """
    function validate_layer_index()::Bool
        res = true
        for i in 0:number_of_modules-1
            j = i ÷ max_module_stride + 1
            res = layer[j] < 10
            res = i >= layer_start[layer[j]]
            res = i < layer_start[layer[j] + 1]
        end
        return res
    end

    @assert validate_layer_index() "Layer from detIndex algorithm is buggy"

    """
    Function to divide a number by 52 using bit shifts.

    ## Arguments
    - `n::UInt16`: The number to divide.

    ## Returns
    - `UInt16`: The quotient after division by 52.
    """
    function divu52(n::UInt16)
        n = n >> 2
        q = (n >> 1) + (n >> 4)
        q = q + (q >> 4) + (q >> 5)
        q = q >> 3
        r = n - q * 13
        return q + ((r + 3) >> 4)
    end

    """
    Function to check if a pixel is at the edge in the x-direction.

    ## Arguments
    - `px::UInt16`: The x-coordinate of the pixel.

    ## Returns
    - `Bool`: `true` if the pixel is at the edge, `false` otherwise.
    """
    @inline function is_edge_x(px)
        return (px == 0) | (px == last_row_in_module)
    end

    """
    Function to check if a pixel is at the edge in the y-direction.

    ## Arguments
    - `py::UInt16`: The y-coordinate of the pixel.

    ## Returns
    - `Bool`: `true` if the pixel is at the edge, `false` otherwise.
    """
    @inline function is_edge_y(py)
        return (py == 0) | (py == last_col_in_module)
    end

    """
    Function to convert a pixel x-coordinate to ROC x-coordinate.

    ## Arguments
    - `px::UInt16`: The x-coordinate of the pixel.

    ## Returns
    - `UInt16`: The ROC x-coordinate.
    """
    @inline function to_ROC_x(px)
        return (px < num_rows_in_ROC) ? px : px - num_rows_in_ROC
    end

    """
    Function to convert a pixel y-coordinate to ROC y-coordinate.

    ## Arguments
    - `py::UInt16`: The y-coordinate of the pixel.

    ## Returns
    - `UInt16`: The ROC y-coordinate.
    """
    @inline function to_ROC_y(py)
        roc = py ÷ 52
        return py - 52 * roc
    end

    """
    Function to check if a pixel is a big pixel in the y-direction.

    ## Arguments
    - `py::UInt16`: The y-coordinate of the pixel.

    ## Returns
    - `Bool`: `true` if the pixel is a big pixel, `false` otherwise.
    """
    @inline function is_big_pix_y(py)
        ly = to_ROC_y(py)
        return (ly == 0) | (ly == last_col_in_roc)
    end

    @inline function is_big_pix_x(px)
        return (px == 79) | (px == 80)
    end
    """
    Function to convert a local x-coordinate to a global x-coordinate.

    ## Arguments
    - `px::UInt16`: The local x-coordinate.

    ## Returns
    - `UInt16`: The global x-coordinate.
    """
    @inline function local_x(px::UInt16)::UInt16
        shift = 0
        if px > last_row_in_roc
            shift += 1
        end
        if px > num_rows_in_ROC
            shift += 1
        end
        return px + shift
    end

    """
    Function to convert a local y-coordinate to a global y-coordinate.

    ## Arguments
    - `py::UInt16`: The local y-coordinate.

    ## Returns
    - `UInt16`: The global y-coordinate.
    """
    @inline function local_y(py::UInt16)::UInt16
        roc = py ÷ 52
        shift = 2 * roc
        y_in_ROC = py - 52 * roc
        if y_in_ROC > 0
            shift += 1
        end
        return py + shift
    end

    # FIXME why we need static arrays?
    using StaticArrays

    """
    Struct representing the average geometry of the detector.

    ## Fields
    - `number_of_ladders_in_barrel::Int`: Number of ladders in the barrel.
    - `ladderZ::SVector{number_of_ladders_in_barrel, Float32}`: Z-coordinates of the ladders.
    - `ladderX::SVector{number_of_ladders_in_barrel, Float32}`: X-coordinates of the ladders.
    - `ladderY::SVector{number_of_ladders_in_barrel, Float32}`: Y-coordinates of the ladders.
    - `ladderR::SVector{number_of_ladders_in_barrel, Float32}`: Radii of the ladders.
    - `ladderMinZ::SVector{number_of_ladders_in_barrel, Float32}`: Minimum Z-coordinates of the ladders.
    - `ladderMaxZ::SVector{number_of_ladders_in_barrel, Float32}`: Maximum Z-coordinates of the ladders.
    - `endCapZ::NTuple{2, Float32}`: Z-coordinates for the positive and negative endcap Layer1.
    """
    struct AverageGeometry
        number_of_ladders_in_barrel::UInt32
        ladderZ::Vector{Float32}
        ladderX::Vector{Float32}
        ladderY::Vector{Float32}
        ladderR::Vector{Float32}
        ladderMinZ::Vector{Float32}
        ladderMaxZ::Vector{Float32}
        endCapZ::Vector{Float32}  # just for pos and neg Layer1


        function AverageGeometry(number_of_ladders_in_barrel,ladderZ,ladderX,ladderY,ladderR,ladderMinZ,ladderMaxZ,endCapZ)
            return new(number_of_ladders_in_barrel,ladderZ,ladderX,ladderY,ladderR,ladderMinZ,ladderMaxZ,endCapZ)
        end

        function AverageGeometry()
            number_of_ladders_in_barrel = 0
            ladderZ = zeros(Float64, 148)
            ladderX = zeros(Float64, 148)
            ladderY = zeros(Float64, 148)
            ladderR = zeros(Float64, 148)
            ladderMinZ = zeros(Float64, 148)
            ladderMaxZ = zeros(Float64, 148)
            endCapZ = zeros(Float64, 2)
            new(number_of_ladders_in_barrel, ladderZ, ladderX, ladderY, ladderR, ladderMinZ, ladderMaxZ, endCapZ)
        end
    end


end

end
