module PixelGPU_h

using ..Geometry_TrackerGeometryBuilder_phase1PixelTopology_h.phase1PixelTopology: AverageGeometry, local_x, local_y, is_big_pix_y, is_big_pix_x,  last_row_in_module, last_col_in_module, x_offset, y_offset
using ..SOA_h
using ..CUDADataFormatsSiPixelClusterInterfaceGPUClusteringConstants.pixelGPUConstants
using ..Printf

export CommonParams, DetParams, LayerGeometry, ParamsOnGPU, ClusParamsT, averageGeometry, MaxHitsInIter, commonParams, detParams, position_corr, errorFromDB, layerGeometry
"""
 Struct for common detector parameters including thickness, pitch, and default values.
    
    - `theThicknessB::Float32`: Thickness for barrel detectors.
    - `theThicknessE::Float32`: Thickness for endcap detectors.
    - `thePitchX::Float32`: Pitch in the x-direction.
    - `thePitchY::Float32`: Pitch in the y-direction.
"""
struct CommonParams
    theThicknessB::Float32
    theThicknessE::Float32
    thePitchX::Float32
    thePitchY::Float32
    function CommonParams()
        new(0,0,0,0)
    end

    function CommonParams(a, b, c, d)
        new(a, b, c, d)
    end
end

"""
### DetParams
    Struct for detector-specific parameters, including error values and positional offsets.

    - `isBarrel::Bool`: Indicates if the detector is a barrel type.
    - `isPosZ::Bool`: Indicates the z-position.
    - `layer::UInt16`: Layer number.
    - `index::UInt16`: Index within the layer.
    - `rawId::UInt32`: Raw identifier for the detector.
    - `shiftX::Float32`: X-offset.
    - `shiftY::Float32`: Y-offset.
    - `chargeWidthX::Float32`: Charge width in x-direction.
    - `chargeWidthY::Float32`: Charge width in y-direction.
    - `x0::Float32`: X-coordinate offset for position calculation.
    - `y0::Float32`: Y-coordinate offset for position calculation.
    - `z0::Float32`: Z-coordinate offset for position calculation.
    - `sx::NTuple{3, Float32}`: Error values in x-direction.
    - `sy::NTuple{3, Float32}`: Error values in y-direction.
    - `frame::SOAFrame{Float32}`: Frame data structure.


"""

struct DetParams
    isBarrel::Bool
    isPosZ::Bool
    layer::UInt16
    index::UInt16
    rawId::UInt32

    shiftX::Float32
    shiftY::Float32
    chargeWidthX::Float32
    chargeWidthY::Float32

    x0::Float32
    y0::Float32
    z0::Float32

    sx::NTuple{3, Float32}
    sy::NTuple{3, Float32}

    frame::SOAFrame{Float32}

    function DetParams(isBarrel,isPosZ,layer,index,rawId,shiftX,shiftY,chargeWidthX,chargeWidthY,x0,y0,z0,sx,sy,frame)
        return new(isBarrel,isPosZ,layer,index,rawId,shiftX,shiftY,chargeWidthX,chargeWidthY,x0,y0,z0,sx,sy,frame)
    end


    function DetParams()
        new(
            false,          # isBarrel
            false,          # is    PosZ
            0x0000,         # layer
            0x0000,         # index
            0x00000000,     # rawId
            0.0f0,          # shiftX
            0.0f0,          # shiftY
            0.0f0,          # chargeWidthX
            0.0f0,          # chargeWidthY
            0.0f0,          # x0
            0.0f0,          # y0
            0.0f0,          # z0
            (0.0f0, 0.0f0, 0.0f0),  # sx
            (0.0f0, 0.0f0, 0.0f0),  # sy
            SOAFrame{Float32}()  # frame
        )
    end
end

# const AverageGeometry = Phase1PixelTopology.AverageGeometry
""" 
    ### LayerGeometry
    Struct for layer geometry including start indices and layer numbers.

    - `layerStart::Vector{UInt32}`: Vector of start indices for layers.
    - `layer::Vector{UInt8}`: Vector of layer numbers.
    
"""

struct LayerGeometry
    layerStart::Vector{UInt32}
    layer::Vector{UInt8}

    function LayerGeometry(a, b)
        new(a, b)
    end
    function LayerGeometry()
        new(
            Vector{UInt32}(),  # Empty vector for layerStart
            Vector{UInt8}()    # Empty vector for layer
        )
    end
end
"""
 ### ParamsOnGPU
    Struct for storing parameters needed on GPU, including common parameters, detector parameters, 
    layer geometry, and average geometry.

    - `m_commonParams::CommonParams`: Common parameters for the detector.
    - `m_detParams::Vector{DetParams}`: Vector of detector parameters.
    - `m_layerGeometry::LayerGeometry`: Layer geometry.
    - `m_averageGeometry::AverageGeometry`: Average geometry data.

"""
struct ParamsOnGPU
    m_commonParams::CommonParams
    m_detParams::Vector{DetParams}
    m_layerGeometry::LayerGeometry
    m_averageGeometry::AverageGeometry

    function ParamsOnGPU(
        commonParams::CommonParams,
        detParams::Vector{DetParams},
        layerGeometry::LayerGeometry,
        averageGeometry::AverageGeometry
    )
        new(commonParams, detParams, layerGeometry, averageGeometry)
    end
    function ParamsOnGPU()
        temp_vec = [DetParams()]
        new(CommonParams(),temp_vec,LayerGeometry(),AverageGeometry())
    end
end


function commonParams(params::ParamsOnGPU)
    return params.m_commonParams
end

function detParams(params::ParamsOnGPU, i::UInt32)
    return params.m_detParams[i]
end

function layerGeometry(params::ParamsOnGPU)
    return params.m_layerGeometry
end

function averageGeometry(params::ParamsOnGPU)
    return params.m_averageGeometry
end

function layer(params::ParamsOnGPU, id::UInt16)
    return params.m_layerGeometry.layer[id ÷ Phase1PixelTopology.maxModuleStride]
end

const MaxHitsInIter = MAX_HITS_IN_ITER()

"""
   ### ClusParamsT{N}
    Template struct for cluster parameters with a fixed size N.

    - `minRow::NTuple{N, UInt32}`: Minimum row indices for clusters.
    - `maxRow::NTuple{N, UInt32}`: Maximum row indices for clusters.
    - `minCol::NTuple{N, UInt32}`: Minimum column indices for clusters.
    - `maxCol::NTuple{N, UInt32}`: Maximum column indices for clusters.
    - `Q_f_X::NTuple{N, Int32}`: First charge values in x-direction.
    - `Q_l_X::NTuple{N, Int32}`: Last charge values in x-direction.
    - `Q_f_Y::NTuple{N, Int32}`: First charge values in y-direction.
    - `Q_l_Y::NTuple{N, Int32}`: Last charge values in y-direction.
    - `charge::NTuple{N, Int32}`: Charge values.
    - `xpos::NTuple{N, Float32}`: X positions.
    - `ypos::NTuple{N, Float32}`: Y positions.
    - `xerr::NTuple{N, Float32}`: X error values.
    - `yerr::NTuple{N, Float32}`: Y error values.
    - `xsize::NTuple{N, Int16}`: X sizes.
    - `ysize::NTuple{N, Int16}`: Y sizes.

"""

struct ClusParamsT{N}
    minRow::Vector{UInt32}
    maxRow::Vector{UInt32}
    minCol::Vector{UInt32}
    maxCol::Vector{UInt32}

    Q_f_X::Vector{Int32}
    Q_l_X::Vector{Int32}
    Q_f_Y::Vector{Int32}
    Q_l_Y::Vector{Int32}

    charge::Vector{Int32}

    xpos::Vector{Float32}
    ypos::Vector{Float32}

    xerr::Vector{Float32}
    yerr::Vector{Float32}

    xsize::Vector{Int16}
    ysize::Vector{Int16}

    function ClusParamsT{N}() where N
        return new(zeros(UInt32,N),zeros(UInt32,N),zeros(UInt32,N),zeros(UInt32,N),
        zeros(Int32,N),zeros(Int32,N),zeros(Int32,N),zeros(Int32,N),zeros(Int32,N),
        zeros(Float32,N),zeros(Float32,N),zeros(Float32,N),zeros(Float32,N),zeros(Int16,N),zeros(Int16,N))
    end

end

"""
### computeAnglesFromDet
    Computes the angles of a position relative to the detector.

    **Inputs**:
    - `detParams::DetParams`: Detector parameters including offsets and z-coordinate.
    - `x::Float32`: X-coordinate of the position.
    - `y::Float32`: Y-coordinate of the position.

    **Outputs**:
    - `cotalpha::Float32`: Cotangent of the alpha angle.
    - `cotbeta::Float32`: Cotangent of the beta angle.

"""

function computeAnglesFromDet(detParams::DetParams, x::Float32, y::Float32)
    gvx = x - detParams.x0
    gvy = y - detParams.y0
    gvz = -1.0f0 / detParams.z0

    cotalpha = gvx * gvz
    cotbeta = gvy * gvz
    return cotalpha, cotbeta
end

"""
### correction
    Calculates the correction factor based on cluster size, charge values, and detector parameters.

    **Inputs**:
    - `sizeM1::Int32`: Size of the cluster minus one.
    - `Q_f::Int32`: First charge value.
    - `Q_l::Int32`: Last charge value.
    - `upper_edge_first_pix::UInt16`: Index of the first pixel at the upper edge.
    - `lower_edge_last_pix::UInt16`: Index of the last pixel at the lower edge.
    - `lorentz_shift::Float32`: Lorentz shift correction.
    - `theThickness::Float32`: Thickness of the detector.
    - `cot_angle::Float32`: Cotangent of the angle.
    - `pitch::Float32`: Pitch of the detector.
    - `first_is_big::Bool`: Whether the first pixel is big.
    - `last_is_big::Bool`: Whether the last pixel is big.

    **Outputs**:
    - `Float32`: Computed correction value.

"""
function correction(sizeM1, Q_f, Q_l, upper_edge_first_pix, lower_edge_last_pix,
                    lorentz_shift::Float32, theThickness::Float32, cot_angle::Float32, pitch::Float32,
                    first_is_big::Bool, last_is_big::Bool)::Float32
    if sizeM1 == 0
        return 0.0f0
    end

    W_eff = 0.0f0
    simple = true
    if sizeM1 == 1
        W_inner = pitch * Float32(lower_edge_last_pix - upper_edge_first_pix)
        W_pred = theThickness * cot_angle - lorentz_shift
        W_eff = abs(W_pred) - W_inner
        simple = (W_eff < 0.0f0) || (W_eff > pitch)
    end

    if simple
        sum_of_edge = 2.0f0
        sum_of_edge += first_is_big ? 1.0f0 : 0.0f0
        sum_of_edge += last_is_big ? 1.0f0 : 0.0f0
        W_eff = pitch * 0.5f0 * sum_of_edge
    end

    Qdiff = Float32(Q_l - Q_f)
    Qsum = Float32(Q_l + Q_f)
    Qsum = Qsum == 0.0f0 ? 1.0f0 : Qsum

    return 0.5f0 * (Qdiff / Qsum) * W_eff
end

"""
### position
    Computes the position of a cluster in the detector and applies corrections.

    **Inputs**:
    - `comParams::CommonParams`: Common parameters including pitch and thickness.
    - `detParams::DetParams`: Detector-specific parameters.
    - `cp::ClusParamsT{N}`: Cluster parameters.
    - `ic::UInt32`: Index of the cluster.

    **Outputs**:
    - Updates `cp.xpos[ic]` and `cp.ypos[ic]` with the corrected x and y positions.

"""
function position_corr(comParams::CommonParams, detParams::DetParams, cp::ClusParamsT{N}, ic::UInt32) where {N}
    # file = open("continue.txt", "w")

    llx = UInt16(cp.minRow[ic] + 1)
    # write(file,"llx = $llx\n")
    lly = UInt16(cp.minCol[ic] + 1)
    # write(file,"lly = $lly\n")
    urx = UInt16(cp.maxRow[ic])
    # write(file,"urx = $urx\n")
    ury = UInt16(cp.maxCol[ic])
    # write(file,"ury = $ury\n")

    llxl = local_x(llx)
    # write(file,"llxl = $llxl\n")
    llyl = local_y(lly)
    # write(file,"llyl = $llyl\n")
    urxl = local_x(urx)
    # write(file,"urxl = $urxl\n")
    uryl = local_y(ury)
    # write(file,"uryl = $uryl\n")

    mx = llxl + urxl
    # write(file, "mx = $mx\n")  
    my = llyl + uryl
    # write(file, "my = $my\n")  

    xsize = Int32(urxl) + 2 - Int32(llxl)
    # write(file,"xsize = $xsize\n")
    ysize = Int32(uryl) + 2 - Int32(llyl)
    # write(file,"ysize = $ysize\n")
    @assert xsize >= 0
    @assert ysize >= 0

    if is_big_pix_x(cp.minRow[ic])
        xsize += 1
    #    write(file,"xsize = $xsize\n")
    end
    if is_big_pix_x(cp.maxRow[ic])
        xsize += 1
    #    write(file,"xsize = $xsize\n")
    end
    if is_big_pix_y(cp.minCol[ic])
        ysize += 1
    #    write(file,"ysize = $ysize\n")
    end
    if is_big_pix_y(cp.maxCol[ic])
        ysize += 1
    #    write(file,"ysize = $ysize\n")
    end
    # write(file,"xsize = $xsize\n")
    # write(file,"ysize = $ysize\n")


    unbalanceX = Int32(trunc(8.0 * abs(Float32(cp.Q_f_X[ic] - cp.Q_l_X[ic])) / Float32(cp.Q_f_X[ic] + cp.Q_l_X[ic])))
    # write(file,"unbalanceX = $unbalanceX\n")
    unbalanceY = Int32(trunc(8.0 * abs(Float32(cp.Q_f_Y[ic] - cp.Q_l_Y[ic])) / Float32(cp.Q_f_Y[ic] + cp.Q_l_Y[ic])))
    # write(file,"unbalanceY = $unbalanceY\n")
    xsize = 8 * xsize - unbalanceX
#    write(file,"xsize = $xsize\n")
    ysize = 8 * ysize - unbalanceY
#    write(file,"ysize = $ysize\n")


    # print(unbalanceX, " ", unbalanceY, " ", xsize, " ", ysize)

    cp.xsize[ic] = UInt32(min(xsize, 1023))
#    write(file,"cp.xsize[ic] = $(cp.xsize[ic])\n")
    cp.ysize[ic] = UInt32(min(ysize, 1023))
#    write(file,"cp.ysize[ic] = $(cp.ysize[ic])\n")

    if cp.minRow[ic] == 0 || cp.maxRow[ic] == last_row_in_module
        cp.xsize[ic] = -cp.xsize[ic]
    #    write(file,"cp.xsize[ic] = $(cp.xsize[ic])\n")
    end
    if cp.minCol[ic] == 0 || cp.maxCol[ic] == last_col_in_module
        cp.ysize[ic] = -cp.ysize[ic]
    #    write(file,"cp.ysize[ic] = $(cp.ysize[ic])\n")
    end

    xPos = detParams.shiftX + comParams.thePitchX * (0.5f0 * Float32(mx) + Float32(x_offset))
    # write(file, "detParams.shiftX = ", @sprintf("%.9f", detParams.shiftX), "\n")
    # write(file, "comParams.thePitchX = ", @sprintf("%.9f", comParams.thePitchX), "\n")
    # write(file, "Val = ", @sprintf("%.9f", (0.5f0 * Float32(mx) + Float32(x_offset))), "\n")
    # write(file, @sprintf("%.9f", xPos), "\n")

    yPos = detParams.shiftY + comParams.thePitchY * (0.5f0 * Float32(my) + Float32(y_offset))
    # write(file, "detParams.shiftY = ", @sprintf("%.9f", detParams.shiftY), "\n")
    # write(file, "comParams.thePitchY = ", @sprintf("%.9f", comParams.thePitchY), "\n")
    # write(file, "Val = ", @sprintf("%.9f", (0.5f0 * Float32(my) + Float32(y_offset))), "\n")
    # write(file, @sprintf("%.9f", yPos), "\n")


    cotalpha, cotbeta = computeAnglesFromDet(detParams, xPos, yPos)

#    write(file,"cotalpha = $cotalpha\n")
#    write(file,"cotbeta = $cotbeta\n")


    thickness = detParams.isBarrel ? comParams.theThicknessB : comParams.theThicknessE
#    write(file,"thickness = $thickness\n")



    xcorr = correction(cp.maxRow[ic] - cp.minRow[ic], cp.Q_f_X[ic], cp.Q_l_X[ic], llxl, urxl, detParams.chargeWidthX,
                    thickness, cotalpha, comParams.thePitchX, is_big_pix_x(cp.minRow[ic]), is_big_pix_x(cp.maxRow[ic]))
    # write(file, "maxRow - minRow: $(cp.maxRow[ic] - cp.minRow[ic])\n")
    # write(file, "Q_f_X: $(cp.Q_f_X[ic])\n")
    # write(file, "Q_l_X: $(cp.Q_l_X[ic])\n")
    # write(file, "llxl: $(llxl)\n")
    # write(file, "urxl: $(urxl)\n")
    # write(file, "chargeWidthX: ", @sprintf("%.9f", detParams.chargeWidthX), "\n")
    # write(file, "thickness: ", @sprintf("%.9f", thickness), "\n")
    # write(file, "cotalpha: ", @sprintf("%.9f", cotalpha), "\n")
    # write(file, "thePitchX: ", @sprintf("%.9f", comParams.thePitchX), "\n")
    # write(file, "isBigPixX(minRow): $(Int(is_big_pix_x(cp.minRow[ic])))\n")
    # write(file, "isBigPixX(maxRow): $(Int(is_big_pix_x(cp.maxRow[ic])))\n")
    # write(file, "xcorr: ", @sprintf("%.9f", xcorr), "\n")

    ycorr = correction(cp.maxCol[ic] - cp.minCol[ic], cp.Q_f_Y[ic], cp.Q_l_Y[ic], llyl, uryl, detParams.chargeWidthY,
                    thickness, cotbeta, comParams.thePitchY, is_big_pix_y(cp.minCol[ic]), is_big_pix_y(cp.maxCol[ic]))

    # write(file, "maxCol - minCol: $(cp.maxCol[ic] - cp.minCol[ic])\n")
    # write(file, "Q_f_Y: $(cp.Q_f_Y[ic])\n")
    # write(file, "Q_l_Y: $(cp.Q_l_Y[ic])\n")
    # write(file, "llyl: $(llyl)\n")
    # write(file, "uryl: $(uryl)\n")
    # write(file, "chargeWidthY: ", @sprintf("%.9f", detParams.chargeWidthY), "\n")
    # write(file, "thickness: ", @sprintf("%.9f", thickness), "\n")
    # write(file, "cotbeta: ", @sprintf("%.9f", cotbeta), "\n")
    # write(file, "thePitchY: ", @sprintf("%.9f", comParams.thePitchY), "\n")
    # write(file, "isBigPixY(minCol): $(Int(is_big_pix_y(cp.minCol[ic])))\n")
    # write(file, "isBigPixY(maxCol): $(Int(is_big_pix_y(cp.maxCol[ic])))\n")
    # write(file, "ycorr: ", @sprintf("%.9f", ycorr), "\n")


    cp.xpos[ic] = xPos + xcorr
    cp.ypos[ic] = yPos + ycorr

#    write(file,"cp.xpos[ic] = $(cp.xpos[ic])\n")
#    write(file,"cp.ypos[ic] = $(cp.ypos[ic])\n")
end

"""
### errorFromSize
    Updates error estimates based on cluster size and detector parameters.

    **Inputs**:
    - `comParams::CommonParams`: Common parameters for the detector.
    - `detParams::DetParams`: Detector-specific parameters including error values.
    - `cp::ClusParamsT{N}`: Cluster parameters.
    - `ic::Int`: Index of the cluster.

    **Outputs**:
    - Updates `cp.xerr[ic]` and `cp.yerr[ic]` with the computed error values based on cluster size and position.

"""

function errorFromSize(comParams::CommonParams, detParams::DetParams, cp::ClusParamsT{N}, ic::Int) where {N}
    # Edge cluster errors
    cp.xerr[ic] = 0.0050f0
    cp.yerr[ic] = 0.0085f0

    # FIXME these are errors from Run1
    xerr_barrel_l1 = (0.00115f0, 0.00120f0, 0.00088f0)
    xerr_barrel_l1_def = 0.00200f0
    yerr_barrel_l1 = (0.00375f0, 0.00230f0, 0.00250f0, 0.00250f0, 0.00230f0, 0.00230f0, 0.00210f0, 0.00210f0, 0.00240f0)
    yerr_barrel_l1_def = 0.00210f0
    xerr_barrel_ln = (0.00115f0, 0.00120f0, 0.00088f0)
    xerr_barrel_ln_def = 0.00200f0
    yerr_barrel_ln = (0.00375f0, 0.00230f0, 0.00250f0, 0.00250f0, 0.00230f0, 0.00230f0, 0.00210f0, 0.00210f0, 0.00240f0)
    yerr_barrel_ln_def = 0.00210f0
    xerr_endcap = (0.0020f0, 0.0020f0)
    xerr_endcap_def = 0.0020f0
    yerr_endcap = (0.00210f0,)
    yerr_endcap_def = 0.00210f0

    sx = cp.maxRow[ic] - cp.minRow[ic]
    sy = cp.maxCol[ic] - cp.minCol[ic]

    # is edgy ?
    isEdgeX = cp.minRow[ic] == 0 || cp.maxRow[ic] == last_row_in_module
    isEdgeY = cp.minCol[ic] == 0 || cp.maxCol[ic] == last_col_in_module
    # is one and big?
    isBig1X = (0 == sx) && isBigPixX(cp.minRow[ic])
    isBig1Y = (0 == sy) && isBigPixY(cp.minCol[ic])

    if !isEdgeX && !isBig1X
        if !detParams.isBarrel
            cp.xerr[ic] = sx < length(xerr_endcap) ? xerr_endcap[sx] : xerr_endcap_def
        elseif detParams.layer == 1
            cp.xerr[ic] = sx < length(xerr_barrel_l1) ? xerr_barrel_l1[sx] : xerr_barrel_l1_def
        else
            cp.xerr[ic] = sx < length(xerr_barrel_ln) ? xerr_barrel_ln[sx] : xerr_barrel_ln_def
        end
    end

    if !isEdgeY && !isBig1Y
        if !detParams.isBarrel
            cp.yerr[ic] = sy < length(yerr_endcap) ? yerr_endcap[sy] : yerr_endcap_def
        elseif detParams.layer == 1
            cp.yerr[ic] = sy < length(yerr_barrel_l1) ? yerr_barrel_l1[sy] : yerr_barrel_l1_def
        else
            cp.yerr[ic] = sy < length(yerr_barrel_ln) ? yerr_barrel_ln[sy] : yerr_barrel_ln_def
        end
    end
end

"""
### errorFromDB
    Computes error estimates based on cluster size and database parameters.

    **Inputs**:
    - `comParams::CommonParams`: Common parameters for the detector.
    - `detParams::DetParams`: Detector-specific parameters including error values.
    - `cp::ClusParamsT{N}`: Cluster parameters.
    - `ic::Int`: Index of the cluster.

    **Outputs**:
    - Updates `cp.xerr[ic]` and `cp.yerr[ic]` with the error values based on the cluster size and position.
"""

function errorFromDB(comParams::CommonParams, detParams::DetParams, cp::ClusParamsT{N}, ic::UInt32) where {N}
    # file = open("continue2.txt", "w")

    # Edge cluster errors
    cp.xerr[ic] = 0.0050f0
    cp.yerr[ic] = 0.0085f0

    sx = cp.maxRow[ic] - cp.minRow[ic]
#    write(file,"sx = $sx\n")
    sy = cp.maxCol[ic] - cp.minCol[ic]
#    write(file,"sy = $sy\n")

    # is edgy ?
    isEdgeX = cp.minRow[ic] == 0 || cp.maxRow[ic] == last_row_in_module
#    write(file,"isEdgeX = $isEdgeX\n")
    isEdgeY = cp.minCol[ic] == 0 || cp.maxCol[ic] == last_col_in_module
#    write(file,"isEdgeY = $isEdgeY\n")
    # is one and big?
    ix = (0 == sx) ? 1 : 0
#    write(file,"ix = $ix\n")
    iy = (0 == sy) ? 1 : 0
#    write(file,"iy = $iy\n")
    ix += (0 == sx) && is_big_pix_x(cp.minRow[ic]) ? 1 : 0
#    write(file,"ix = $ix\n")
    iy += (0 == sy) && is_big_pix_y(cp.minCol[ic]) ? 1 : 0
#    write(file,"iy = $iy\n")

    if !isEdgeX
        cp.xerr[ic] = detParams.sx[ix + 1]
    #    write(file,"cp.xerr[ic] = $(cp.xerr[ic])\n")
    end
    if !isEdgeY
        cp.yerr[ic] = detParams.sy[iy + 1]
    #    write(file,"cp.yerr[ic] = $(cp.yerr[ic])\n")
    end
end

end