module condFormatsSiPixelObjectsSiPixelGainForHLTonGPU

export RANGE_COUNT

# using ..gpuConfig

using StaticArrays

export SiPixelGainForHLTonGPU, RangeAndCols, DecodingStructure

struct SiPixelGainForHLTonGPUDecodingStructure
    gain::UInt8
    ped::UInt8
end

const DecodingStructure = SiPixelGainForHLTonGPUDecodingStructure
struct RangeAndCols
    first::UInt32
    last::UInt32
    cols::Int32
end

const RANGE_COUNT = 2000

# copy of SiPixelGainCalibrationForHL
struct SiPixelGainForHLTonGPU
    v_pedestals::Vector{DecodingStructure}
    
    range_and_cols::Vector{RangeAndCols}
    

    _min_ped::Float32
    _max_ped::Float32
    _min_gain::Float32
    _max_gain::Float32

    ped_precision::Float32
    gain_precision::Float32

    _number_of_rows_averaged_over::UInt32 # this is 80!!!!
    _n_bins_to_use_for_encoding::UInt32
    _dead_flag::UInt32
    _noisy_flag::UInt32
end
    decode_gain(structure::SiPixelGainForHLTonGPU, gain)::Float32 = gain * structure.gain_precision + structure._min_gain
    decode_ped(structure::SiPixelGainForHLTonGPU, ped)::Float32 = ped * structure.ped_precision + structure._min_ped
@inline function get_ped_and_gain(structure::SiPixelGainForHLTonGPU, module_ind, col, row, is_dead_column_is_noisy_column::MVector{2, Bool})
    range = Pair(structure.range_and_cols[module_ind+1].first,structure.range_and_cols[module_ind+1].last)
    n_cols = structure.range_and_cols[module_ind+1].cols
    # print("module id : ",module_ind," ",range[1]," ",range[2]," ",n_cols,'\n')
    # open("pixel_gain_for_hlt_test.txt","a") do file
    #     write(file,"module id : ",string(module_ind)," ",string(range[1])," ",string(range[2])," ",string(n_cols),'\n')
    # end

    # determine what averaged data block we are in (there should be 1 or 2 of these depending on if plaquette is 1 by X or 2 by X
    length_of_column_data = (range[2] - range[1]) ÷ n_cols
    length_of_averaged_data_in_each_column = 2 # we always only have two values per column averaged block
    number_of_data_blocks_to_skip = row ÷ structure._number_of_rows_averaged_over
    offset = range[1] + col * length_of_column_data + length_of_averaged_data_in_each_column * number_of_data_blocks_to_skip
    @assert (offset < range[2])
    @assert (offset < 3088384)
    @assert ((offset % 2) == 0)
    
    lp = structure.v_pedestals
    s = lp[(offset ÷ 2) + 1]

    is_dead_column_is_noisy_column[1] = ((s.ped & 0xFF) == structure._dead_flag)
    is_dead_column_is_noisy_column[2] = ((s.ped & 0xFF) == structure._noisy_flag)
    # println(structure.ped_precision," ",structure._min_ped)

    # open("dataa.txt","a") do file
    #     write(file,'\n')
    # end
    return tuple(decode_ped(structure, s.ped & 0xFF), decode_gain(structure, s.gain & 0xFF))
end



end # module condFormatsSiPixelObjectsSiPixelGainForHLTonGPU
