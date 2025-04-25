using .CalibTrackerSiPixelESProducersInterfaceSiPixelGainCalibrationForHLTGPU
using .condFormatsSiPixelObjectsSiPixelGainForHLTonGPU
using .ESPluginFactory


struct SiPixelGainCalibrationForHLTGPUESProducer <: ESProducer
    data::String  # Use String to represent the path

    function SiPixelGainCalibrationForHLTGPUESProducer(datadir::String)
        new(datadir)
    end
end

function readGain(io::IOStream,es::EventSetup)

    read(io,8) # skip the _v_pedestals pointer

    range_and_cols = Vector{RangeAndCols}(undef,RANGE_COUNT)
    read!(io,range_and_cols)


    _min_ped = read(io,Float32)
    _max_ped = read(io,Float32)
    _min_gain = read(io,Float32)
    _max_gain = read(io,Float32)

    ped_precision = read(io,Float32)
    # println(ped_precision)
    gain_precision = read(io,Float32)
    # println(gain_precision)

    _number_of_rows_averaged_over = read(io,UInt32)
    _n_bins_to_use_for_encoding = read(io,UInt32)
    _dead_flag = read(io,UInt32)
    _noisy_flag = read(io,UInt32)

    nbytes = read(io,UInt32)
    # println(nbytes)
    

    size:: UInt32= nbytes ÷ 2 # over 2 because we need half the bytes
    v_pedestals = Vector{DecodingStructure}(undef,size) 
    read!(io,v_pedestals)

    gain = SiPixelGainForHLTonGPU(v_pedestals,range_and_cols,_min_ped,_max_ped,_min_gain,_max_gain,ped_precision,gain_precision,_number_of_rows_averaged_over,_n_bins_to_use_for_encoding,_dead_flag,_noisy_flag)
    put!(es,SiPixelGainCalibrationForHLTGPU(gain))
end


function produce(producer::SiPixelGainCalibrationForHLTGPUESProducer, eventSetup::EventSetup)
    gain_file = joinpath(producer.data, "gain.bin")
    
    #read gain.bin
    open(gain_file, "r") do io
        readGain(io,eventSetup)        
    end
end


add_plugin("SiPixelGainCalibrationForHLTGPUESProducer",x -> SiPixelGainCalibrationForHLTGPUESProducer(x))
