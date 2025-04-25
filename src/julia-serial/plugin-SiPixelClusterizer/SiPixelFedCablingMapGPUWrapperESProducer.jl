using .condFormatsSiPixelFedIds:SiPixelFedIds

using .recoLocalTrackerSiPixelClusterizerSiPixelFedCablingMapGPU:SiPixelFedCablingMapGPU
using .recoLocalTrackerSiPixelClusterizerSiPixelFedCablingMapGPU.pixelGPUDetails

using .recoLocalTrackerSiPixelClusterizerSiPixelFedCablingMapGPUWrapper:SiPixelFedCablingMapGPUWrapper

using .condFormatsSiPixelFedIds

using .ESPluginFactory


struct SiPixelFedCablingMapGPUWrapperESProducer <: ESProducer
    data::String  # Use String to represent the path

    function SiPixelFedCablingMapGPUWrapperESProducer(datadir::String)
        new(datadir)
    end
end

function readData(io::IOStream, type, size)
    container = Vector{type}(undef, size)
    return read!(io,container)
end


function readCablingMap(io::IOStream,es::EventSetup)

    maxSize = recoLocalTrackerSiPixelClusterizerSiPixelFedCablingMapGPU.pixelGPUDetails.MAX_SIZE

    fed = readData(io, UInt32, maxSize)
    link = readData(io, UInt32, maxSize)
    roc = readData(io, UInt32, maxSize)
    raw_id = readData(io, UInt32, maxSize)
    roc_in_det = readData(io, UInt32, maxSize)
    module_id = readData(io, UInt32, maxSize)
    bad_rocs = readData(io, UInt8, maxSize)
    size = UInt32(0)

    cablingMap = SiPixelFedCablingMapGPU(fed, link, roc, raw_id, roc_in_det, module_id, bad_rocs, size)

    mod_to_unp_def_size = read(io,UInt32)
    mod_to_unp_default = readData(io, UInt8, mod_to_unp_def_size)

    put!(es,SiPixelFedCablingMapGPUWrapper(cablingMap,mod_to_unp_default))
end


function produce(producer::SiPixelFedCablingMapGPUWrapperESProducer, eventSetup::EventSetup)
    fed_ids_file = joinpath(producer.data, "fedIds.bin")
    # Read fedIds.bin
    open(fed_ids_file, "r") do io
        nfeds = read(io, UInt32)
        fed_ids = Vector{UInt32}(undef, nfeds)
        read!(io, fed_ids)
        put!(eventSetup,SiPixelFedIds(fed_ids))
    end
    

    # Read cablingMap.bin
    cabling_map_file = joinpath(producer.data, "cablingMap.bin")
    
    open(cabling_map_file, "r") do io
        readCablingMap(io,eventSetup)
    end

end



add_plugin("SiPixelFedCablingMapGPUWrapperESProducer",x -> SiPixelFedCablingMapGPUWrapperESProducer(x))


