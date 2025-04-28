using .Geometry_TrackerGeometryBuilder_phase1PixelTopology_h.phase1PixelTopology
using .PixelGPU_h
using .SOA_h
using .ESPluginFactory


struct PixelCPEFastESProducer <: ESProducer
    data::String  # Use String to represent the path

    function PixelCPEFastESProducer(datadir::String)
        new(datadir)
    end
end


function readDetParam(io::IOStream)
    isBarrel = read(io,Bool)
    isPosZ = read(io,Bool)
    layer = read(io,UInt16)
    index = read(io,UInt16)
    read(io,2)
    rawId = read(io,UInt32)
    # read(io,2)

    shiftX = read(io,Float32)
    shiftY = read(io,Float32)
    chargeWidthX = read(io,Float32)
    chargeWidthY = read(io,Float32)

    x0 = read(io,Float32)
    y0 = read(io,Float32)
    z0 = read(io,Float32)

    sx = NTuple{3, Float32}
    sy = NTuple{3, Float32}

    sx = (read(io, Float32), read(io, Float32), read(io, Float32))
    sy = (read(io, Float32), read(io, Float32), read(io, Float32))

    px = read(io,Float32); py = read(io,Float32); pz = read(io,Float32)

    R11 = read(io,Float32); R12 = read(io,Float32); R13 = read(io,Float32);
    R21 = read(io,Float32); R22 = read(io,Float32); R23 = read(io,Float32);
    R31 = read(io,Float32); R32 = read(io,Float32); R33 = read(io,Float32);


    Rotation = SOARotation{Float32}(R11,R12,R13,
                                    R21,R22,R23,
                                    R31,R32,R33)

    Frame = SOAFrame{Float32}(px,py,pz,Rotation)


    DetParamss = DetParams(isBarrel,isPosZ,layer,index,rawId,shiftX,shiftY,chargeWidthX,chargeWidthY,x0,y0,z0,sx,sy,Frame)

    return DetParamss
end




function readCpeFast(io::IOStream,es::EventSetup)
    theThicknessB = read(io,Float32)
    theThicknessE = read(io,Float32)
    thePitchX = read(io,Float32)
    thePitchY = read(io,Float32)

    cmParams = CommonParams(theThicknessB,theThicknessE,thePitchX,thePitchY)

    ndetParams = read(io,UInt32)

    detParamsGPU = Vector{DetParams}()

    for i ∈ 1:ndetParams
        dParam = readDetParam(io)
        push!(detParamsGPU,dParam)
    end

    numberOfLaddersInBarrel = number_of_ladders_in_barrel
    ladderZ = readData(io,Float32,number_of_ladders_in_barrel)
    ladderX = readData(io,Float32,number_of_ladders_in_barrel)
    ladderY = readData(io,Float32,number_of_ladders_in_barrel)
    ladderR = readData(io,Float32,number_of_ladders_in_barrel)
    ladderMinZ = readData(io,Float32,number_of_ladders_in_barrel)
    ladderMaxZ = readData(io,Float32,number_of_ladders_in_barrel)

    endCapZ = zeros(Float64, 2)
    endCapZ[1] = read(io,Float32); endCapZ[2] = read(io,Float32);

    averageGeometry = AverageGeometry(UInt32(numberOfLaddersInBarrel),ladderZ,ladderX,ladderY,ladderR,ladderMinZ,ladderMaxZ,endCapZ)


    layerStart = readData(io,UInt32,number_of_layers + 1)
    layer = readData(io,UInt8,layer_index_size)

    layerGeometry = LayerGeometry(layerStart,layer)

    cpuData = ParamsOnGPU(cmParams,detParamsGPU,layerGeometry,averageGeometry)

    p_CPEFast = PixelCPEFast(detParamsGPU,cmParams,layerGeometry,averageGeometry,cpuData)

    put!(es,p_CPEFast)
end


function produce(producer::PixelCPEFastESProducer, eventSetup::EventSetup)
    cpe_file = joinpath(producer.data, "cpefast.bin")
    
    #read beamspot.bin
    open(cpe_file, "r") do io
        readCpeFast(io,eventSetup)
    end
end


add_plugin("PixelCPEFastESProducer",x -> PixelCPEFastESProducer(x))
