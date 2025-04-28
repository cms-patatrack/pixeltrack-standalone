using .CUDADataFormatsSiPixelClusterInterfaceSiPixelClustersSoA:SiPixelClustersSoA

using .CUDADataFormatsSiPixelDigiInterfaceSiPixelDigisSoA:SiPixelDigisSoA

using .cudaDataFormatsSiPixelDigiInterfaceSiPixelDigiErrorsSoA:SiPixelDigiErrorsSoA

using .CalibTrackerSiPixelESProducersInterfaceSiPixelGainCalibrationForHLTGPU

using .recoLocalTrackerSiPixelClusterizerSiPixelFedCablingMapGPUWrapper

using .condFormatsSiPixelFedIds

using .dataFormats

using .errorChecker

using .pixelGPUDetails: SiPixelRawToClusterGPUKernel, WordFedAppender

using .DataFormatsSiPixelDigiInterfacePixelErrors: PixelErrorCompact, PixelFormatterErrors

import .recoLocalTrackerSiPixelClusterizerSiPixelFedCablingMapGPUWrapper.get_cpu_product

using .pixelGPUDetails

using .PluginFactory

mutable struct SiPixelRawToClusterCUDA <: EDProducer
    gpu_algo::SiPixelRawToClusterGPUKernel
    word_fed_appender::WordFedAppender
    errors::PixelFormatterErrors
    is_run2::Bool
    include_errors::Bool
    use_quality::Bool

    raw_get_token::EDGetTokenT{FedRawDataCollection}
    digi_put_token::EDPutTokenT{SiPixelDigisSoA}
    digi_error_put_token::EDPutTokenT{SiPixelDigiErrorsSoA}
    cluster_put_token::EDPutTokenT{SiPixelClustersSoA}

    function SiPixelRawToClusterCUDA(reg::ProductRegistry)
        is_run2 = true
        include_errors = true
        use_quality = true
        word_fed_appender = WordFedAppender()
        errors = PixelFormatterErrors()
        new(
            SiPixelRawToClusterGPUKernel(),
            word_fed_appender, errors, is_run2, include_errors, use_quality,
            consumes(reg,FedRawDataCollection),produces(reg,SiPixelDigisSoA),produces(reg,SiPixelDigiErrorsSoA),produces(reg,SiPixelClustersSoA))
    end
end


function produce(self:: SiPixelRawToClusterCUDA,event::Event, iSetup::EventSetup)
    hgpu_map = get(iSetup,SiPixelFedCablingMapGPUWrapper)   
    # if(has_quality(hgpu_map) != self.use_quality)
    #     error_message = "use_quality of the module ($self.use_quality) differs from SiPixelFedCablingMapGPUWrapper. Please fix your configuration."
    #     error(error_message)
    # end
    gpu_map = get_cpu_product(hgpu_map)
    gpu_modules_to_unpack::Vector{UInt8} = get_mod_to_unp_all(hgpu_map)
    hgains = get(iSetup,SiPixelGainCalibrationForHLTGPU)
    gpu_gains = CalibTrackerSiPixelESProducersInterfaceSiPixelGainCalibrationForHLTGPU.get_cpu_product(hgains)
    fed_ids::Vector{UInt} = get(iSetup,SiPixelFedIds)._fed_ids
    buffers::FedRawDataCollection = get(event,self.raw_get_token) #fedData
    empty!(self.errors)

    # # Data Extraction for Raw to Digi
    word_counter_gpu :: Int = 0 
    fed_counter:: Int = 0 
    errors_in_event:: Bool = false 
    error_check = ErrorChecker()
    #open("testingDigis.txt","w") do file

        for fed_id ∈ fed_ids
            if(fed_id == 40) # Skipping Pilot Blade Data
                continue
            end
            # write(file,"Fed ID : ")
            # write(file,string(fed_id)," ")
            @assert(fed_id >= 1200)
            fed_counter += 1
            raw_data = FedData(buffers,fed_id) 
            # write(file,string(raw_data.fedid)," ")
            # write(file,string(length(raw_data.data)))
            # write(file,"\n")
            # get event data for the following feds
            # Im using the fedId in fedIds to get the rawData of that fedId which is in buffers the FedRawDataCollection

            n_words = length(raw_data) ÷ sizeof(Int64)
            if(n_words == 0)
                continue
            end
            trailer_byte_start = length(raw_data) - 7
            trailer =  @views (dataFormats.data(raw_data)[trailer_byte_start:trailer_byte_start+7]) # The last 8 bytes

            #FIXME
            # if (!check_crc(error_check,errors_in_event, fed_id, trailer, self.errors)) 
            #     continue
            # end 
            header_byte_start = 1 
            header = @views (dataFormats.data(raw_data)[header_byte_start:header_byte_start+7])

            moreHeaders = true
            while moreHeaders
                headerStatus =  check_header(error_check,errors_in_event, fed_id, header, self.errors)
                moreHeaders = headerStatus
                if moreHeaders
                    header_byte_start += 8
                    header = @views (data(rawData)[header_byte_start:header_byte_start+7])
                end
            end

            moreTrailer = true
            while (moreTrailer)
                trailerStatus = check_trailer(error_check,errors_in_event, fed_id, n_words, trailer, self.errors)
                moreTrailer = trailerStatus
                if moreTrailer
                    trailer_byte_start -= 8
                    trailer = @views (dataFormats.data(rawData),trailer_byte_start:trailer_byte_start+7)
                end
            end 
            
            begin_word32_index = header_byte_start + 8
            end_word32_index = trailer_byte_start - 1 
            @assert((end_word32_index - begin_word32_index + 1) % 4 == 0)
            num_word32 = (end_word32_index - begin_word32_index + 1) ÷ sizeof(UInt32)
            @assert (0 == num_word32 % 2) # Number of 32 bit words should be a multiple of 2
            initialize_word_fed(self.word_fed_appender,fed_id,view(dataFormats.data(raw_data),begin_word32_index:end_word32_index),word_counter_gpu)
            word_counter_gpu += num_word32
        end 
    
    tmp = make_clusters(self.gpu_algo,self.is_run2,
                        gpu_map, 
                        gpu_modules_to_unpack, 
                        gpu_gains, 
                        self.word_fed_appender, 
                        self.errors, 
                        word_counter_gpu, # number of 32 bit words
                        fed_counter, # number of feds
                        self.use_quality, 
                        self.include_errors, 
                        false) #make clusters
    emplace(event,self.digi_put_token,tmp[1])
    x = tmp[2]
    
    emplace(event,self.cluster_put_token,tmp[2])

end  # module SiPixelRawToClusterCUDA


add_plugin_module("SiPixelRawToClusterCUDA",x -> SiPixelRawToClusterCUDA(x))
