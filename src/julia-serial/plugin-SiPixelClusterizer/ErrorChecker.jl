module errorChecker

    using ..DataFormatsSiPixelRawDataError: SiPixelRawDataError
    using ..constants
    using ..DataFormatsSiPixelDigiInterfacePixelErrors
    using ..fedHeader
    using ..fedTrailer


    # Type aliases for convenience
    const Word32 = UInt32
    const Word64 =  UInt64
    const DetErrors = Vector{SiPixelRawDataError}
    const Errors = PixelFormatterErrors

    export ErrorChecker, check_crc, check_header, check_trailer

    """
    ErrorChecker struct

    Represents an error checker object for validating FED data integrity.

    Fields:
    - _includeErrors::Bool: Flag indicating whether to include errors in analysis.

    Constructor:
    - ErrorChecker(): Constructs an ErrorChecker object with _includeErrors set to false.
    """

    struct ErrorChecker 
        _includeErrors::Bool
        function ErrorChecker()
            new(false)
        end
    end

    """
    check_crc function

    Checks CRC validity in a FED trailer.

    Arguments:
    - self::ErrorChecker: The ErrorChecker object.
    - errors_in_event::Bool: Indicates if errors are present in the event.
    - fed_id::Int: FED identifier.
    - trailer::Vector{UInt8}: Trailer data.
    - errors::Errors: Dictionary to store errors.

    Returns:
    - Bool: True if CRC is valid, false otherwise.
    """
    function check_crc(self::ErrorChecker, errors_in_event::Bool, fed_id::Integer, trailer::Vector{UInt8}, errors::Errors)::Bool
        the_trailer = FedTrailer(trailer)
        crc_bit::bool = crc_modified(the_trailer)
        error_word::UInt64 = reinterpret(UInt64,trailer[1:8])[1]
        if (crc_bit == false)
            return true
        end
        errors_in_event = true
        if(self._includeErrors)
            error = SiPixelRawDataError(error_word, 39, fed_id)
            push!(errors[dummyDetId], error)
        end 
        return false
    end

    """
    check_header function

    Checks header validity in a FED header.

    Arguments:
    - self::ErrorChecker: The ErrorChecker object.
    - errors_in_event::Bool: Indicates if errors are present in the event.
    - fed_id::Int: FED identifier.
    - header::Vector{UInt8}: Header data.
    - errors::Errors: Dictionary to store errors.

    Returns:
    - Bool: True if more headers follow, false otherwise.
    """
    function check_header(self::ErrorChecker, errors_in_event::Bool, fed_id::Integer, header::AbstractArray, errors::Errors)::Bool
        the_header = FedHeader(header) # allocations caused by this
        error_word::UInt64 = reinterpret(UInt64,header[1:8])[1] # allocations caused by this
        if(!fedHeader.check(the_header))
            return false
        end 
        source_id = fedHeader.source_id(the_header)
        if( source_id != fed_id)
            println("PixelDataFormatter::interpretRawData, fedHeader.sourceID() != fedId, sourceID = ",source_id, " fedId = ",fed_id, "errorType = ",32)
            errorsInEvent = true
            if(self._includeErrors)
                error = SiPixelRawDataError(error_word, 39, fedId)
                push!(errors[dummyDetId], error)
            end 
        end
        return fedHeader.more_headers(the_header)
    end 

    """
    check_trailer function

    Checks trailer validity in a FED trailer.

    Arguments:
    - self::ErrorChecker: The ErrorChecker object.
    - errors_in_event::Bool: Indicates if errors are present in the event.
    - fed_id::Int: FED identifier.
    - num_words::UInt: Number of words.
    - trailer::Vector{UInt8}: Trailer data.
    - errors::Errors: Dictionary to store errors.

    Returns:
    - Bool: True if more trailers follow, false otherwise.
    """
    function check_trailer(self::ErrorChecker, errors_in_event::Bool, fed_id::Integer, num_words::Integer, trailer::AbstractArray, errors::Errors)::Bool
        the_trailer = FedTrailer(trailer)
        error_word::UInt64 = reinterpret(UInt64,trailer[1:8])[1]
        if (!fedTrailer.check(the_trailer))
            if(self._includeErrors)
                error = SiPixelRawDataError(trailer, 39, fed_id)
                push!(errors[dummyDetId], error)
            end
            errors_in_event = true
            println("fedTrailer.check failed, Fed:",fedId," errorType = ",33)
            return false
        end
        if (fragment_length(the_trailer) != num_words)
            println("fedTrailer.fragmentLength()!= nWords !! Fed: ",fedId, " errorType = ",34)
            errors_in_event = true
            if (_includeErrors)
                error = SiPixelRawDataError(error_word, 39, fedId)
                push!(errors[dummyDetId], error)
            end
        end
        return more_trailers(the_trailer)
    end
end