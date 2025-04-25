module dataFormats

    export FedRawDataCollection, FedRawData, FedData
    """
    FedNumbering constants and functions

    Constants:
    - MINSiPixeluTCAFedID: Minimum Fed ID for SiPixel uTCA.
    - MAXSiPixeluTCAFedID: Maximum Fed ID for SiPixel uTCA.
    - MAXFedID: Maximum Fed ID.

    Functions:
    - inrange(FedID::Int): Checks if a given Fed ID is within the SiPixel uTCA Fed ID range.
    """
    const MIN_SiPixel_uTCA_FED_ID = 1200
    const MAX_SiPixel_uTCA_FED_ID = 1349
    const MAX_FED_ID = 4096

    function inrange(fed_id::Int)
        if fed_id >= MIN_SiPixel_uTCA_FED_ID && Fed_id <= MAX_SiPixel_uTCA_FED_ID
            return true 
        end
        return false 
    end


    """
    FedRawData struct

    Represents raw data from a Fed.

    Fields:
    - data::Vector{UInt8}: The raw data buffer.

    Constructor:
    - FedRawData(newsize::Int): Constructs a FedRawData object with a preallocated size in bytes. 
      The size must be a multiple of 8 bytes.
    - FedRawData(in::FedRawData): Copy constructor.
    """

    mutable struct FedRawData
        fedid::Int32
        data::Vector{UInt8}

        function FedRawData(new_size::Int)
            if newsize % 8 != 0
                throw(ArgumentError("FedRawData: newsize $newsize is not a multiple of 8 bytes."))
            end
            new(Vector{UInt8}(undef, new_size))
        end

        function FedRawData(id::Int32, dataa::Vector{UInt8})
            new(id,dataa)
        end
        # function FedRawData()
        #     new(0,UInt8[])
        # end
    end

    """
    Return the data buffer
    """
    function data(self::FedRawData)::Vector{UInt8}
        return self.data
    end

    """
    Length of the data buffer in bytes.
    """
    function Base.length(self::FedRawData)::Int
        return length(self.data)
    end

    """
    Resize to the specified size in bytes. It is required that the size is a multiple of the size of a Fed word (8 bytes).
    """
    function resize(self::FedRawData, newsize::Int)
        if newsize % 8 != 0
            throw(ArgumentError("FedRawData::resize: newsize $newsize is not a multiple of 8 bytes."))
        end
        resize!(self.data, newsize)
    end



    """
    FedRawDataCollection struct

    Represents a collection of FedRawData objects.

    Fields:
    - data::Vector{FedRawData}: Vector of FedRawData objects.

    Constructor:
    - FedRawDataCollection(in::FedRawDataCollection): Copy constructor.
    """



    mutable struct FedRawDataCollection
        data::Vector{FedRawData}  # Vector of FedRawData
        """
        Copy constructor.
        """
        FedRawDataCollection(in::FedRawDataCollection) = new(copy(in.data)) # copy constructor
        FedRawDataCollection(data::Vector{FedRawData}) = new(data)
        FedRawDataCollection() = new(Vector{FedRawData}(undef,MAX_FED_ID))
    end

    """
    Swap function for FedRawDataCollection.
    """
    function swap(a::FedRawDataCollection, b::FedRawDataCollection)
        a.data, b.data = b.data, a.data
    end
    """ 
    function for getting FedRawData
    """
    function FedData(self::FedRawDataCollection,Fedid :: Integer)
        return self.data[Fedid]
    end

    
    mutable struct SiPixelRawDataError
        errorWord32::UInt32
        errorWord64::UInt64
        errorType::Int
        FedId::Int
        errorMessage::String
        """
        Constructor for 32-bit error word
        """
        
        function SiPixelRawDataError(errorWord32::UInt32, errorType::Int, FedId::Int)
            new(errorWord32, UInt64(0), errorType, FedId, "")
        end
        """
        Constructor with 64-bit error word and type included (header or trailer word)
        """
        
        function SiPixelRawDataError(errorWord64::UInt64, errorType::Int, FedId::Int)
            new(UInt32(0), errorWord64, errorType, FedId, "")
        end
    end
    function setType!(error::SiPixelRawDataError, errorType::Int)
        error.errorType = errorType
        setMessage!(error)
    end
    function setFedId!(error::SiPixelRawDataError, FedId::Int)
        error.FedId = FedId
    end
    function setMessage!(error::SiPixelRawDataError)
        error.errorMessage = errorTypeMessage(error.errorType)
    end
    function errorTypeMessage(errorType::Int)::String
        return Dict(
            25 => "Error: Disabled Fed channel (ROC=25)",
            26 => "Error: Gap word",
            27 => "Error: Dummy word",
            28 => "Error: FIFO nearly full",
            29 => "Error: Timeout",
            30 => "Error: Trailer",
            31 => "Error: Event number mismatch",
            32 => "Error: Invalid or missing header",
            33 => "Error: Invalid or missing trailer",
            34 => "Error: Size mismatch",
            35 => "Error: Invalid channel",
            36 => "Error: Invalid ROC number",
            37 => "Error: Invalid dcol/pixel address",
        )[errorType, "Error: Unknown error type"]
    end

end # module DataFormats