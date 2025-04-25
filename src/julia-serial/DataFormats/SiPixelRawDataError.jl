module DataFormatsSiPixelRawDataError

    """
    SiPixelRawDataError struct represents errors in SiPixel raw data processing.

    Fields:
      - error_word_32::UInt32: Error word (32-bit representation).
      - error_word_64::UInt64: Error word (64-bit representation).
      - error_type::Int: Type of error.
      - fed_id::Int: FED ID associated with the error.
      - error_message::String: Human-readable error message.

    Constructors:
      - SiPixelRawDataError(error_word_32::UInt32, error_type::Int, fed_id::Int): Creates an instance with 32-bit error word.
      - SiPixelRawDataError(error_word_64::UInt64, error_type::Int, fed_id::Int): Creates an instance with 64-bit error word.

    """
    struct SiPixelRawDataError
        error_word_32::UInt32
        error_word_64::UInt64
        error_type::Int
        fed_id::Int
        error_message::String

        function SiPixelRawDataError()
           new(zero(UInt32),zero(UInt64),zero(Int),zero(Int),"")
        end

        function SiPixelRawDataError(error_word_32::UInt32, error_type::Int, fed_id::Int)
            error_word_32 = error_word_32
            error_word_64 = 0
            error_type = error_type
            fed_id = fed_id
            error_message = set_message(_error_type_)
            
            new(error_word_32, error_word_64, _error_type_, fed_id, error_message)
        end
        
        function SiPixelRawDataError(error_word_64::UInt64, error_type::Int, fed_id::Int)
            error_word_32 = 0
            error_word_64 = error_word_64
            error_type = error_type
            fed_id = fed_id
            error_message = set_message(_error_type_)
            
            new(error_word_32, error_word_64, error_type, _fed_id_, error_message)
        end
    end

    """
    set_message(error_type::Int) -> String

    Returns a human-readable error message based on the error type.

    Inputs:
      - error_type::Int: Type of error.

    Output:
      - error_message::String: Error message corresponding to the error type.

    """
    function set_message(error_type::Int)
        if error_type == 25
            return "Error: Disabled FED channel (ROC=25)"
        elseif error_type == 26
            return "Error: Gap word"
        elseif error_type == 27
            return "Error: Dummy word"
        elseif error_type == 28
            return "Error: FIFO nearly full"
        elseif error_type == 29
            return "Error: Timeout"
        elseif error_type == 30
            return "Error: Trailer"
        elseif error_type == 31
            return "Error: Event number mismatch"
        elseif error_type == 32
            return "Error: Invalid or missing header"
        elseif error_type == 33
            return "Error: Invalid or missing trailer"
        elseif error_type == 34
            return "Error: Size mismatch"
        elseif error_type == 35
            return "Error: Invalid channel"
        elseif error_type == 36
            return "Error: Invalid ROC number"
        elseif error_type == 37
            return "Error: Invalid dcol/pixel address"
        else
            return "Error: Unknown error type"
        end
    end

    """
    set_word_32(self::SiPixelRawDataError, error_word_32::UInt32) -> nothing

    Sets the 32-bit error word in the SiPixelRawDataError struct.

    Inputs:
      - self::SiPixelRawDataError: Instance of SiPixelRawDataError.
      - error_word_32::UInt32: 32-bit error word to set.

    Output:
      - Nothing.

    """
    function set_word_32(self::SiPixelRawDataError, error_word_32::UInt32)
        self.error_word_32 = error_word_32
    end

    """
    set_word_64(self::SiPixelRawDataError, error_word_64::UInt64) -> nothing

    Sets the 64-bit error word in the SiPixelRawDataError struct.

    Inputs:
      - self::SiPixelRawDataError: Instance of SiPixelRawDataError.
      - error_word_64::UInt64: 64-bit error word to set.

    Output:
      - Nothing.
    """
    function set_word_64(self::SiPixelRawDataError, error_word_64::UInt64)
        self.error_word_64 = error_word_64
    end

    """
    set_Type(self::SiPixelRawDataError, error_type::Int) -> nothing

    Sets the error type and updates the error message in the SiPixelRawDataError struct.

    Inputs:
      - self::SiPixelRawDataError: Instance of SiPixelRawDataError.
      - error_type::Int: Error type to set.

    Output:
      - Nothing.
    """
    function set_Type(self::SiPixelRawDataError, error_type::Int)
        self.error_type = error_type
        self.error_message = set_message(error_type)
    end

    """
    set_fed_id(self::SiPixelRawDataError, fed_id::Int) -> nothing

    Sets the FED ID in the SiPixelRawDataError struct.

    Inputs:
      - self::SiPixelRawDataError: Instance of SiPixelRawDataError.
      - fed_id::Int: FED ID to set.

    Output:
      - Nothing.

    """
    @inline function set_fed_id(self::SiPixelRawDataError, fed_id::Int)
        self.fed_id = fed_id
    end

    """
    get_message(self::SiPixelRawDataError) -> String

    Returns the error message associated with the SiPixelRawDataError instance.

    Inputs:
      - self::SiPixelRawDataError: Instance of SiPixelRawDataError.

    Output:
      - error_message::String: Error message associated with the instance.

    """
    @inline function get_message(self::SiPixelRawDataError)
        return self.error_message
    end

    """
    get_word_32(self::SiPixelRawDataError) -> UInt32

    Returns the 32-bit error word from the SiPixelRawDataError instance.

    Inputs:
      - self::SiPixelRawDataError: Instance of SiPixelRawDataError.

    Output:
      - error_word_32::UInt32: 32-bit error word.

    """
    @inline function get_word_32(self::SiPixelRawDataError)::UInt32
        return self.error_word_32
    end

    """
    get_word_64(self::SiPixelRawDataError) -> UInt64

    Returns the 64-bit error word from the SiPixelRawDataError instance.

    Inputs:
      - self::SiPixelRawDataError: Instance of SiPixelRawDataError.

    Output:
      - error_word_64::UInt64: 64-bit error word.

    """
    @inline function get_word_64(self::SiPixelRawDataError)::UInt64
        return self.error_word_64
    end

    """
    get_type(self::SiPixelRawDataError) -> Int

    Returns the error type from the SiPixelRawDataError instance.

    Inputs:
      - self::SiPixelRawDataError: Instance of SiPixelRawDataError.

    Output:
      - error_type::Int: Error type.

    """
    @inline function get_type(self::SiPixelRawDataError)::Int
        return self.error_type
    end

    """
    get_fed_id(self::SiPixelRawDataError) -> Int

    Returns the FED ID from the SiPixelRawDataError instance.

    Inputs:
      - self::SiPixelRawDataError: Instance of SiPixelRawDataError.

    Output:
      - fed_id::Int: FED ID.

    """
    @inline function get_fed_id(self::SiPixelRawDataError)::Int
        return self.fed_id
    end

    """
    Custom comparison function for SiPixelRawDataError instances.

    Inputs:
      - one::SiPixelRawDataError: First instance for comparison.
      - other::SiPixelRawDataError: Second instance for comparison.

    Output:
      - Bool: Returns true if the FED ID of `one` is less than `other`.
    """
    
    import Base: isless

    @inline isless(one::SiPixelRawDataError, other::SiPixelRawDataError) = one.get_fed_id() < other.get_fed_id()

end # module
