

module DataFormatsSiPixelDigiInterfacePixelErrors
  using ..DataFormatsSiPixelRawDataError: SiPixelRawDataError 
  export PixelErrorCompact, PixelFormatterErrors
  """
  Definition of PixelErrorCompact struct representing compact pixel error information.

  Fields:
    - rawId::UInt32: Raw ID associated with the error.
    - word::UInt32: Word representing error details.
    - errorType::UInt8: Type of error.
    - fedId::UInt8: FED ID associated with the error.
  """
  struct PixelErrorCompact
      raw_id::UInt32
      word::UInt32
      erro_type::UInt8
      fed_id::UInt8

      function PixelErrorCompact()
        new(zero(UInt32),zero(UInt32),zero(UInt8),zero(UInt8))
      end
      function PixelErrorCompact(raw_id::UInt32,word::UInt32,erro_type::UInt8,fed_id::UInt8)
        new(raw_id,word,erro_type,fed_id)
      end
  end

  """
  PixelFormatterErrors is a dictionary storing pixel formatter errors.

  Key Type: UInt32
  Value Type: Vector{Main.DataFormats_SiPixelRawDataError_h.SiPixelRawDataError}

  """
  const PixelFormatterErrors = Dict{UInt32, Vector{SiPixelRawDataError}}

end # module
