module condFormatsSiPixelFedIds

export fed_ids

    """
    Struct to hold siPixel FED IDs

    This struct stores a list of FED (Front-End Driver) IDs that are used in the pixel detector system. FED IDs uniquely identify each FED unit, which is responsible for reading out data from the detector.

    # Fields
    - _fed_ids::Vector{UInt}: A vector storing the list of FED IDs.

    # Constructor
    Initializes the SiPixelFedIds structure with a given list of FED IDs.
    """

    # Stripped-down version of SiPixelFedCablingMap
    struct SiPixelFedIds
        _fed_ids::Vector{UInt32}

        function SiPixelFedIds(fed_ids::Vector{UInt32})
            new(fed_ids)
        end
    end

    """
    Retrieves the list of FED IDs from the SiPixelFedIds structure.
    """
    fed_ids(ids::SiPixelFedIds)::Vector{UInt32} = ids._fed_ids

end # module condFormatsSiPixelFedIds
