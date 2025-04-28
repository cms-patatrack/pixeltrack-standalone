"""
    module fed_header

A module for handling FED headers in a typical FED system.

# FED System Overview
In a typical FED system, data is organized into frames or blocks, each beginning with a header containing metadata. The fields are extracted from these headers to interpret data correctly, ensure synchronization, and manage data flow efficiently across the system.

# Constants and Extraction Functions
This module provides constants and functions to extract various fields from the FED header.
"""

module fedHeader

    export FedHeader, check

    struct Fedh_t
        source_id::UInt32 # The Source Identifier
        event_id::UInt32 # The event Identifier
    end
    """
    event_id
    [ Control ID | Event Type |          Level 1 ID           ]
    [  4 bits    |  4 bits    |           24 bits             ]

    source_id
    [  BXID  | Source ID | Version | More Headers | Reserved ]
    [ 12 bits | 12 bits  |  4 bits |    1 bit     |  3 bits  ]


    *********************************************************************************
    *                                                                               *
    *  Usage in a FED System                                                        *
    *  ---------------------                                                        *
    *  In a typical FED system, data is organized into frames or blocks, each       *
    *  beginning with a header containing metadata. The fields above are extracted  *
    *  from these headers to interpret data correctly, ensure synchronization, and  *
    *  manage data flow efficiently across the system.                              *
    *                                                                               *
    *********************************************************************************

    """


    const FED_SLINK_START_MARKER = 0x5 # A const to indicate the start of a data frame or block in the FED system.

    """
    *********************************************************************************
    *                                                                               *
    *  Control ID (FED_HCTRLID)                                                     *
    *  -------------------------                                                    *
    *  - 4-bit field located at bits 28-31                                          *
    *  - Purpose: Identifies the type of header or control word, distinguishing     *
    *             it from other data within the FED system.                         *
    *                                                                               *
    *********************************************************************************
    """

    const FED_HCTRLID_WIDTH = 0x0000000f
    const FED_HCTRLID_SHIFT = 28
    FED_HCTRLID_EXTRACT(a::UInt32) = (a >> FED_HCTRLID_SHIFT) & FED_HCTRLID_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  Event Type (FED_EVTY)                                                        *
    *  ----------------------                                                       *
    *  - 4-bit field located at bits 24-27                                          *
    *  - Purpose: Categorizes the type of event being recorded, helping in          *
    *             sorting and processing events based on predefined categories.     *
    *                                                                               *
    *********************************************************************************
    """

    const FED_EVTY_WIDTH = 0x0000000f
    const FED_EVTY_SHIFT = 24
    FED_EVTY_EXTRACT(a::UInt32) = (a >> FED_EVTY_SHIFT) & FED_EVTY_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  Level 1 ID (FED_LVL1)                                                        *
    *  ----------------------                                                       *
    *  - 24-bit field located at bits 0-23                                          *
    *  - Purpose: Uniquely identifies an event at the Level 1 trigger stage,        *
    *             critical for quick decision-making in the data acquisition system.*
    *                                                                               *
    *********************************************************************************
    """

    const FED_LVL1_WIDTH = 0x00ffffff
    const FED_LVL1_SHIFT = 0
    FED_LVL1_EXTRACT(a::UInt32) = (a >> FED_LVL1_SHIFT) & FED_LVL1_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  Bunch Crossing ID (BXID) (FED_BXID)                                          *
    *  ----------------------------                                                 *
    *  - 12-bit field located at bits 20-31                                         *
    *  - Purpose: Identifies the bunch crossing instance, syncing data acquisition  *
    *             with particle collision timing in detectors.                      *
    *                                                                               *
    *********************************************************************************
    """

    const FED_BXID_WIDTH = 0x00000fff
    const FED_BXID_SHIFT = 20
    FED_BXID_EXTRACT(a::UInt32) = (a >> FED_BXID_SHIFT) & FED_BXID_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  Source ID (FED_SOID)                                                         *
    *  -------------------                                                          *
    *  - 12-bit field located at bits 8-19                                          *
    *  - Purpose: Identifies the source of data, specifying which detector or       *
    *             sub-detector produced the data.                                   *
    *                                                                               *
    *********************************************************************************
    """

    const FED_SOID_WIDTH = 0x00000fff
    const FED_SOID_SHIFT = 8
    FED_SOID_EXTRACT(a::UInt32) = (a >> FED_SOID_SHIFT) & FED_SOID_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  Version (FED_VERSION)                                                        *
    *  ---------------------                                                        *
    *  - 4-bit field located at bits 4-7                                            *
    *  - Purpose: Specifies the version of data format or protocol, aiding in       *
    *             maintaining compatibility and proper data interpretation.         *
    *                                                                               *
    *********************************************************************************
    """

    const FED_VERSION_WIDTH = 0x0000000f
    const FED_VERSION_SHIFT = 4
    FED_VERSION_EXTRACT(a::UInt32) = (a >> FED_VERSION_SHIFT) & FED_VERSION_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  More Headers (FED_MORE_HEADERS)                                              *
    *  ------------------------------                                               *
    *  - 1-bit field located at bit 3                                               *
    *  - Purpose: Indicates if additional headers follow, essential for handling    *
    *             metadata or control information in data processing.               *
    *                                                                               *
    *********************************************************************************
    """

    const FED_MORE_HEADERS_WIDTH = 0x00000001
    const FED_MORE_HEADERS_SHIFT = 3
    FED_MORE_HEADERS_EXTRACT(a::UInt32) = (a >> FED_MORE_HEADERS_SHIFT) & FED_MORE_HEADERS_WIDTH

    struct FedHeader
        theHeader::Fedh_t
        length::UInt32
    end
    function FedHeader(header::AbstractArray) 
        source_id::UInt32 = reinterpret(UInt32,header[1:4])[1]
        event_id::UInt32 =  reinterpret(UInt32,header[5:8])[1]
        header = Fedh_t(source_id,event_id)
        FedHeader(header,8) # The size of a FEDHeader is 8 bytes
    end
    """
    Functions for extracting fields
    """
    trigger_type(self::FedHeader)::UInt8 = FED_EVTY_EXTRACT(self.theHeader.event_id)
    lvl1_id(self::FedHeader)::UInt32 = FED_LVL1_EXTRACT(self.theHeader.event_id)
    bx_id(self::FedHeader)::UInt16 = FED_BXID_EXTRACT(self.theHeader.source_id)
    source_id(self::FedHeader)::UInt16 = FED_SOID_EXTRACT(self.theHeader.source_id)
    version(self::FedHeader)::Uint8 = FED_VERSION_EXTRACT(self.theHeader.source_id)
    more_headers(self::FedHeader)::Bool = FED_MORE_HEADERS_EXTRACT(self.theHeader.source_id) != 0
    
    """
    Checker to check whether it is a header or not
    """
    check(self::FedHeader)::Bool = FED_HCTRLID_EXTRACT(self.theHeader.event_id) == FED_SLINK_START_MARKER
    
end