"""
A module for handling FED trailers in a typical FED system.

# FED System Overview
In a typical FED system, each block of data ends with a trailer containing metadata. The fields are extracted from these trailers to interpret data correctly, ensure synchronization, and manage data flow efficiently across the system.

# Constants and Extraction Functions
This module provides constants and functions to extract various fields from the FED trailer.
"""
module fedTrailer



    export FED_SLINK_END_MARKER, FED_SLINK_ERROR_WIDTH, FED_TCTRLID_EXTRACT, FED_EVSZ_EXTRACT, FED_CRCS_EXTRACT, FED_STAT_EXTRACT, FED_TTSI_EXTRACT, FED_MORE_TRAILERS_EXTRACT, FED_CRC_MODIFIED_EXTRACT, FED_SLINK_ERROR_EXTRACT, FED_WRONG_FEDID_EXTRACT, FedTrailer, check, fragment_length, more_trailers


    struct Fedt_t
        cons_check::UInt32 # Consistensy Check to ensure data meets expected standards and rules
        event_size::UInt32 # Size of event data blocks
    end
    """
    event_size
    [ Control ID | length     |          Event Size           ]
    [  4 bits    |  4 bits    |           24 bits             ]

    cons_check
    [  CRC   | Wrong FED ID | S-Link Error |   Reserved   |    Status     |   TTS Info        | More Trailers | CRC Modified | Reserved]
    [16 bits |    1 bit     |   1 bit      |   2 bits      |    4 bits     |   4 bits          |  1 bits       |    1 bit    |  2 bits ]
    """

    const FED_SLINK_END_MARKER = 0xa # A const to indicate the end of a data frame or block in the FED system. Compared with TCTRLID
    
    

    """
    *********************************************************************************
    *                                                                               *
    *  Control ID (FED_TCTRLID)                                                     *
    *  -------------------------                                                    *
    *  - 4-bit field located at bits 28-31                                          *
    *  - Purpose: Identifies the type of trailer or control word, distinguishing    *
    *             it from other data within the FED system.                         *
    *                                                                               *
    *********************************************************************************
    """

    const FED_TCTRLID_WIDTH = 0x0000000f
    const FED_TCTRLID_SHIFT = 28
    FED_TCTRLID_EXTRACT(a::UInt32) = (a >> FED_TCTRLID_SHIFT) & FED_TCTRLID_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  Event Size (FED_EVSZ)                                                        *
    *  ----------------------                                                       *
    *  - 24-bit field located at bits 0-23                                          *
    *  - Purpose: Indicates the size of the event data block, aiding in proper      *
    *             allocation and processing of data packets in the FED system.      *
    *                                                                               *
    *********************************************************************************
    """

    const FED_EVSZ_WIDTH = 0x00ffffff
    const FED_EVSZ_SHIFT = 0
    FED_EVSZ_EXTRACT(a::UInt32) = (a >> FED_EVSZ_SHIFT) & FED_EVSZ_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  CRC (FED_CRCS)                                                               *
    *  -----------------                                                            *
    *  - 16-bit field located at bits 16-31                                         *
    *  - Purpose: Provides a cyclic redundancy check value to verify the integrity  *
    *             of the event data block during transmission.                      *
    *                                                                               *
    *********************************************************************************
    """

    const FED_CRCS_WIDTH = 0x0000ffff
    const FED_CRCS_SHIFT = 16
    FED_CRCS_EXTRACT(a::UInt32) = (a >> FED_CRCS_SHIFT) & FED_CRCS_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  Status (FED_STAT)                                                            *
    *  -----------------                                                            *
    *  - 4-bit field located at bits 8-11                                           *
    *  - Purpose: Provides status information about the event data block, aiding in *
    *             handling and processing of data within the FED system.            *
    *                                                                               *
    *********************************************************************************
    """

    const FED_STAT_WIDTH = 0x0000000f
    const FED_STAT_SHIFT = 8
    FED_STAT_EXTRACT(a::UInt32) = (a >> FED_STAT_SHIFT) & FED_STAT_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  TTS Info (FED_TTSI)                                                          *
    *  -------------------                                                          *
    *  - 4-bit field located at bits 4-7                                            *
    *  - Purpose: Contains information related to the Trigger Throttling System,    *
    *             aiding in event prioritization and management within the FED      *
    *             system.                                                           *
    *                                                                               *
    *********************************************************************************
    """

    const FED_TTSI_WIDTH = 0x0000000f
    const FED_TTSI_SHIFT = 4
    FED_TTSI_EXTRACT(a::UInt32) = (a >> FED_TTSI_SHIFT) & FED_TTSI_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  More Trailers (FED_MORE_TRAILERS)                                            *
    *  ------------------------------                                               *
    *  - 1-bit field located at bit 3                                               *
    *  - Purpose: Indicates if additional trailers follow, essential for handling   *
    *             metadata or control information in data processing.               *
    *                                                                               *
    *********************************************************************************
    """

    const FED_MORE_TRAILERS_WIDTH = 0x00000001
    const FED_MORE_TRAILERS_SHIFT = 3
    FED_MORE_TRAILERS_EXTRACT(a::UInt32) = (a >> FED_MORE_TRAILERS_SHIFT) & FED_MORE_TRAILERS_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  CRC Modified (FED_CRC_MODIFIED)                                              *
    *  ---------------------------------                                            *
    *  - 1-bit field located at bit 2                                               *
    *  - Purpose: Indicates if the CRC value has been modified by the S-link sender *
    *             card during data transmission.                                    *
    *                                                                               *
    *********************************************************************************
    """

    const FED_CRC_MODIFIED_WIDTH = 0x00000001
    const FED_CRC_MODIFIED_SHIFT = 2
    FED_CRC_MODIFIED_EXTRACT(a::UInt32) = (a >> FED_CRC_MODIFIED_SHIFT) & FED_CRC_MODIFIED_WIDTH

    """
    *********************************************************************************
    *                                                                               *
    *  S-Link Error (FED_SLINK_ERROR)                                               *
    *  ----------------------------                                                 *
    *  - 1-bit field located at bit 14                                              *
    *  - Purpose: Indicates if the FRL has detected a transmission error over the   *
    *             S-link cable.                                                     *
    *                                                                               *
    *********************************************************************************
    """

    const FED_SLINK_ERROR_WIDTH = 0x00000001
    const FED_SLINK_ERROR_SHIFT = 14
    FED_SLINK_ERROR_EXTRACT(a::UInt32) = (a >> FED_SLINK_ERROR_SHIFT) & FED_SLINK_ERROR_WIDTH

 `   """
    *********************************************************************************
    *                                                                               *
    *  Wrong FED ID (FED_WRONG_FEDID)                                               *
    *  ------------------------------                                               *
    *  - 1-bit field located at bit 15                                              *
    *  - Purpose: Indicates if the FED_ID given by the FED is not the one expected  *
    *             by the FRL.                                                       *
    *                                                                               *
    *********************************************************************************
    """
`
    const FED_WRONG_FEDID_WIDTH = 0x00000001
    const FED_WRONG_FEDID_SHIFT = 15
    FED_WRONG_FEDID_EXTRACT(a::UInt32) = (a >> FED_WRONG_FEDID_SHIFT) & FED_WRONG_FEDID_WIDTH

    struct FedTrailer
        theTrailer::Fedt_t
        length::UInt32
    end
    function FedTrailer(trailer::AbstractArray)
        cons_check::UInt32 = reinterpret(UInt32, trailer[1:4])[1]
        event_size::UInt32 = reinterpret(UInt32, trailer[5:8])[1]
        trailer_t = Fedt_t(cons_check,event_size)
        FedTrailer(trailer_t,8)
    end
    """
    Functions for extracting fields
    """
    fragment_length(self::FedTrailer)::UInt32 = FED_EVSZ_EXTRACT(self.theTrailer.event_size)

    crc(self::FedTrailer)::UInt16 = FED_CRCS_EXTRACT(self.theTrailer.cons_check)

    evt_status(self::FedTrailer)::UInt8 = FED_STAT_EXTRACT(self.theTrailer.cons_check)

    tts_bits(self::FedTrailer)::UInt8 = FED_TTSI_EXTRACT(self.theTrailer.cons_check)

    more_trailers(self::FedTrailer)::Bool = (FED_MORE_TRAILERS_EXTRACT(self.theTrailer.cons_check) != 0)

    crc_modified(self::FedTrailer)::Bool = FED_CRC_MODIFIED_EXTRACT(self.theTrailer.cons_check) != 0

    slink_error(self::FedTrailer)::Bool = FED_SLINK_ERROR_EXTRACT(self.theTrailer.cons_check) != 0

    wrong_fedid(self::FedTrailer)::Bool = FED_WRONG_FEDID_EXTRACT(self.theTrailer.cons_check) != 0

    cons_check(self::FedTrailer)::UInt32 = self.theTrailer.cons_check

    """
    Checker to check whether it is a trailer or not
    """
    check(self::FedTrailer)::Bool = FED_TCTRLID_EXTRACT(self.theTrailer.event_size) == FED_SLINK_END_MARKER
end # module
