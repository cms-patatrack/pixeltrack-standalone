from std.sys import size_of

from MojoSerial.MojoBridge.DTypes import UChar
struct PixelGPUDetails:
    comptime layerStartBit: UInt32 = 20
    comptime ladderStartBit: UInt32 = 12
    comptime moduleStartBit: UInt32 = 2

    comptime panelStartBit: UInt32 = 10
    comptime diskStartBit: UInt32 = 18
    comptime bladeStartBit: UInt32 = 12

    comptime layerMask: UInt32 = 0xF
    comptime ladderMask: UInt32 = 0xFF
    comptime moduleMask: UInt32 = 0x3FF
    comptime panelMask: UInt32 = 0x3
    comptime diskMask: UInt32 = 0xF
    comptime bladeMask: UInt32 = 0x3F

    comptime LINK_bits: UInt32 = 6
    comptime ROC_bits: UInt32 = 5
    comptime DCOL_bits: UInt32 = 5
    comptime PXID_bits: UInt32 = 8
    comptime ADC_bits: UInt32 = 8

    # special for layer 1
    comptime LINK_bits_l1: UInt32 = 6
    comptime ROC_bits_l1: UInt32 = 5
    comptime COL_bits_l1: UInt32 = 6
    comptime ROW_bits_l1: UInt32 = 7
    comptime OMIT_ERR_bits: UInt32 = 1

    comptime maxROCIndex: UInt32 = 8
    comptime numRowsInRoc: UInt32 = 80
    comptime numColsInRoc: UInt32 = 52

    comptime MAX_WORD: UInt32 = 2000

    comptime ADC_shift: UInt32 = 0
    comptime PXID_shift: UInt32 = Self.ADC_shift + Self.ADC_bits
    comptime DCOL_shift: UInt32 = Self.PXID_shift + Self.PXID_bits
    comptime ROC_shift: UInt32 = Self.DCOL_shift + Self.DCOL_bits
    comptime LINK_shift: UInt32 = Self.ROC_shift + Self.ROC_bits_l1

    # special for layer 1 ROC
    comptime ROW_shift: UInt32 = Self.ADC_shift + Self.ADC_bits
    comptime COL_shift: UInt32 = Self.ROW_shift + Self.ROW_bits_l1
    comptime OMIT_ERR_shift: UInt32 = 20

    comptime LINK_mask: UInt32 = ~(~UInt32(0) << Self.LINK_bits_l1)
    comptime ROC_mask: UInt32 = ~(~UInt32(0) << Self.ROC_bits_l1)
    comptime COL_mask: UInt32 = ~(~UInt32(0) << Self.COL_bits_l1)
    comptime ROW_mask: UInt32 = ~(~UInt32(0) << Self.ROW_bits_l1)
    comptime DCOL_mask: UInt32 = ~(~UInt32(0) << Self.DCOL_bits)
    comptime PXID_mask: UInt32 = ~(~UInt32(0) << Self.PXID_bits)
    comptime ADC_mask: UInt32 = ~(~UInt32(0) << Self.ADC_bits)
    comptime ERROR_mask: UInt32 = ~(~UInt32(0) << Self.ROC_bits_l1)
    comptime OMIT_ERR_mask: UInt32 = ~(~UInt32(0) << Self.OMIT_ERR_bits)

    # Maximum fed for phase1 is 150 but not all of them are filled
    # Update the number FED based on maximum fed found in the cabling map
    comptime MAX_FED: UInt32 = 150
    comptime MAX_LINK: UInt32 = 48  # maximum links/channels for Phase 1
    comptime MAX_ROC: UInt32 = 8
    comptime MAX_SIZE = Self.MAX_FED * Self.MAX_LINK * Self.MAX_ROC
    comptime MAX_SIZE_BYTE_BOOL = Self.MAX_SIZE * UInt32(size_of[UChar]())
    # number of words for all the FEDs
    comptime MAX_FED_WORDS = Self.MAX_FED * Self.MAX_WORD
