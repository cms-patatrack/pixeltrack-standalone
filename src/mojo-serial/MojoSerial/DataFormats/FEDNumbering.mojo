struct FEDNumbering:
    comptime _in = initIn()

    comptime NOT_A_FEDID = -1
    comptime MAXFEDID = 4096  # must be larger than largest used FED id
    comptime MINSiPixelFEDID = 0
    comptime MAXSiPixelFEDID = 40  # increase from 39 for the pilot blade fed
    comptime MINSiStripFEDID = 50
    comptime MAXSiStripFEDID = 489
    comptime MINPreShowerFEDID = 520
    comptime MAXPreShowerFEDID = 575
    comptime MINTotemTriggerFEDID = 577
    comptime MAXTotemTriggerFEDID = 577
    comptime MINTotemRPHorizontalFEDID = 578
    comptime MAXTotemRPHorizontalFEDID = 581
    comptime MINCTPPSDiamondFEDID = 582
    comptime MAXCTPPSDiamondFEDID = 583
    comptime MINTotemRPVerticalFEDID = 584
    comptime MAXTotemRPVerticalFEDID = 585
    comptime MINTotemRPTimingVerticalFEDID = 586
    comptime MAXTotemRPTimingVerticalFEDID = 587
    comptime MINECALFEDID = 600
    comptime MAXECALFEDID = 670
    comptime MINCASTORFEDID = 690
    comptime MAXCASTORFEDID = 693
    comptime MINHCALFEDID = 700
    comptime MAXHCALFEDID = 731
    comptime MINLUMISCALERSFEDID = 735
    comptime MAXLUMISCALERSFEDID = 735
    comptime MINCSCFEDID = 750
    comptime MAXCSCFEDID = 757
    comptime MINCSCTFFEDID = 760
    comptime MAXCSCTFFEDID = 760
    comptime MINDTFEDID = 770
    comptime MAXDTFEDID = 779
    comptime MINDTTFFEDID = 780
    comptime MAXDTTFFEDID = 780
    comptime MINRPCFEDID = 790
    comptime MAXRPCFEDID = 795
    comptime MINTriggerGTPFEDID = 812
    comptime MAXTriggerGTPFEDID = 813
    comptime MINTriggerEGTPFEDID = 814
    comptime MAXTriggerEGTPFEDID = 814
    comptime MINTriggerGCTFEDID = 745
    comptime MAXTriggerGCTFEDID = 749
    comptime MINTriggerLTCFEDID = 816
    comptime MAXTriggerLTCFEDID = 824
    comptime MINTriggerLTCmtccFEDID = 815
    comptime MAXTriggerLTCmtccFEDID = 815
    comptime MINTriggerLTCTriggerFEDID = 816
    comptime MAXTriggerLTCTriggerFEDID = 816
    comptime MINTriggerLTCHCALFEDID = 817
    comptime MAXTriggerLTCHCALFEDID = 817
    comptime MINTriggerLTCSiStripFEDID = 818
    comptime MAXTriggerLTCSiStripFEDID = 818
    comptime MINTriggerLTCECALFEDID = 819
    comptime MAXTriggerLTCECALFEDID = 819
    comptime MINTriggerLTCTotemCastorFEDID = 820
    comptime MAXTriggerLTCTotemCastorFEDID = 820
    comptime MINTriggerLTCRPCFEDID = 821
    comptime MAXTriggerLTCRPCFEDID = 821
    comptime MINTriggerLTCCSCFEDID = 822
    comptime MAXTriggerLTCCSCFEDID = 822
    comptime MINTriggerLTCDTFEDID = 823
    comptime MAXTriggerLTCDTFEDID = 823
    comptime MINTriggerLTCSiPixelFEDID = 824
    comptime MAXTriggerLTCSiPixelFEDID = 824
    comptime MINCSCDDUFEDID = 830
    comptime MAXCSCDDUFEDID = 869
    comptime MINCSCContingencyFEDID = 880
    comptime MAXCSCContingencyFEDID = 887
    comptime MINCSCTFSPFEDID = 890
    comptime MAXCSCTFSPFEDID = 901
    comptime MINDAQeFEDFEDID = 902
    comptime MAXDAQeFEDFEDID = 931
    comptime MINMetaDataSoftFEDID = 1022
    comptime MAXMetaDataSoftFEDID = 1022
    comptime MINDAQmFEDFEDID = 1023
    comptime MAXDAQmFEDFEDID = 1023
    comptime MINTCDSuTCAFEDID = 1024
    comptime MAXTCDSuTCAFEDID = 1099
    comptime MINHCALuTCAFEDID = 1100
    comptime MAXHCALuTCAFEDID = 1199
    comptime MINSiPixeluTCAFEDID = 1200
    comptime MAXSiPixeluTCAFEDID = 1349
    comptime MINRCTFEDID = 1350
    comptime MAXRCTFEDID = 1359
    comptime MINCalTrigUp = 1360
    comptime MAXCalTrigUp = 1367
    comptime MINDTUROSFEDID = 1369
    comptime MAXDTUROSFEDID = 1371
    comptime MINTriggerUpgradeFEDID = 1372
    comptime MAXTriggerUpgradeFEDID = 1409
    comptime MINSiPixel2nduTCAFEDID = 1500
    comptime MAXSiPixel2nduTCAFEDID = 1649
    comptime MINSiPixelTestFEDID = 1450
    comptime MAXSiPixelTestFEDID = 1461
    comptime MINSiPixelAMC13FEDID = 1410
    comptime MAXSiPixelAMC13FEDID = 1449
    comptime MINCTPPSPixelsFEDID = 1462
    comptime MAXCTPPSPixelsFEDID = 1466
    comptime MINGEMFEDID = 1467
    comptime MAXGEMFEDID = 1472
    comptime MINME0FEDID = 1473
    comptime MAXME0FEDID = 1478
    comptime MINDAQvFEDFEDID = 2815
    comptime MAXDAQvFEDFEDID = 4095

    @staticmethod
    @always_inline
    def lastFEDId() -> Int:
        return FEDNumbering.MAXFEDID

    @staticmethod
    @always_inline
    def inRange(var i: Int) -> Bool:
        return (
            FEDNumbering._in[i // 64] >> UInt64(i % 64)
        ) & 1 == 1

    @staticmethod
    @always_inline
    def inRangeNoGT(var i: Int) -> Bool:
        if (
            i >= FEDNumbering.MINTriggerGTPFEDID
            and i <= FEDNumbering.MAXTriggerGTPFEDID
        ) or (
            i >= FEDNumbering.MINTriggerEGTPFEDID
            and i <= FEDNumbering.MAXTriggerEGTPFEDID
        ):
            return False
        return FEDNumbering.inRange(i)


# C++ memoizes this as `static bool in_[MAXFEDID+1]`. Mojo has no global `var`
# ("global variables are not supported; use 'comptime'"), and a `comptime`
# container read at a runtime index needs `materialize`, which would rebuild the
# whole table on every call. A `comptime` SIMD *does* index at a runtime index
# with no ceremony (doc §12), so the table is one: 4097 flags as a bitmask,
# 8192 bits in 128 UInt64 lanes.
def initIn() -> SIMD[DType.uint64, 128]:
    var _in = SIMD[DType.uint64, 128](0)

    comptime for i in range(
        FEDNumbering.MINSiPixelFEDID, FEDNumbering.MAXSiPixelFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINSiStripFEDID, FEDNumbering.MAXSiStripFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINPreShowerFEDID, FEDNumbering.MAXPreShowerFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(FEDNumbering.MINECALFEDID, FEDNumbering.MAXECALFEDID + 1):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINCASTORFEDID, FEDNumbering.MAXCASTORFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(FEDNumbering.MINHCALFEDID, FEDNumbering.MAXHCALFEDID + 1):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINLUMISCALERSFEDID, FEDNumbering.MAXLUMISCALERSFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(FEDNumbering.MINCSCFEDID, FEDNumbering.MAXCSCFEDID + 1):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(FEDNumbering.MINCSCTFFEDID, FEDNumbering.MAXCSCTFFEDID + 1):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(FEDNumbering.MINDTFEDID, FEDNumbering.MAXDTFEDID + 1):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(FEDNumbering.MINDTTFFEDID, FEDNumbering.MAXDTTFFEDID + 1):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(FEDNumbering.MINRPCFEDID, FEDNumbering.MAXRPCFEDID + 1):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINTriggerGTPFEDID, FEDNumbering.MAXTriggerGTPFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINTriggerEGTPFEDID, FEDNumbering.MAXTriggerEGTPFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINTriggerGCTFEDID, FEDNumbering.MAXTriggerGCTFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINTriggerLTCFEDID, FEDNumbering.MAXTriggerLTCFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINTriggerLTCmtccFEDID,
        FEDNumbering.MAXTriggerLTCmtccFEDID + 1,
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINCSCDDUFEDID, FEDNumbering.MAXCSCDDUFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINCSCContingencyFEDID,
        FEDNumbering.MAXCSCContingencyFEDID + 1,
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINCSCTFSPFEDID, FEDNumbering.MAXCSCTFSPFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINDAQeFEDFEDID, FEDNumbering.MAXDAQeFEDFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINDAQmFEDFEDID, FEDNumbering.MAXDAQmFEDFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINTCDSuTCAFEDID, FEDNumbering.MAXTCDSuTCAFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINHCALuTCAFEDID, FEDNumbering.MAXHCALuTCAFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINSiPixeluTCAFEDID, FEDNumbering.MAXSiPixeluTCAFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINDTUROSFEDID, FEDNumbering.MAXDTUROSFEDID + 1
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    comptime for i in range(
        FEDNumbering.MINTriggerUpgradeFEDID,
        FEDNumbering.MAXTriggerUpgradeFEDID + 1,
    ):
        _in[i // 64] |= UInt64(1) << UInt64(i % 64)

    return _in
