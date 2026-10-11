from MojoSerial.DataFormats.SiPixelRawDataError import SiPixelRawDataError
from MojoSerial.DataFormats.PixelErrors import PixelFormatterErrors
from MojoSerial.DataFormats.FEDHeader import FEDHeader, FedhType
from MojoSerial.DataFormats.FEDTrailer import FEDTrailer, FedtType
from MojoSerial.MojoBridge.DTypes import Typeable


@fieldwise_init
struct ErrorChecker(Copyable, Defaultable, Movable, Typeable):
    comptime Word32 = UInt32
    comptime Word64 = UInt64
    comptime DetErrors = PixelFormatterErrors.V
    comptime Errors = PixelFormatterErrors

    comptime CRC_bits = 1
    comptime LINK_bits = 6
    comptime ROC_bits = 5
    comptime DCOL_bits = 5
    comptime PXID_bits = 8
    comptime ADC_bits = 8
    comptime OMIT_ERR_bits = 1

    comptime CRC_shift = 2
    comptime ADC_shift = 0
    comptime PXID_shift = Self.ADC_shift + Self.ADC_bits
    comptime DCOL_shift = Self.PXID_shift + Self.PXID_bits
    comptime ROC_shift = Self.DCOL_shift + Self.DCOL_bits
    comptime LINK_shift = Self.ROC_shift + Self.ROC_bits
    comptime OMIT_ERR_shift = 20

    comptime dummyDetId: UInt32 = 0xFFFFFFFF

    comptime CRC_mask: Self.Word64 = ~(~Self.Word64(0) << Self.CRC_bits)
    comptime ERROR_mask: Self.Word32 = ~(~Self.Word32(0) << Self.ROC_bits)
    comptime LINK_mask: Self.Word32 = ~(~Self.Word32(0) << Self.LINK_bits)
    comptime ROC_mask: Self.Word32 = ~(~Self.Word32(0) << Self.ROC_bits)
    comptime OMIT_ERR_mask: Self.Word32 = ~(~Self.Word32(0) << Self.OMIT_ERR_bits)

    # a flag to include errors in the output
    # if set, errors will be added to the errors dict
    var includeErrors: Bool

    @always_inline
    def __init__(out self):
        self.includeErrors = False

    def checkCRC(
        self,
        mut errorsInEvent: Bool,
        var fedId: Int32,
        trailer: Pointer[Self.Word64, _],
        mut errors: Self.Errors,
    ) -> Bool:
        var CRC_BIT: Int32 = (
            (trailer[] >> Self.CRC_shift) & Self.CRC_mask
        ).cast[DType.int32]()
        if CRC_BIT == 0:
            return True
        errorsInEvent = True
        if self.includeErrors:
            comptime errorType = 39
            var error = SiPixelRawDataError(trailer[], errorType, fedId)
            try:
                errors[UInt(Self.dummyDetId)].append(error^)
            except e:
                print("Handled an exception in ErrorChecker,", e)
        return False

    def checkHeader(
        self,
        mut errorsInEvent: Bool,
        var fedId: Int32,
        header: Pointer[Self.Word64, _],
        mut errors: Self.Errors,
    ) -> Bool:
        var fedHeader = FEDHeader(header.unsafe_bitcast[FedhType]()[])
        if not fedHeader.check():
            return False
        if fedHeader.sourceID().cast[DType.int32]() != fedId:
            print(
                (
                    "PixelDataFormatter::interpretRawData, fedHeader.sourceID()"
                    " != fedId"
                ),
                ", sourceID = ",
                fedHeader.sourceID(),
                ", fedId = ",
                fedId,
                ", errorType = 32",
                sep="",
            )
            errorsInEvent = True
            if self.includeErrors:
                comptime errorType = 32
                var error = SiPixelRawDataError(header[], errorType, fedId)
                try:
                    errors[UInt(Self.dummyDetId)].append(error^)
                except e:
                    print("Handled an exception in ErrorChecker,", e)
        return fedHeader.moreHeaders()

    def checkTrailer(
        self,
        mut errorsInEvent: Bool,
        fedId: Int32,
        nWords: UInt32,
        trailer: Pointer[Self.Word64, _],
        mut errors: Self.Errors,
    ) -> Bool:
        var fedTrailer = FEDTrailer(trailer.unsafe_bitcast[FedtType]()[])
        if not fedTrailer.check():
            if self.includeErrors:
                comptime errorType = 33
                var error = SiPixelRawDataError(trailer[], errorType, fedId)
                try:
                    errors[UInt(Self.dummyDetId)].append(error^)
                except e:
                    print("Handled an exception in ErrorChecker,", e)
            errorsInEvent = True
            print(
                "fedTrailer.check failed, Fed: ",
                fedId,
                ", errorType = 33",
                sep="",
            )
            return False
        if fedTrailer.fragmentLength() != nWords:
            print(
                "fedTrailer.fragmentLength()!= nWords !! Fed: ",
                fedId,
                ", errorType = 34",
                sep="",
            )
            errorsInEvent = True
            if self.includeErrors:
                comptime errorType = 34
                var error = SiPixelRawDataError(trailer[], errorType, fedId)
                try:
                    errors[UInt(Self.dummyDetId)].append(error^)
                except e:
                    print("Handled an exception in ErrorChecker,", e)
        return fedTrailer.moreTrailers()

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "ErrorChecker"
