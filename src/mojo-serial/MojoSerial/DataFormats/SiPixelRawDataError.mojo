from MojoSerial.MojoBridge.DTypes import Typeable


struct SiPixelRawDataError(
    Comparable, Copyable, Defaultable, Movable, Typeable
):
    """
    Pixel error -- collection of errors and error information
    Class to contain and store all information about errors.
    """

    # the 32-bit word that contains the error information
    var _errorWord32: UInt32
    # the 64-bit word that contains the error information
    var _errorWord64: UInt64
    # the number associated with the error type (26-31 for ROC number errors, 32-33 for calibration errors)
    var _errorType: Int32
    # the fedId where the error occurred
    var _fedId: Int32
    # the error message to be displayed with the error
    var _errorMessage: String

    # default constructor
    def __init__(out self):
        self._errorWord32 = 0
        self._errorWord64 = 0
        self._errorType = 0
        self._fedId = 0
        self._errorMessage = ""

    # constructor for 32-bit error word
    def __init__(
        out self,
        var errorWord32: UInt32,
        var errorType: Int32,
        var fedId: Int32,
    ):
        # Mojo currently does not infer setting constructor fields outside of the constructor
        self = Self()

        self._errorWord32 = errorWord32
        self._errorType = errorType
        self._fedId = fedId

        self.setMessage()

    # constructor for 64-bit error word and type included (header or trailer word)
    def __init__(
        out self,
        var errorWord64: UInt64,
        var errorType: Int32,
        var fedId: Int32,
    ):
        # Mojo currently does not infer setting constructor fields outside of the constructor
        self = Self()

        self._errorWord64 = errorWord64
        self._errorType = errorType
        self._fedId = fedId

        self.setMessage()

    def __init__(out self, *, deinit move: Self):
        self._errorWord32 = move._errorWord32
        self._errorWord64 = move._errorWord64
        self._errorType = move._errorType
        self._fedId = move._fedId
        self._errorMessage = move._errorMessage^

    def __init__(out self, *, copy: Self):
        self._errorWord32 = copy._errorWord32
        self._errorWord64 = copy._errorWord64
        self._errorType = copy._errorType
        self._fedId = copy._fedId
        self._errorMessage = copy._errorMessage

    def setWord32(mut self, var errorWord32: UInt32):
        self._errorWord32 = errorWord32

    def setWord64(mut self, var errorWord64: UInt64):
        self._errorWord64 = errorWord64

    def setType(mut self, var errorType: Int32):
        self._errorType = errorType
        self.setMessage()

    def setFedId(mut self, var fedId: Int32):
        self._fedId = fedId

    @always_inline
    def getWord32(self) -> UInt32:
        return self._errorWord32

    @always_inline
    def getWord64(self) -> UInt64:
        return self._errorWord64

    @always_inline
    def getType(self) -> Int32:
        return self._errorType

    @always_inline
    def getFedId(self) -> Int32:
        return self._fedId

    @always_inline
    def getMessage(self) -> String:
        return self._errorMessage

    def setMessage(mut self):
        if self._errorType == 25:
            self._errorMessage = "Error: Disabled FED channel (ROC=25)"
        elif self._errorType == 26:
            self._errorMessage = "Error: Gap word"
        elif self._errorType == 27:
            self._errorMessage = "Error: Dummy word"
        elif self._errorType == 28:
            self._errorMessage = "Error: FIFO nearly full"
        elif self._errorType == 29:
            self._errorMessage = "Error: Timeout"
        elif self._errorType == 30:
            self._errorMessage = "Error: Trailer"
        elif self._errorType == 31:
            self._errorMessage = "Error: Event number mismatch"
        elif self._errorType == 32:
            self._errorMessage = "Error: Invalid or missing header"
        elif self._errorType == 33:
            self._errorMessage = "Error: Invalid or missing trailer"
        elif self._errorType == 34:
            self._errorMessage = "Error: Size mismatch"
        elif self._errorType == 35:
            self._errorMessage = "Error: Invalid channel"
        elif self._errorType == 36:
            self._errorMessage = "Error: Invalid ROC number"
        elif self._errorType == 37:
            self._errorMessage = "Error: Invalid dcol/pixel address"
        else:
            self._errorMessage = "Error: Unknown error type"

    @always_inline
    def __eq__(self, rhs: Self) -> Bool:
        return self._fedId == rhs._fedId

    @always_inline
    def __ne__(self, rhs: Self) -> Bool:
        return self._fedId != rhs._fedId

    @always_inline
    def __lt__(self, rhs: Self) -> Bool:
        return self._fedId < rhs._fedId

    @always_inline
    def __le__(self, rhs: Self) -> Bool:
        return self._fedId <= rhs._fedId

    @always_inline
    def __gt__(self, rhs: Self) -> Bool:
        return self._fedId > rhs._fedId

    @always_inline
    def __ge__(self, rhs: Self) -> Bool:
        return self._fedId >= rhs._fedId

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelRawDataError"
