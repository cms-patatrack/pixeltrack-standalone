from MojoSerial.CUDACore.SimpleVector import SimpleVector, make_SimpleVector
from MojoSerial.DataFormats.PixelErrors import (
    PixelErrorCompact,
    PixelFormatterErrors,
)
from MojoSerial.MojoBridge.DTypes import SizeType, Typeable


struct SiPixelDigiErrorsSoA(Defaultable, Movable, Typeable):
    comptime _error_dtype = SimpleVector[
        PixelErrorCompact, PixelErrorCompact.dtype()
    ]
    # C++ keeps the buffer in a separate `data_d` and points `error_d` into it;
    # SimpleVector owns its storage here, so there is no `data_d`.
    var error_d: Self._error_dtype
    var formatterErrors_h: PixelFormatterErrors

    @always_inline
    def __init__(out self):
        self.error_d = Self._error_dtype()
        self.formatterErrors_h = PixelFormatterErrors()

    @always_inline
    def __init__(
        out self, maxFedWords: SizeType, var errors: PixelFormatterErrors
    ):
        self.formatterErrors_h = errors^
        self.error_d = make_SimpleVector[
            PixelErrorCompact, PixelErrorCompact.dtype()
        ](maxFedWords.cast[DType.int32]())
        debug_assert(self.error_d.empty())
        debug_assert(self.error_d.capacity() == maxFedWords.cast[DType.int32]())

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self.error_d = move.error_d^
        self.formatterErrors_h = move.formatterErrors_h^

    def formatterErrors(
        self,
    ) -> ref [self.formatterErrors_h] PixelFormatterErrors:
        return self.formatterErrors_h

    # C++ returns SimpleVector<PixelErrorCompact>*; a borrow says the same thing
    # without a raw pointer. Neither accessor currently has a caller.
    def error(ref self) -> ref [self.error_d] Self._error_dtype:
        return self.error_d

    def c_error(self) -> ref [self.error_d] Self._error_dtype:
        return self.error_d

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "SiPixelDigiErrorsSoA"
