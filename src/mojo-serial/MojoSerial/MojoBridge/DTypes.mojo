from std.memory import bitcast, OwnedPointer

comptime SizeType = UInt32  # size_t
comptime Short = Int16  # short
comptime Float = Float32  # float
comptime Double = Float64  # double
comptime Char = Int8  # char
comptime UChar = UInt8  # unsigned char

# CUDA Driver API result type (mirrors CUresult enum values from CUDA headers).
comptime CUresult = Int32

comptime CUDA_SUCCESS: CUresult = 0
comptime CUDA_ERROR_INVALID_VALUE: CUresult = 1
comptime CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2
comptime CUDA_ERROR_NOT_INITIALIZED: CUresult = 3
comptime CUDA_ERROR_DEINITIALIZED: CUresult = 4
comptime CUDA_ERROR_PROFILER_DISABLED: CUresult = 5
comptime CUDA_ERROR_PROFILER_NOT_INITIALIZED: CUresult = 6
comptime CUDA_ERROR_PROFILER_ALREADY_STARTED: CUresult = 7
comptime CUDA_ERROR_PROFILER_ALREADY_STOPPED: CUresult = 8
comptime CUDA_ERROR_STUB_LIBRARY: CUresult = 34
comptime CUDA_ERROR_DEVICE_UNAVAILABLE: CUresult = 46
comptime CUDA_ERROR_NO_DEVICE: CUresult = 100
comptime CUDA_ERROR_INVALID_DEVICE: CUresult = 101
comptime CUDA_ERROR_DEVICE_NOT_LICENSED: CUresult = 102
comptime CUDA_ERROR_INVALID_IMAGE: CUresult = 200
comptime CUDA_ERROR_INVALID_CONTEXT: CUresult = 201
comptime CUDA_ERROR_CONTEXT_ALREADY_CURRENT: CUresult = 202
comptime CUDA_ERROR_MAP_FAILED: CUresult = 205
comptime CUDA_ERROR_UNMAP_FAILED: CUresult = 206
comptime CUDA_ERROR_ARRAY_IS_MAPPED: CUresult = 207
comptime CUDA_ERROR_ALREADY_MAPPED: CUresult = 208
comptime CUDA_ERROR_NO_BINARY_FOR_GPU: CUresult = 209
comptime CUDA_ERROR_ALREADY_ACQUIRED: CUresult = 210
comptime CUDA_ERROR_NOT_MAPPED: CUresult = 211
comptime CUDA_ERROR_NOT_MAPPED_AS_ARRAY: CUresult = 212
comptime CUDA_ERROR_NOT_MAPPED_AS_POINTER: CUresult = 213
comptime CUDA_ERROR_ECC_UNCORRECTABLE: CUresult = 214
comptime CUDA_ERROR_UNSUPPORTED_LIMIT: CUresult = 215
comptime CUDA_ERROR_CONTEXT_ALREADY_IN_USE: CUresult = 216
comptime CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: CUresult = 217
comptime CUDA_ERROR_INVALID_PTX: CUresult = 218
comptime CUDA_ERROR_INVALID_GRAPHICS_CONTEXT: CUresult = 219
comptime CUDA_ERROR_NVLINK_UNCORRECTABLE: CUresult = 220
comptime CUDA_ERROR_JIT_COMPILER_NOT_FOUND: CUresult = 221
comptime CUDA_ERROR_UNSUPPORTED_PTX_VERSION: CUresult = 222
comptime CUDA_ERROR_JIT_COMPILATION_DISABLED: CUresult = 223
comptime CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY: CUresult = 224
comptime CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC: CUresult = 225
comptime CUDA_ERROR_INVALID_SOURCE: CUresult = 300
comptime CUDA_ERROR_FILE_NOT_FOUND: CUresult = 301
comptime CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: CUresult = 302
comptime CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: CUresult = 303
comptime CUDA_ERROR_OPERATING_SYSTEM: CUresult = 304
comptime CUDA_ERROR_INVALID_HANDLE: CUresult = 400
comptime CUDA_ERROR_ILLEGAL_STATE: CUresult = 401
comptime CUDA_ERROR_LOSSY_QUERY: CUresult = 402
comptime CUDA_ERROR_NOT_FOUND: CUresult = 500
comptime CUDA_ERROR_NOT_READY: CUresult = 600
comptime CUDA_ERROR_ILLEGAL_ADDRESS: CUresult = 700
comptime CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: CUresult = 701
comptime CUDA_ERROR_LAUNCH_TIMEOUT: CUresult = 702
comptime CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: CUresult = 703
comptime CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: CUresult = 704
comptime CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: CUresult = 705
comptime CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: CUresult = 708
comptime CUDA_ERROR_CONTEXT_IS_DESTROYED: CUresult = 709
comptime CUDA_ERROR_ASSERT: CUresult = 710
comptime CUDA_ERROR_TOO_MANY_PEERS: CUresult = 711
comptime CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: CUresult = 712
comptime CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: CUresult = 713
comptime CUDA_ERROR_HARDWARE_STACK_ERROR: CUresult = 714
comptime CUDA_ERROR_ILLEGAL_INSTRUCTION: CUresult = 715
comptime CUDA_ERROR_MISALIGNED_ADDRESS: CUresult = 716
comptime CUDA_ERROR_INVALID_ADDRESS_SPACE: CUresult = 717
comptime CUDA_ERROR_INVALID_PC: CUresult = 718
comptime CUDA_ERROR_LAUNCH_FAILED: CUresult = 719
comptime CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE: CUresult = 720
comptime CUDA_ERROR_NOT_PERMITTED: CUresult = 800
comptime CUDA_ERROR_NOT_SUPPORTED: CUresult = 801
comptime CUDA_ERROR_SYSTEM_NOT_READY: CUresult = 802
comptime CUDA_ERROR_SYSTEM_DRIVER_MISMATCH: CUresult = 803
comptime CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: CUresult = 804
comptime CUDA_ERROR_MPS_CONNECTION_FAILED: CUresult = 805
comptime CUDA_ERROR_MPS_RPC_FAILURE: CUresult = 806
comptime CUDA_ERROR_MPS_SERVER_NOT_READY: CUresult = 807
comptime CUDA_ERROR_MPS_MAX_CLIENTS_REACHED: CUresult = 808
comptime CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED: CUresult = 809
comptime CUDA_ERROR_MPS_CLIENT_TERMINATED: CUresult = 810
comptime CUDA_ERROR_CDP_NOT_SUPPORTED: CUresult = 811
comptime CUDA_ERROR_CDP_VERSION_MISMATCH: CUresult = 812
comptime CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED: CUresult = 900
comptime CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: CUresult = 901
comptime CUDA_ERROR_STREAM_CAPTURE_MERGE: CUresult = 902
comptime CUDA_ERROR_STREAM_CAPTURE_UNMATCHED: CUresult = 903
comptime CUDA_ERROR_STREAM_CAPTURE_UNJOINED: CUresult = 904
comptime CUDA_ERROR_STREAM_CAPTURE_ISOLATION: CUresult = 905
comptime CUDA_ERROR_STREAM_CAPTURE_IMPLICIT: CUresult = 906
comptime CUDA_ERROR_CAPTURED_EVENT: CUresult = 907
comptime CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD: CUresult = 908
comptime CUDA_ERROR_TIMEOUT: CUresult = 909
comptime CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE: CUresult = 910
comptime CUDA_ERROR_EXTERNAL_DEVICE: CUresult = 911
comptime CUDA_ERROR_INVALID_CLUSTER_SIZE: CUresult = 912
comptime CUDA_ERROR_FUNCTION_NOT_LOADED: CUresult = 913
comptime CUDA_ERROR_INVALID_RESOURCE_TYPE: CUresult = 914
comptime CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION: CUresult = 915
comptime CUDA_ERROR_UNKNOWN: CUresult = 999
comptime CUDA_ERROR_INVALID_PTX_JIT_INPUT: CUresult = 10000
comptime CUDA_ERROR_INVALID_PTX_JIT_OPTION: CUresult = 10001
comptime CUDA_ERROR_INVALID_PTX_JIT_LOG: CUresult = 10002
comptime CUDA_ERROR_PTX_JIT_COMPILER_NOT_FOUND: CUresult = 10003
comptime CUDA_ERROR_UNSUPPORTED_PTX_JIT_VERSION: CUresult = 10004
comptime CUDA_ERROR_PTX_JIT_COMPILATION_DISABLED: CUresult = 10005
comptime CUDA_ERROR_PTX_JIT_COMPILER_UNSUPPORTED_OPERATION: CUresult = 10006


def cuResultName(result: CUresult) -> String:
    if result == CUDA_SUCCESS:
        return "CUDA_SUCCESS"
    elif result == CUDA_ERROR_INVALID_VALUE:
        return "CUDA_ERROR_INVALID_VALUE"
    elif result == CUDA_ERROR_OUT_OF_MEMORY:
        return "CUDA_ERROR_OUT_OF_MEMORY"
    elif result == CUDA_ERROR_NOT_INITIALIZED:
        return "CUDA_ERROR_NOT_INITIALIZED"
    elif result == CUDA_ERROR_DEINITIALIZED:
        return "CUDA_ERROR_DEINITIALIZED"
    elif result == CUDA_ERROR_NO_DEVICE:
        return "CUDA_ERROR_NO_DEVICE"
    elif result == CUDA_ERROR_INVALID_DEVICE:
        return "CUDA_ERROR_INVALID_DEVICE"
    elif result == CUDA_ERROR_NOT_SUPPORTED:
        return "CUDA_ERROR_NOT_SUPPORTED"
    elif result == CUDA_ERROR_UNKNOWN:
        return "CUDA_ERROR_UNKNOWN"
    return "CUresult(" + String(result) + ")"


def cuResultMessage(result: CUresult) -> String:
    if result == CUDA_SUCCESS:
        return "success"
    elif result == CUDA_ERROR_INVALID_VALUE:
        return "invalid value"
    elif result == CUDA_ERROR_OUT_OF_MEMORY:
        return "out of memory"
    elif result == CUDA_ERROR_NOT_INITIALIZED:
        return "driver not initialized"
    elif result == CUDA_ERROR_DEINITIALIZED:
        return "driver deinitialized"
    elif result == CUDA_ERROR_NO_DEVICE:
        return "no CUDA-capable device detected"
    elif result == CUDA_ERROR_INVALID_DEVICE:
        return "invalid device"
    elif result == CUDA_ERROR_NOT_SUPPORTED:
        return "operation not supported"
    elif result == CUDA_ERROR_UNKNOWN:
        return "unknown CUDA driver error"
    return "CUDA driver error code " + String(result)


# this trait is essential for supporting the framework
# currently, the framework uses some clever rebind trickery to bypass statically typed objects and store arbitrary objects within a container, but to have the same type flexibility, we must also be able to identify objects by type
trait Typeable:
    @always_inline
    @staticmethod
    def dtype() -> String:
        ...


def hex_to_float[fld: Int32]() -> Float:
    return bitcast[src_dtype = DType.int32, src_width=1, DType.float32](fld)


def signed_to_unsigned[T: DType]() -> DType:
    comptime if T == DType.int8 or T == DType.uint8:
        return DType.uint8
    elif T == DType.int16 or T == DType.uint16:
        return DType.uint16
    elif T == DType.int32 or T == DType.uint32:
        return DType.uint32
    elif T == DType.int64 or T == DType.uint64:
        return DType.uint64
    elif T == DType.int128 or T == DType.uint128:
        return DType.uint128
    elif T == DType.int256 or T == DType.uint256:
        return DType.uint256
    else:
        # 1.0 removed DType.invalid, so an unsupported dtype is a build error at
        # the instantiation site. The assert must sit in this `else` branch: at
        # the function tail it is evaluated unconditionally, rejecting every T.
        comptime assert False, "signed_to_unsigned requires an integral DType"
        return DType.uint8


@always_inline
def enumerate[T: Movable & Copyable](K: Span[T, _]) -> List[Tuple[Int, T]]:
    var L: List[Tuple[Int, T]] = []
    for i in range(len(K)):
        L.append((i, K[i].copy()))
    return L^


@fieldwise_init
struct TypeableInt(Copyable, Movable, Typeable, TrivialRegisterPassable):
    var val: Int

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "TypeableInt"


@fieldwise_init
struct TypeableUInt(Copyable, Movable, Typeable, TrivialRegisterPassable):
    var val: UInt

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "TypeableUInt"


# OwnedPointer itself doesn't conform to Typeable, so a bare OwnedPointer[T]
# can't be registered with ProductRegistry.consumes/produces (both require
# Typeable). This wrapper delegates dtype() to the inner type, closing that
# gap so heterogeneous-SoA products (see CUDADataFormats/HeterogeneousSoA.mojo)
# can be registered under their own concrete type, matching the C++ originals
# (e.g. PixelTrackHeterogeneous, ZVertexHeterogeneous) directly.
struct TypeableOwnedPointer[T: Typeable & Movable & Deinitable](
    Movable, Typeable
):
    var _inner: OwnedPointer[Self.T]

    @always_inline
    def __init__(out self, var value: Self.T):
        self._inner = OwnedPointer(value^)

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self._inner = move._inner^

    # Borrow the pointee, mirroring C++'s unique_ptr::operator*. This is the
    # only accessor: every caller wants the SoA, not its address.
    @always_inline
    def __getitem__(ref self) -> ref [self._inner[]] Self.T:
        return self._inner[]

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "TypeableOwnedPointer[" + Self.T.dtype() + "]"
