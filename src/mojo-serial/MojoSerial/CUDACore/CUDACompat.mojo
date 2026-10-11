from std.memory import Pointer

# Vestigial: the serial backend has no streams, so this is only ever a
# placeholder default argument and is never dereferenced. `Pointer` is
# non-nullable in 1.0, so the null stream is an empty Optional.
comptime CUDAStreamType = Optional[Pointer[NoneType, MutUntrackedOrigin]]
comptime cudaStreamDefault = CUDAStreamType(None)


@deprecated(
    "Any methods using CUDACompat should be redirected to perform the regular"
    " operations since we are not in a CUDA environment."
)
struct CUDACompat:
    # Serial backend: these are plain read-modify-write, not atomics. They take
    # `mut` rather than a pointer so the borrow is tracked. atomicCAS is the
    # exception -- GPUCACell's CUDA-only branch CASes on a reinterpreted
    # pointer field, which has no Scalar lvalue to borrow.
    @staticmethod
    @deprecated(
        "Any methods using CUDACompat should be redirected to perform the"
        " regular operations since we are not in a CUDA environment."
    )
    def atomicCAS[
        T1: DType, //
    ](
        address: Pointer[mut=True, Scalar[T1], _],
        compare: Scalar[T1],
        val: Scalar[T1],
    ) -> Scalar[T1]:
        var old = address[]
        address[] = val if old == compare else old
        return old

    @staticmethod
    @deprecated(
        "Any methods using CUDACompat should be redirected to perform the"
        " regular operations since we are not in a CUDA environment."
    )
    def atomicInc[
        T1: DType, //
    ](mut a: Scalar[T1], b: Scalar[T1]) -> Scalar[T1]:
        var ret = a
        if a < b:
            a += 1
        return ret

    @staticmethod
    @deprecated(
        "Any methods using CUDACompat should be redirected to perform the"
        " regular operations since we are not in a CUDA environment."
    )
    def atomicAdd[
        T1: DType, //
    ](mut a: Scalar[T1], b: Scalar[T1]) -> Scalar[T1]:
        var ret = a
        a += b
        return ret

    @staticmethod
    @deprecated(
        "Any methods using CUDACompat should be redirected to perform the"
        " regular operations since we are not in a CUDA environment."
    )
    def atomicSub[
        T1: DType, //
    ](mut a: Scalar[T1], b: Scalar[T1]) -> Scalar[T1]:
        var ret = a
        a -= b
        return ret

    @staticmethod
    @deprecated(
        "Any methods using CUDACompat should be redirected to perform the"
        " regular operations since we are not in a CUDA environment."
    )
    def atomicMin[
        T1: DType, //
    ](mut a: Scalar[T1], b: Scalar[T1]) -> Scalar[T1]:
        var ret = a
        a = min(a, b)
        return ret

    @staticmethod
    @deprecated(
        "Any methods using CUDACompat should be redirected to perform the"
        " regular operations since we are not in a CUDA environment."
    )
    def atomicMax[
        T1: DType, //
    ](mut a: Scalar[T1], b: Scalar[T1]) -> Scalar[T1]:
        var ret = a
        a = max(a, b)
        return ret