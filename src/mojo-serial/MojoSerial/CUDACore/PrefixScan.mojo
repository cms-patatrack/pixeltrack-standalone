def blockPrefixScan[
    VT: DType
](
    ci: Span[Scalar[VT], _],
    co: Span[mut=True, Scalar[VT], _],
    size: UInt32,
):
    co[0] = ci[0]
    for i in range(1, Int(size)):
        co[i] = ci[i] + co[i - 1]


# `size` stays a parameter rather than becoming len(c): callers scan a prefix
# of a longer buffer (C++ passes moduleStart + 1 with size 1024).
def blockPrefixScan[
    VT: DType
](c: Span[mut=True, Scalar[VT], _], size: UInt32):
    for i in range(1, Int(size)):
        c[i] += c[i - 1]


# multiBlockPrefixScan is actually a non-working stub, ignore
