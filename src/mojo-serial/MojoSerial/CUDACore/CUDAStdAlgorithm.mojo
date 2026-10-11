from std.sys import size_of
struct CUDAStdAlgorithm:
    @staticmethod
    def lower_bound[
        T: DType, //
    ](s: Span[Scalar[T], _], var value: Scalar[T]) -> Int:
        """Returns the index of the first element not less than `value`."""
        var first = 0
        var count = len(s)

        while count > 0:
            var it = first
            var step = count // 2
            it += step

            if s[it] < value:
                it += 1
                first = it
                count -= step + 1
            else:
                count = step

        return first

    @staticmethod
    def upper_bound[
        T: DType, //
    ](s: Span[Scalar[T], _], var value: Scalar[T]) -> Int:
        """Returns the index of the first element greater than `value`."""
        var first = 0
        var count = len(s)

        while count > 0:
            var it = first
            var step = count // 2
            it += step

            if value >= s[it]:
                it += 1
                first = it
                count -= step + 1
            else:
                count = step

        return first

    @staticmethod
    def binary_find[
        T: DType, //
    ](s: Span[Scalar[T], _], var value: Scalar[T]) -> Int:
        """Returns the index of `value`, or `len(s)` if not present."""
        var first = Self.lower_bound(s, value)

        return first if (first != len(s)) and (value >= s[first]) else len(s)
