from std.memory import bitcast
from std.math import pi

from MojoSerial.MojoBridge.DTypes import Short, Float, Double, hex_to_float
struct ApproxAtan2:
    """
    Approximate atan2 evaluations. Polynomials were obtained using Sollya scripts.
    """

    @staticmethod
    def approx_atan2f_P[DEGREE: Int](x: Float) -> Float:
        comptime assert (
            DEGREE == 3
            or DEGREE == 5
            or DEGREE == 7
            or DEGREE == 9
            or DEGREE == 11
            or DEGREE == 13
            or DEGREE == 15
        ), (
            "degree of the polynomial to approximate atan(x) must be one of"
            " {3, 5, 7, 9, 11, 13, 15}."
        )
        var z = x * x

        comptime if DEGREE == 3:
            # degree =  3   => absolute accuracy is  7 bits
            return x * (
                hex_to_float[0xBF78EED2]() + z * hex_to_float[0x3E448E00]()
            )
        elif DEGREE == 5:
            # degree =  5   => absolute accuracy is  10 bits
            return x * (
                hex_to_float[0xBF7ECFC8]()
                + z
                * (hex_to_float[0x3E93CF3A]() + z * hex_to_float[0xBDA27C92]())
            )
        elif DEGREE == 7:
            # degree =  7   => absolute accuracy is  13 bits
            return x * (
                hex_to_float[0xBF7FCC7A]()
                + z
                * (
                    hex_to_float[0x3EA4710C]()
                    + z
                    * (
                        hex_to_float[0xBE15C65A]()
                        + z * hex_to_float[0x3D1FB050]()
                    )
                )
            )
        elif DEGREE == 9:
            # degree =  9   => absolute accuracy is  16 bits
            return x * (
                hex_to_float[0xBF7FF73E]()
                + z
                * (
                    hex_to_float[0x3EA91DC2]()
                    + z
                    * (
                        hex_to_float[0xBE387BFA]()
                        + z
                        * (
                            hex_to_float[0x3DAE672A]()
                            + z * hex_to_float[0xBCAAC48A]()
                        )
                    )
                )
            )
        elif DEGREE == 11:
            # degree =  11   => absolute accuracy is  19 bits
            return x * (
                hex_to_float[0xBF7FFE82]()
                + z
                * (
                    hex_to_float[0x3EAA4D90]()
                    + z
                    * (
                        hex_to_float[0xBE462FAA]()
                        + z
                        * (
                            hex_to_float[0x3DEE71DE]()
                            + z
                            * (
                                hex_to_float[0xBD57A64A]()
                                + z * hex_to_float[0x3C4003A8]()
                            )
                        )
                    )
                )
            )
        elif DEGREE == 13:
            # degree =  13   => absolute accuracy is  21 bits
            return x * (
                hex_to_float[0xBF7FFFBE]()
                + z
                * (
                    hex_to_float[0x3EAA95A0]()
                    + z
                    * (
                        hex_to_float[0xBE4AD37E]()
                        + z
                        * (
                            hex_to_float[0x3E077DE4]()
                            + z
                            * (
                                hex_to_float[0xBDA30408]()
                                + z
                                * (
                                    hex_to_float[0x3D099028]()
                                    + z * hex_to_float[0xBBDF05E2]()
                                )
                            )
                        )
                    )
                )
            )
        elif DEGREE == 15:
            # degree =  15   => absolute accuracy is  24 bits
            return x * (
                hex_to_float[0xBF7FFFF4]()
                + z
                * (
                    hex_to_float[0x3EAAA5F2]()
                    + z
                    * (
                        hex_to_float[0xBE4C3DCA]()
                        + z
                        * (
                            hex_to_float[0x3E0E6098]()
                            + z
                            * (
                                hex_to_float[0xBDC54406]()
                                + z
                                * (
                                    hex_to_float[0x3D6484D6]()
                                    + z
                                    * (
                                        hex_to_float[0xBCB27AA0]()
                                        + z * hex_to_float[0x3B843AEE]()
                                    )
                                )
                            )
                        )
                    )
                )
            )
        else:
            # will never happen
            return 0

    @staticmethod
    def unsafe_atan2f_impl[DEGREE: Int](y: Float, x: Float) -> Float:
        comptime pi4f: Float = 3.1415926535897932384626434 / 4
        comptime pi34f: Float = 3.1415926535897932384626434 * 3 / 4

        var r: Float = (abs(x) - abs(y)) / (abs(x) + abs(y))
        if x < 0.0:
            r = -r

        var angle: Float = pi4f if x >= 0.0 else pi34f
        angle += Self.approx_atan2f_P[DEGREE](r)

        return -angle if y < 0.0 else angle

    @staticmethod
    def unsafe_atan2f[DEGREE: Int](y: Float, x: Float) -> Float:
        return Self.unsafe_atan2f_impl[DEGREE](y, x)

    @staticmethod
    def safe_atan2f[DEGREE: Int](y: Float, x: Float) -> Float:
        return Self.unsafe_atan2f[DEGREE](
            y, 0.2 if y == 0.0 and x == 0.0 else x
        )

    @staticmethod
    def approx_atan2i_P[DEGREE: Int](x: Float) -> Float:
        comptime assert (
            DEGREE == 3
            or DEGREE == 5
            or DEGREE == 7
            or DEGREE == 9
            or DEGREE == 11
            or DEGREE == 13
            or DEGREE == 15
        ), (
            "degree of the polynomial to approximate atan(x) must be one of"
            " {3, 5, 7, 9, 11, 13, 15}."
        )
        var z = x * x

        comptime if DEGREE == 3:
            # degree =  3   => absolute accuracy is  6*10^6
            return x * (-664694912.0 + z * 131209024.0)
        elif DEGREE == 5:
            # degree =  5   => absolute accuracy is  4*10^5
            return x * (-680392064.0 + z * (197338400.0 + z * (-54233256.0)))
        elif DEGREE == 7:
            # degree =  7   => absolute accuracy is  6*10^4
            return x * (
                -683027840.0
                + z * (219543904.0 + z * (-99981040.0 + z * 26649684.0))
            )
        elif DEGREE == 9:
            # degree =  9   => absolute accuracy is  8000
            return x * (
                -683473920.0
                + z
                * (
                    225785056.0
                    + z * (-123151184.0 + z * (58210592.0 + z * (-14249276.0)))
                )
            )
        elif DEGREE == 11:
            # degree =  11   => absolute accuracy is  1000
            return x * (
                -683549696.0
                + z
                * (
                    227369312.0
                    + z
                    * (
                        -132297008.0
                        + z * (79584144.0 + z * (-35987016.0 + z * 8010488.0))
                    )
                )
            )
        elif DEGREE == 13:
            # degree =  13   => absolute accuracy is  163
            return x * (
                -683562624.0
                + z
                * (
                    227746080.0
                    + z
                    * (
                        -135400128.0
                        + z
                        * (
                            90460848.0
                            + z
                            * (
                                -54431464.0
                                + z * (22973256.0 + z * (-4657049.0))
                            )
                        )
                    )
                )
            )
        elif DEGREE == 15:
            return x * (
                -683562624.0
                + z
                * (
                    227746080.0
                    + z
                    * (
                        -135400128.0
                        + z
                        * (
                            90460848.0
                            + z
                            * (
                                -54431464.0
                                + z * (22973256.0 + z * (-4657049.0))
                            )
                        )
                    )
                )
            )
        else:
            # will never happen
            return 0

    @staticmethod
    def unsafe_atan2i_impl[DEGREE: Int](y: Float, x: Float) -> Int:
        comptime pi4: Int = Int((Int32.MAX.cast[DType.int64]() + 1) // 4)
        comptime pi34: Int = Int(3 * (Int32.MAX.cast[DType.int64]() + 1) // 4)

        var r: Float = (abs(x) - abs(y)) / (abs(x) + abs(y))
        if x < 0:
            r = -r

        var angle: Int = pi4 if x >= 0.0 else pi34
        angle += Int(Self.approx_atan2i_P[DEGREE](r))

        return -angle if y < 0.0 else angle

    @staticmethod
    def unsafe_atan2i[DEGREE: Int](y: Float, x: Float) -> Int:
        return Self.unsafe_atan2i_impl[DEGREE](y, x)

    @staticmethod
    def approx_atan2s_P[DEGREE: Int](x: Float) -> Float:
        comptime assert (
            DEGREE == 3 or DEGREE == 5 or DEGREE == 7 or DEGREE == 9
        ), (
            "degree of the polynomial to approximate atan(x) must be one of"
            " {3, 5, 7, 9}."
        )
        var z = x * x

        comptime if DEGREE == 3:
            # degree =  3   => absolute accuracy is  53
            return x * ((-10142.439453125) + z * 2002.0908203125)
        elif DEGREE == 5:
            # degree =  5   => absolute accuracy is  7
            return x * (
                (-10381.9609375)
                + z * ((3011.1513671875) + z * (-827.538330078125))
            )
        elif DEGREE == 7:
            # degree =  7   => absolute accuracy is  2
            return x * (
                (-10422.177734375)
                + z
                * (
                    3349.97412109375
                    + z * ((-1525.589599609375) + z * 406.64190673828125)
                )
            )
        elif DEGREE == 9:
            # degree =  9   => absolute accuracy is 1
            return x * (
                (-10428.984375)
                + z
                * (
                    3445.20654296875
                    + z
                    * (
                        (-1879.137939453125)
                        + z * (888.22314453125 + z * (-217.42669677734375))
                    )
                )
            )
        else:
            # will never happen
            return 0

    @staticmethod
    def unsafe_atan2s_impl[DEGREE: Int](y: Float, x: Float) -> Short:
        comptime pi4: Short = ((Int16.MAX.cast[DType.int64]() + 1) // 4).cast[
            DType.int16
        ]()
        comptime pi34: Short = (3 * (Int16.MAX.cast[DType.int64]() + 1) // 4).cast[
            DType.int16
        ]()

        var r: Float = (abs(x) - abs(y)) / (abs(x) + abs(y))
        if x < 0:
            r = -r

        var angle: Short = pi4 if x >= 0.0 else pi34
        angle += Self.approx_atan2s_P[DEGREE](r).cast[DType.int16]()

        return -angle if y < 0.0 else angle

    @staticmethod
    def unsafe_atan2s[DEGREE: Int](y: Float, x: Float) -> Short:
        return Self.unsafe_atan2s_impl[DEGREE](y, x)

    @staticmethod
    def phi2int(x: Float) -> Int:
        comptime p2i: Float = (
            (Int32.MAX.cast[DType.int64]() + 1).cast[DType.float32]() / pi
        )
        return Int(round(x * p2i))

    @staticmethod
    def int2phi(x: Int) -> Float:
        comptime i2p: Float = (
            pi / (Int32.MAX.cast[DType.int64]() + 1).cast[DType.float32]()
        )
        return Float(x) * i2p

    @staticmethod
    def int2dphi(x: Int) -> Double:
        comptime i2p: Double = (
            pi / (Int32.MAX.cast[DType.int64]() + 1).cast[DType.float64]()
        )
        return Double(x) * i2p

    @staticmethod
    def phi2short(x: Float) -> Short:
        comptime p2i: Float = (
            (Int16.MAX.cast[DType.int32]() + 1).cast[DType.float32]() / pi
        )
        return Short(round(x * p2i))

    @staticmethod
    def short2phi(x: Short) -> Float:
        comptime i2p: Float = (
            pi / (Int16.MAX.cast[DType.int32]() + 1).cast[DType.float32]()
        )
        return Float(x) * i2p
