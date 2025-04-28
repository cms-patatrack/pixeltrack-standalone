module DataFormatsMathAPPROX_ATAN2_H

export unsafe_atan2s
# Define polynomial approximations for different degrees for float (32 bits)


# degree =  3   => absolute accuracy is  7 bits
function approx_atan2f_P(x::Float32, ::Val{3})
    return x * (Float32(-0xf.8eed2p-4) + x * x * Float32(0x3.1238p-4))
end

# degree =  5   => absolute accuracy is  10 bits
function approx_atan2f_P(x::Float32, ::Val{5})
    z = x * x
    return x * (Float32(-0xf.ecfc8p-4) + z * (Float32(0x4.9e79dp-4) + z * Float32(-0x1.44f924p-4)))
end

# degree =  7   => absolute accuracy is  13 bits
function approx_atan2f_P(x::Float32, ::Val{7})
    z = x * x
    return x * (Float32(-0xf.fcc7ap-4) + z * (Float32(0x5.23886p-4) + z * (Float32(-0x2.571968p-4) + z * Float32(0x9.fb05p-8))))
end

# degree =  9   => absolute accuracy is  16 bits
function approx_atan2f_P(x::Float32, ::Val{9})
    z = x * x
    return x * (Float32(-0xf.ff73ep-4) + z * (Float32(0x5.48ee1p-4) + z * (Float32(-0x2.e1efe8p-4) + z * (Float32(0x1.5cce54p-4) + z * Float32(-0x5.56245p-8)))))
end

# degree =  11   => absolute accuracy is  19 bits
function approx_atan2f_P(x::Float32, ::Val{11})
    z = x * x
    return x * (Float32(-0xf.ffe82p-4) + z * (Float32(0x5.526c8p-4) + z * (Float32(-0x3.18bea8p-4) + z * (Float32(0x1.dce3bcp-4) + z * (Float32(-0xd.7a64ap-8) + z * Float32(0x3.000eap-8))))))
end

# degree =  13   => absolute accuracy is  21 bits
function approx_atan2f_P(x::Float32, ::Val{13})
    z = x * x
    return x * (Float32(-0xf.fffbep-4) + z * (Float32(0x5.54adp-4) + z * (Float32(-0x3.2b4df8p-4) + z * (Float32(0x2.1df79p-4) + z * (Float32(-0x1.46081p-4) + z * (Float32(0x8.99028p-8) + z * Float32(-0x1.be0bc4p-8)))))))
end

# degree =  15   => absolute accuracy is  24 bits
function approx_atan2f_P(x::Float32, ::Val{15})
    z = x * x
    return x * (Float32(-0xf.ffff4p-4) + z * (Float32(0x5.552f9p-4) + z * (Float32(-0x3.30f728p-4) + z * (Float32(0x2.39826p-4) + z * (Float32(-0x1.8a880cp-4) + z * (Float32(0xe.484d6p-8) + z * (Float32(-0x5.93d5p-8) + z * Float32(0x1.0875dcp-8))))))))
end

function unsafe_atan2f_impl(y::Float32, x::Float32, ::Val{DEGREE})::Float32 where {DEGREE}
    pi4f = 3.1415926535897932384626434 / 4
    pi34f = 3 * 3.1415926535897932384626434 / 4
    r = (abs(x) - abs(y)) / (abs(x) + abs(y))
    if x < 0
        r = -r
    end

    angle = (x >= 0) ? pi4f : pi34f
    angle += approx_atan2f_P(r, Val{DEGREE}())

    return (y < 0) ? -angle : angle
end

function unsafe_atan2f(y::Float32, x::Float32, ::Val{DEGREE})::Float32 where {DEGREE}
    return unsafe_atan2f_impl(y, x, Val{DEGREE}())
end

function safe_atan2f(y::Float32, x::Float32, ::Val{DEGREE})::Float32 where {DEGREE}
    return unsafe_atan2f_impl(y, (y == 0f0 && x == 0f0) ? 0.2f0 : x, Val{DEGREE}())
end


# Define polynomial approximations for different degrees for int (32 bits)


# degree =  3   => absolute accuracy is  6*10^6
function approx_atan2i_P(x::Float32, ::Val{3})
    z = x * x
    return x * (-664694912f0 + z * 131209024f0)
end

# degree =  5   => absolute accuracy is  4*10^5
function approx_atan2i_P(x::Float32, ::Val{5})
    z = x * x
    return x * (-680392064f0 + z * (197338400f0 + z * (-54233256f0)))
end

# degree =  7   => absolute accuracy is  6*10^4
function approx_atan2i_P(x::Float32, ::Val{7})
    z = x * x
    return x * (-683027840f0 + z * (219543904f0 + z * (-99981040f0 + z * 26649684f0)))
end

# degree =  9   => absolute accuracy is  8000
function approx_atan2i_P(x::Float32, ::Val{9})
    z = x * x
    return x * (-683473920f0 + z * (225785056f0 + z * (-123151184f0 + z * (58210592f0 + z * (-14249276f0)))))
end

# degree =  11   => absolute accuracy is  1000
function approx_atan2i_P(x::Float32, ::Val{11})
    z = x * x
    return x * (-683549696f0 + z * (227369312f0 + z * (-132297008f0 + z * (79584144f0 + z * (-35987016f0 + z * 8010488f0)))))
end

# degree =  13   => absolute accuracy is  163
function approx_atan2i_P(x::Float32, ::Val{13})
    z = x * x
    return x * (-683562624f0 + z * (227746080f0 + z * (-135400128f0 + z * (90460848f0 + z * (-54431464f0 + z * (22973256f0 + z * (-4657049f0)))))))
end

# degree =  15   => absolute accuracy is  163
function approx_atan2i_P(x::Float32, ::Val{15})
    z = x * x
    return x * (-683562624f0 + z * (227746080f0 + z * (-135400128f0 + z * (90460848f0 + z * (-54431464f0 + z * (22973256f0 + z * (-4657049f0)))))))
end

function unsafe_atan2i_impl(y::Float32, x::Float32, ::Val{DEGREE})::Int32 where {DEGREE}
    maxint = (Int64(typemax(Int32)) + 1)
    pi4 = Int32(maxint / 4)
    pi34 = Int32(3 * maxint / 4)

    r = (abs(x) - abs(y)) / (abs(x) + abs(y))
    if x < 0
        r = -r
    end

    angle = (x >= 0) ? pi4 : pi34
    angle += Int32(approx_atan2i_P(r, Val{DEGREE}()))

    return (y < 0) ? -angle : angle
end

function unsafe_atan2i(y::Float32, x::Float32, ::Val{DEGREE})::Int32 where {DEGREE}
    return unsafe_atan2i_impl(y, x, Val{DEGREE}())
end


# Define polynomial approximations for different degrees for short (16 bits)


# # degree =  3   => absolute accuracy is  53
# function approx_atan2s_P(x::Float32, ::Val{3})
#     z = x * x
#     return x * (-10142.439453125f0 + z * 2002.0908203125f0)
# end

# # degree =  5   => absolute accuracy is  7
# function approx_atan2s_P(x::Float32, ::Val{5})
#     z = x * x
#     return x * (-10381.9609375f0 + z * (3011.1513671875f0 + z * (-827.538330078125f0)))
# end

# # degree =  7   => absolute accuracy is  2
# function approx_atan2s_P(x::Float32, ::Val{7})
#     z = x * x
#     return x * (-10422.177734375f0 + z * (3349.97412109375f0 + z * (-1525.589599609375f0 + z * 406.64190673828125f0)))
# end

# degree =  9   => absolute accuracy is 1
function approx_atan2s_P(x::Float32)
    z = x * x
    return x * (-10422.177734375f0 + z * (3349.97412109375f0 + z * (-1525.589599609375f0 + z * 406.64190673828125f0)))
end

function unsafe_atan2s_impl(y::Float32, x::Float32,DEGREE::Int)::Int16 
    maxshort = Int32(typemax(Int16)) + 1
    pi4 = Int16(maxshort / 4)
    pi34 = Int16(3 * maxshort / 4)

    r = (abs(x) - abs(y)) / (abs(x) + abs(y))
    

    if x < 0
        r = -r
    end

    angle = (x >= 0) ? pi4 : pi34
    angle += Int16(trunc(approx_atan2s_P(r)))

    return (y < 0) ? -angle : angle
end

function unsafe_atan2s(y::Float32, x::Float32,DEGREE::Int)::Int16 # TO FIX POLYMORPHISM ACCORDING TO DEGREE LATER !!!!
    return unsafe_atan2s_impl(y, x, DEGREE)
end


# Conversion functions

function phi2int(x::Float32)::Int32
    pi_val = Float64(Base.MathConstants.pi)
    p2i = (Float64(typemax(Int32)) + 1.0) / pi_val
    result = Float32(x * p2i)
    return round(Int32, result)
end

function int2phi(x::Int32)::Float32
    pi_val = Float64(Base.MathConstants.pi)    
    i2p = pi_val / (Float64(typemax(Int32)) + 1.0)
    result = Float32(x) * Float32(i2p)
    return result
end

function int2dphi(x::Int32)::Float64
    pi_val = Float64(Base.MathConstants.pi)    
    i2p = pi_val / (Float64(typemax(Int32)) + 1.0)
    result = Float64(x) * Float64(i2p)
    return result
end

function phi2short(x::Float32)::Int16
    pi_val = Float64(Base.MathConstants.pi)
    p2i = (Float64(typemax(Int16)) + 1.0) / pi_val
    result = x * Float32(p2i)
    return round(Int16, result)
end

function short2phi(x::Int16)::Float32
    pi_val = Float64(Base.MathConstants.pi)
    i2p = pi_val / (Float64(typemax(Int16)) + 1.0)
    result = Float32(x) * Float32(i2p)
    return result
end


# using Printf
#
# function test_functions()
#     Testing every approximation function
#     println("Testing approx_atan2f_P")
#     @printf("%.15f\n",approx_atan2f_P(0.5f0, Val{3}()))
#     @printf("%.15f\n",approx_atan2f_P(0.5f0, Val{5}()))
#     @printf("%.15f\n",approx_atan2f_P(0.5f0, Val{7}()))
#     @printf("%.15f\n",approx_atan2f_P(0.5f0, Val{9}()))
#     @printf("%.15f\n",approx_atan2f_P(0.5f0, Val{11}()))
#     @printf("%.15f\n",approx_atan2f_P(0.5f0, Val{13}()))
#     @printf("%.15f\n",approx_atan2f_P(0.5f0, Val{15}()))
#     println("Testing approx_atan2i_P")
#     @printf("%.15f\n",approx_atan2i_P(0.5f0, Val{3}()))
#     @printf("%.15f\n",approx_atan2i_P(0.5f0, Val{5}()))
#     @printf("%.15f\n",approx_atan2i_P(0.5f0, Val{7}()))
#     @printf("%.15f\n",approx_atan2i_P(0.5f0, Val{9}()))
#     @printf("%.15f\n",approx_atan2i_P(0.5f0, Val{11}()))
#     @printf("%.15f\n",approx_atan2i_P(0.5f0, Val{13}()))
#     @printf("%.15f\n",approx_atan2i_P(0.5f0, Val{15}()))
#     println("Testing approx_atan2s_P")
#     @printf("%.15f\n",approx_atan2s_P(0.5f0, Val{3}()))
#     @printf("%.15f\n",approx_atan2s_P(0.5f0, Val{5}()))
#     @printf("%.15f\n",approx_atan2s_P(0.5f0, Val{7}()))
#     @printf("%.15f\n",approx_atan2s_P(0.5f0, Val{9}()))
#     println("Testing unsafe_atan2f")
#     @printf("%.15f\n",unsafe_atan2f(0.5f0, 0.5f0, Val{3}()))
#     @printf("%.15f\n",unsafe_atan2f(0.5f0, 0.5f0, Val{5}()))
#     @printf("%.15f\n",unsafe_atan2f(0.5f0, 0.5f0, Val{7}()))
#     @printf("%.15f\n",unsafe_atan2f(0.5f0, 0.5f0, Val{9}()))
#     @printf("%.15f\n",unsafe_atan2f(0.5f0, 0.5f0, Val{11}()))
#     @printf("%.15f\n",unsafe_atan2f(0.5f0, 0.5f0, Val{13}()))
#     @printf("%.15f\n",unsafe_atan2f(0.5f0, 0.5f0, Val{15}()))
#     println("Testing unsafe_atan2i")
#     @printf("%.15f\n",unsafe_atan2i(0.5f0, 0.5f0, Val{3}()))
#     @printf("%.15f\n",unsafe_atan2i(0.5f0, 0.5f0, Val{5}()))
#     @printf("%.15f\n",unsafe_atan2i(0.5f0, 0.5f0, Val{7}()))
#     @printf("%.15f\n",unsafe_atan2i(0.5f0, 0.5f0, Val{9}()))
#     @printf("%.15f\n",unsafe_atan2i(0.5f0, 0.5f0, Val{11}()))
#     @printf("%.15f\n",unsafe_atan2i(0.5f0, 0.5f0, Val{13}()))
#     @printf("%.15f\n",unsafe_atan2i(0.5f0, 0.5f0, Val{15}()))
#     println("Testing unsafe_atan2s")
#     @printf("%.15f\n",unsafe_atan2s(0.5f0, 0.5f0, Val{3}()))
#     @printf("%.15f\n",unsafe_atan2s(0.5f0, 0.5f0, Val{5}()))
#     @printf("%.15f\n",unsafe_atan2s(0.5f0, 0.5f0, Val{7}()))
#     @printf("%.15f\n",unsafe_atan2s(0.5f0, 0.5f0, Val{9}()))
#     println("Testing phi2int")
#     @printf("%.15f\n",phi2int(0.5f0))
#     println("Testing int2phi")
#     @printf("%.15f\n",int2phi(Int32(402123)))
#     println("Testing int2dphi")
#     @printf("%.15f\n",int2dphi(Int32(402123)))
#     println("Testing phi2short")
#     @printf("%.15f\n",phi2short(0.5f0))
#     println("Testing short2phi")
#     @printf("%.15f\n",short2phi(Int16(257)))
# end
#
# test_functions()

end # module DataFormatsMathAPPROX_ATAN2_H
