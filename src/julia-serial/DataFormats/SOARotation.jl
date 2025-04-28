module SOA_h

export SOARotation, SOAFrame, toGlobal_Special
using ..Printf

struct SOARotation{T}
    R11::T
    R12::T
    R13::T
    R21::T
    R22::T
    R23::T
    R31::T
    R32::T
    R33::T

    # function SOARotation{T}() where {T}
    #     return new{T}(one(Int32), zero(Int32), zero(Int32), zero(Int32), one(Int32), zero(Int32), zero(Int32), zero(Int32), one(Int32))
    # end

    function SOARotation{T}(v::T) where {T}
        return new{T}(one(Int32), zero(Int32), zero(Int32), zero(Int32), one(Int32), zero(Int32), zero(Int32), zero(Int32), one(Int32))
    end

    function SOARotation{T}(xx::T, xy::T, xz::T, yx::T, yy::T, yz::T, zx::T, zy::T, zz::T) where {T}
        return new{T}(xx, xy, xz, yx, yy, yz, zx, zy, zz)
    end

    function SOARotation{T}(p::Vector{T}) where {T}
        return new{T}(p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9])
    end

    # function SOARotation{T}(a::TkRotation{U}) where {T, U}
    #     return new{T}(a.xx(), a.xy(), a.xz(), a.yx(), a.yy(), a.yz(), a.zx(), a.zy(), a.zz())
    # end
end

function transposed(r::SOARotation{T}) where {T}
    return SOARotation{T}(r.R11, r.R21, r.R31, r.R12, r.R22, r.R32, r.R13, r.R23, r.R33)
end

function multiply(r::SOARotation{T}, vx::T, vy::T, vz::T) where {T}
    ux = r.R11 * vx + r.R12 * vy + r.R13 * vz
    uy = r.R21 * vx + r.R22 * vy + r.R23 * vz
    uz = r.R31 * vx + r.R32 * vy + r.R33 * vz
    return ux, uy, uz
end

function multiplyInverse(r::SOARotation{T}, vx::T, vy::T, vz::T) where {T}
    ux = r.R11 * vx + r.R21 * vy + r.R31 * vz
    uy = r.R12 * vx + r.R22 * vy + r.R32 * vz
    uz = r.R13 * vx + r.R23 * vy + r.R33 * vz
    return ux, uy, uz
end

function multiplyInverse(r::SOARotation{T}, vx::T, vy::T) where {T}
    ux = r.R11 * vx + r.R21 * vy
    uy = r.R12 * vx + r.R22 * vy
    uz = r.R13 * vx + r.R23 * vy


    return ux, uy, uz
end

function multiplyInverse(r::SOARotation{T}, vx::T, vy::T, ux::T, uy::T,uz::T) where T
    ux = r.R11 * vx + r.R12 * vy
    uy = r.R12 * vx + r.R22 * vy
    uz = r.R13 * vx + r.R23 * vy
    return ux, uy, uz
end

xx(r::SOARotation) = r.R11
xy(r::SOARotation) = r.R12
xz(r::SOARotation) = r.R13
yx(r::SOARotation) = r.R21
yy(r::SOARotation) = r.R22
yz(r::SOARotation) = r.R23
zx(r::SOARotation) = r.R31
zy(r::SOARotation) = r.R32
zz(r::SOARotation) = r.R33


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

struct SOAFrame{T}
    px::T
    py::T
    pz::T
    rot::SOARotation{T}

    function SOAFrame{T}() where {T}
        return new{T}()
    end

    function SOAFrame{T}(ix::T, iy::T, iz::T, irot::SOARotation{T}) where {T}
        return new{T}(ix, iy, iz, irot)
    end
end

function rotation(frame::SOAFrame{T}) where {T}
    return frame.rot
end

function toLocal(frame::SOAFrame{T}, vx::T, vy::T, vz::T) where {T}
    multiply(frame.rot,vx - frame.px, vy - frame.py, vz - frame.pz)
end

function toGlobal(frame::SOAFrame{T}, vx::T, vy::T, vz::T) where {T}
    ux, uy, uz = multiplyInverse(frame.rot,vx, vy, vz)
    ux += frame.px
    uy += frame.py
    uz += frame.pz
    return ux, uy, uz
end

function toGlobal(frame::SOAFrame{T}, vx::T, vy::T) where {T}
    ux, uy, uz = multiplyInverse(frame.rot,vx, vy)
    ux += frame.px
    uy += frame.py
    uz += frame.pz
    return ux, uy, uz
end

function toGlobal_Special(frame::SOAFrame{T}, vx::T, vy::T) where {T}
    ux, uy, uz = multiplyInverse(frame.rot, vx, vy);
    # write(io, @sprintf("%.5f", ux), " ")
    # write(io, @sprintf("%.5f", uy), " ")
    # write(io, @sprintf("%.5f", uz), "\n")
    ux += frame.px
    uy += frame.py
    uz += frame.pz
    #println(frame.pz)
    return ux, uy, uz
end


function toGlobal(frame::SOAFrame{T}, cxx::T, cxy::T, cyy::T, gl::Vector{T}) where {T}
    r = frame.rot
    gl[1] = xx(r) * (xx(r) * cxx + yx(r) * cxy) + yx(r) * (xx(r) * cxy + yx(r) * cyy)
    gl[2] = xx(r) * (xy(r) * cxx + yy(r) * cxy) + yx(r) * (xy(r) * cxy + yy(r) * cyy)
    gl[3] = xy(r) * (xy(r) * cxx + yy(r) * cxy) + yy(r) * (xy(r) * cxy + yy(r) * cyy)
    gl[4] = xx(r) * (xz(r) * cxx + yz(r) * cxy) + yx(r) * (xz(r) * cxy + yz(r) * cyy)
    gl[5] = xy(r) * (xz(r) * cxx + yz(r) * cxy) + yy(r) * (xz(r) * cxy + yz(r) * cyy)
    gl[6] = xz(r) * (xz(r) * cxx + yz(r) * cxy) + yz(r) * (xz(r) * cxy + yz(r) * cyy)
end

function toLocal(frame::SOAFrame{T}, ge::Vector{T}) where {T}
    r = frame.rot

    cxx = ge[1]
    cyx = ge[2]
    cyy = ge[3]
    czx = ge[4]
    czy = ge[5]
    czz = ge[6]

    lxx = xx(r) * (xx(r) * cxx + xy(r) * cyx + xz(r) * czx) +
          xy(r) * (xx(r) * cyx + xy(r) * cyy + xz(r) * czy) + xz(r) * (xx(r) * czx + xy(r) * czy + xz(r) * czz)
    lxy = yx(r) * (xx(r) * cxx + xy(r) * cyx + xz(r) * czx) +
          yy(r) * (xx(r) * cyx + xy(r) * cyy + xz(r) * czy) + yz(r) * (xx(r) * czx + xy(r) * czy + xz(r) * czz)
    lyy = yx(r) * (yx(r) * cxx + yy(r) * cyx + yz(r) * czx) +
          yy(r) * (yx(r) * cyx + yy(r) * cyy + yz(r) * czy) + yz(r) * (yx(r) * czx + yy(r) * czy + yz(r) * czz)

    return lxx, lxy, lyy
end

function x(frame::SOAFrame{T}) where {T}
    return frame.px
end

function y(frame::SOAFrame{T}) where {T}
    return frame.py
end

function z(frame::SOAFrame{T}) where {T}
    return frame.pz
end

end