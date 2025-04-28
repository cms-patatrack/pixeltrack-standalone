module DataFormat_Math_choleskyInversion_h

@inline function invert11(src::M1, dst::M2) where {M1,M2}
    F = typeof(src[1, 1])
    dst[1, 1] = F(1.0) / src[1, 1]
end

@inline function invert22(src::M1, dst::M2) where {M1,M2}
    F = typeof(src[1, 1])
    luc0 = F(1.0) / src[1, 1]
    luc1 = src[2, 1] * src[2, 1] * luc0
    luc2 = F(1.0) / (src[2, 2] - luc1)

    li21 = luc1 * luc0 * luc2

    dst[1, 1] = li21 + luc0
    dst[2, 1] = -src[2, 1] * luc0 * luc2
    dst[2, 2] = luc2
end


@inline function invert33(src::M1, dst::M2) where {M1,M2}
    F = typeof(src[1, 1])
    luc0 = F(1.0) / src[1, 1]
    luc1 = src[2, 1]
    luc2 = src[2, 2] - luc0 * luc1 * luc1
    luc2 = F(1.0) / luc2
    luc3 = src[3, 1]
    luc4 = src[3, 2] - luc0 * luc1 * luc3
    luc5 = src[3, 3] - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4)
    luc5 = F(1.0) / luc5

    li21 = -luc0 * luc1
    li32 = -luc2 * luc4
    li31 = (luc1 * luc2 * luc4 - luc3) * luc0

    dst[1, 1] = luc5 * li31 * li31 + luc2 * li21 * li21 + luc0
    dst[2, 1] = luc5 * li31 * li32 + luc2 * li21
    dst[2, 2] = luc5 * li32 * li32 + luc2
    dst[3, 1] = luc5 * li31
    dst[3, 2] = luc5 * li32
    dst[3, 3] = luc5
end

@inline function invert44(src::M1, dst::M2) where {M1,M2}
    F = typeof(src[1, 1])
    luc0 = F(1.0) / src[1, 1]
    luc1 = src[2, 1]
    luc2 = src[2, 2] - luc0 * luc1 * luc1
    luc2 = F(1.0) / luc2
    luc3 = src[3, 1]
    luc4 = src[3, 2] - luc0 * luc1 * luc3
    luc5 = src[3, 3] - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4)
    luc5 = F(1.0) / luc5
    luc6 = src[4, 1]
    luc7 = src[4, 2] - luc0 * luc1 * luc6
    luc8 = src[4, 3] - luc0 * luc3 * luc6 - luc2 * luc4 * luc7
    luc9 = src[4, 4] - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * luc8 * luc5)
    luc9 = F(1.0) / luc9

    li21 = -luc0 * luc1
    li32 = -luc2 * luc4
    li31 = (luc1 * luc2 * luc4 - luc3) * luc0
    li43 = -luc8 * luc5
    li42 = (luc4 * luc8 * luc5 - luc7) * luc2
    li41 = (-luc1 * luc2 * luc4 * luc8 * luc5 + luc1 * luc2 * luc7 + luc3 * luc8 * luc5 - luc6) * luc0

    dst[1, 1] = luc9 * li41 * li41 + luc5 * li31 * li31 + luc2 * li21 * li21 + luc0
    dst[2, 1] = luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21
    dst[2, 2] = luc9 * li42 * li42 + luc5 * li32 * li32 + luc2
    dst[3, 1] = luc9 * li41 * li43 + luc5 * li31
    dst[3, 2] = luc9 * li42 * li43 + luc5 * li32
    dst[3, 3] = luc9 * li43 * li43 + luc5
    dst[4, 1] = luc9 * li41
    dst[4, 2] = luc9 * li42
    dst[4, 3] = luc9 * li43
    dst[4, 4] = luc9
end
@inline function invert55(src::M1, dst::M2) where {M1,M2}
    F = typeof(src[1, 1])
    luc0 = F(1.0) / src[1, 1]
    luc1 = src[2, 1]
    luc2 = src[2, 2] - luc0 * luc1 * luc1
    luc2 = F(1.0) / luc2
    luc3 = src[3, 1]
    luc4 = src[3, 2] - luc0 * luc1 * luc3
    luc5 = src[3, 3] - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4)
    luc5 = F(1.0) / luc5
    luc6 = src[4, 1]
    luc7 = src[4, 2] - luc0 * luc1 * luc6
    luc8 = src[4, 3] - luc0 * luc3 * luc6 - luc2 * luc4 * luc7
    luc9 = src[4, 4] - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * luc8 * luc5)
    luc9 = F(1.0) / luc9
    luc10 = src[5, 1]
    luc11 = src[5, 2] - luc0 * luc1 * luc10
    luc12 = src[5, 3] - luc0 * luc3 * luc10 - luc2 * luc4 * luc11
    luc13 = src[5, 4] - luc0 * luc6 * luc10 - luc2 * luc7 * luc11 - luc5 * luc8 * luc12
    luc14 = src[5, 5] - (luc0 * luc10 * luc10 + luc2 * luc11 * luc11 + luc5 * luc12 * luc12 + luc9 * luc13 * luc13)
    luc14 = F(1.0) / luc14

    li21 = -luc0 * luc1
    li32 = -luc2 * luc4
    li31 = (luc1 * luc2 * luc4 - luc3) * luc0
    li43 = -luc8 * luc5
    li42 = (luc4 * luc8 * luc5 - luc7) * luc2
    li41 = (-luc1 * luc2 * luc4 * luc8 * luc5 + luc1 * luc2 * luc7 + luc3 * luc8 * luc5 - luc6) * luc0
    li54 = -luc13 * luc9
    li53 = (luc13 * luc8 * luc9 - luc12) * luc5
    li52 = (-luc4 * luc8 * luc13 * luc5 * luc9 + luc4 * luc12 * luc5 + luc7 * luc13 * luc9 - luc11) * luc2
    li51 = (luc1 * luc4 * luc8 * luc13 * luc2 * luc5 * luc9 - luc13 * luc8 * luc3 * luc9 * luc5 -
            luc12 * luc4 * luc1 * luc2 * luc5 - luc13 * luc7 * luc1 * luc9 * luc2 + luc11 * luc1 * luc2 +
            luc12 * luc3 * luc5 + luc13 * luc6 * luc9 - luc10) * luc0

    dst[1, 1] = luc14 * li51 * li51 + luc9 * li41 * li41 + luc5 * li31 * li31 + luc2 * li21 * li21 + luc0
    dst[2, 1] = luc14 * li51 * li52 + luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21
    dst[2, 2] = luc14 * li52 * li52 + luc9 * li42 * li42 + luc5 * li32 * li32 + luc2
    dst[3, 1] = luc14 * li51 * li53 + luc9 * li41 * li43 + luc5 * li31
    dst[3, 2] = luc14 * li52 * li53 + luc9 * li42 * li43 + luc5 * li32
    dst[3, 3] = luc14 * li53 * li53 + luc9 * li43 * li43 + luc5
    dst[4, 1] = luc14 * li51 * li54 + luc9 * li41
    dst[4, 2] = luc14 * li52 * li54 + luc9 * li42
    dst[4, 3] = luc14 * li53 * li54 + luc9 * li43
    dst[4, 4] = luc14 * li54 * li54 + luc9
    dst[5, 1] = luc14 * li51
    dst[5, 2] = luc14 * li52
    dst[5, 3] = luc14 * li53
    dst[5, 4] = luc14 * li54
    dst[5, 5] = luc14
end
@inline function invert66(src::M1, dst::M2) where {M1,M2}
    F = typeof(src[1, 1])
    luc0 = F(1.0) / src[1, 1]
    luc1 = src[2, 1]
    luc2 = src[2, 2] - luc0 * luc1 * luc1
    luc2 = F(1.0) / luc2
    luc3 = src[3, 1]
    luc4 = src[3, 2] - luc0 * luc1 * luc3
    luc5 = src[3, 3] - (luc0 * luc3 * luc3 + luc2 * luc4 * luc4)
    luc5 = F(1.0) / luc5
    luc6 = src[4, 1]
    luc7 = src[4, 2] - luc0 * luc1 * luc6
    luc8 = src[4, 3] - luc0 * luc3 * luc6 - luc2 * luc4 * luc7
    luc9 = src[4, 4] - (luc0 * luc6 * luc6 + luc2 * luc7 * luc7 + luc8 * luc8 * luc5)
    luc9 = F(1.0) / luc9
    luc10 = src[5, 1]
    luc11 = src[5, 2] - luc0 * luc1 * luc10
    luc12 = src[5, 3] - luc0 * luc3 * luc10 - luc2 * luc4 * luc11
    luc13 = src[5, 4] - luc0 * luc6 * luc10 - luc2 * luc7 * luc11 - luc5 * luc8 * luc12
    luc14 = src[5, 5] - (luc0 * luc10 * luc10 + luc2 * luc11 * luc11 + luc5 * luc12 * luc12 + luc9 * luc13 * luc13)
    luc14 = F(1.0) / luc14
    luc15 = src[6, 1]
    luc16 = src[6, 2] - luc0 * luc1 * luc15
    luc17 = src[6, 3] - luc0 * luc3 * luc15 - luc2 * luc4 * luc16
    luc18 = src[6, 4] - luc0 * luc6 * luc15 - luc2 * luc7 * luc16 - luc5 * luc8 * luc17
    luc19 = src[6, 5] - (luc0 * luc10 * luc15 + luc2 * luc11 * luc16 + luc5 * luc12 * luc17 + luc9 * luc13 * luc18)
    luc20 = src[6, 6] - (luc0 * luc15 * luc15 + luc2 * luc16 * luc16 + luc5 * luc17 * luc17 + luc9 * luc18 * luc18 + luc14 * luc19 * luc19)
    luc20 = F(1.0) / luc20

    li21 = -luc1 * luc0
    li32 = -luc2 * luc4
    li31 = (luc1 * luc2 * luc4 - luc3) * luc0
    li43 = -luc8 * luc5
    li42 = (luc4 * luc8 * luc5 - luc7) * luc2
    li41 = (-luc1 * luc2 * luc4 * luc8 * luc5 + luc1 * luc2 * luc7 + luc3 * luc8 * luc5 - luc6) * luc0
    li54 = -luc13 * luc9
    li53 = (luc13 * luc8 * luc9 - luc12) * luc5
    li52 = (-luc4 * luc8 * luc13 * luc5 * luc9 + luc4 * luc12 * luc5 + luc7 * luc13 * luc9 - luc11) * luc2
    li51 = (luc1 * luc4 * luc8 * luc13 * luc2 * luc5 * luc9 - luc13 * luc8 * luc3 * luc9 * luc5 - luc12 * luc4 * luc1 * luc2 * luc5 - luc13 * luc7 * luc1 * luc9 * luc2 + luc11 * luc1 * luc2 + luc12 * luc3 * luc5 + luc13 * luc6 * luc9 - luc10) * luc0
    li65 = -luc19 * luc14
    li64 = (luc19 * luc14 * luc13 - luc18) * luc9
    li63 = (-luc8 * luc13 * luc19 * luc14 * luc9 + luc8 * luc9 * luc18 + luc12 * luc19 * luc14 - luc17) * luc5
    li62 = (luc4 * luc8 * luc9 * luc13 * luc5 * luc19 * luc14 - luc18 * luc4 * luc8 * luc9 * luc5 - luc19 * luc12 * luc4 * luc14 * luc5 - luc19 * luc13 * luc7 * luc14 * luc9 + luc17 * luc4 * luc5 + luc18 * luc7 * luc9 + luc19 * luc11 * luc14 - luc16) * luc2
    li61 = (-luc19 * luc13 * luc8 * luc4 * luc1 * luc2 * luc5 * luc9 * luc14 + luc18 * luc8 * luc4 * luc1 * luc2 * luc5 * luc9 + luc19 * luc12 * luc4 * luc1 * luc2 * luc5 * luc14 + luc19 * luc13 * luc7 * luc1 * luc2 * luc9 * luc14 + luc19 * luc13 * luc8 * luc3 * luc5 * luc9 * luc14 - luc17 * luc4 * luc1 * luc2 * luc5 - luc18 * luc7 * luc1 * luc2 * luc9 - luc19 * luc11 * luc1 * luc2 * luc14 - luc18 * luc8 * luc3 * luc5 * luc9 - luc19 * luc12 * luc3 * luc5 * luc14 - luc19 * luc13 * luc6 * luc9 * luc14 + luc16 * luc1 * luc2 + luc17 * luc3 * luc5 + luc18 * luc6 * luc9 + luc19 * luc10 * luc14 - luc15) * luc0

    dst[1, 1] = luc20 * li61 * li61 + luc14 * li51 * li51 + luc9 * li41 * li41 + luc5 * li31 * li31 + luc2 * li21 * li21 + luc0
    dst[2, 1] = luc20 * li61 * li62 + luc14 * li51 * li52 + luc9 * li41 * li42 + luc5 * li31 * li32 + luc2 * li21
    dst[2, 2] = luc20 * li62 * li62 + luc14 * li52 * li52 + luc9 * li42 * li42 + luc5 * li32 * li32 + luc2
    dst[3, 1] = luc20 * li61 * li63 + luc14 * li51 * li53 + luc9 * li41 * li43 + luc5 * li31
    dst[3, 2] = luc20 * li62 * li63 + luc14 * li52 * li53 + luc9 * li42 * li43 + luc5 * li32
    dst[3, 3] = luc20 * li63 * li63 + luc14 * li53 * li53 + luc9 * li43 * li43 + luc5
    dst[4, 1] = luc20 * li61 * li64 + luc14 * li51 + luc9 * li41
    dst[4, 2] = luc20 * li62 * li64 + luc14 * li52 + luc9 * li42
    dst[4, 3] = luc20 * li63 * li64 + luc14 * li53 + luc9 * li43
    dst[4, 4] = luc20 * li64 * li64 + luc14 * li54 * li54 + luc9
    dst[5, 1] = luc20 * li61 + luc14 * li51
    dst[5, 2] = luc20 * li62 + luc14 * li52
    dst[5, 3] = luc20 * li63 + luc14 * li53
    dst[5, 4] = luc20 * li64 + luc14 * li54
    dst[5, 5] = luc20 * li65 * li65 + luc14
    dst[6, 6] = luc20
end

function symmetrize11!(dst::AbstractArray{T}) where {T}
end

function symmetrize22!(dst::AbstractArray{T}) where {T}
    dst[1, 2] = dst[2, 1]
end

function symmetrize33!(dst::AbstractArray{T}) where {T}
    symmetrize22!(dst)
    dst[1, 3] = dst[3, 1]
    dst[2, 3] = dst[3, 2]
end

function symmetrize44!(dst::AbstractArray{T}) where {T}
    symmetrize33!(dst)
    dst[1, 4] = dst[4, 1]
    dst[2, 4] = dst[4, 2]
    dst[3, 4] = dst[4, 3]
end

function symmetrize55!(dst::AbstractArray{T}) where {T}
    symmetrize44!(dst)
    dst[1, 5] = dst[5, 1]
    dst[2, 5] = dst[5, 2]
    dst[3, 5] = dst[5, 3]
    dst[4, 5] = dst[5, 4]
end

function symmetrize66!(dst::AbstractArray{T}) where {T}
    symmetrize55!(dst)
    dst[1, 6] = dst[6, 1]
    dst[2, 6] = dst[6, 2]
    dst[3, 6] = dst[6, 3]
    dst[4, 6] = dst[6, 4]
    dst[5, 6] = dst[6, 5]
end

function invert(src::M1, dst::M2) where {M1,M2}
    number_of_col = size(src, 2)
    if number_of_col == 1
        invert11(src, dst)
    end
    if number_of_col == 2
        invert22(src, dst)
        symmetrize22!(dst)
    end
    if number_of_col == 3
        invert33(src, dst)
        symmetrize33!(dst)
    end
    if number_of_col == 4
        invert44(src, dst)
        symmetrize44!(dst)
    end
    if number_of_col == 5
        invert55(src, dst)
        symmetrize55!(dst)
    end
    if number_of_col == 6
        invert66(src, dst)
        symmetrize66!(dst)
    end

end


end

