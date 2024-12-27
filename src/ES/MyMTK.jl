"""
  
  # x0 = Vector{Pair{Num, Float64}}(undef,3)
  # @variables x[1:3]
  # for isp in 1:2
  #     x0[isp] = x[isp] => 1.0isp
  #     @show x0[isp]
  # end
"""

function averiables_xvec(Nx)

    if Nx == 1
        return @variables x
    else
        if Nx == 2
            @variables x1 x2
            return [x1, x2]
        elseif Nx == 3
            @variables x1 x2 x3
            return [x1, x2, x3]
        elseif Nx == 4
            @variables x1 x2 x3 x4
            return [x1, x2, x3, x4]
        elseif Nx == 5
            @variables x1 x2 x3 x4 x5
            return [x1, x2, x3, x4, x5]
        elseif Nx == 6
            @variables x1 x2 x3 x4 x5 x6
            return [x1, x2, x3, x4, x5, x6]
        elseif Nx == 7
            @variables x1 x2 x3 x4 x5 x6 x7
            return [x1, x2, x3, x4, x5, x6, x7]
        elseif Nx == 8
            @variables x1 x2 x3 x4 x5 x6 x7 x8
            return [x1, x2, x3, x4, x5, x6, x7, x8]
        elseif Nx == 9
            @variables x1 x2 x3 x4 x5 x6 x7 x8 x9
            return [x1, x2, x3, x4, x5, x6, x7, x8, x9]
        elseif Nx == 10
            @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10
            return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
        elseif Nx == 11
            @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11
            return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]
        elseif Nx == 12
            @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12
            return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12]
        else
            if Nx == 13
                @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13
                return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13]
            elseif Nx == 14
                @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14
                return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]
            elseif Nx == 15
                @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15
                return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15]
            elseif Nx == 16
                @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16
                return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16]
            elseif Nx == 17
                @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17
                return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17]
            elseif Nx == 18
                @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18
                return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18]
            elseif Nx == 19
                @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19
                return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19]
            elseif Nx == 20
                @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20
                return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20]
            elseif Nx == 21
                @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21
                return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21]
                if Nx == 22
                    @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22
                    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22]
                elseif Nx == 23
                    @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23
                    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23]
                elseif Nx == 24
                    @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24
                    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24]
                elseif Nx == 25
                    @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25
                    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25]
                elseif Nx == 26
                    @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26
                    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26]
                elseif Nx == 27
                    @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27
                    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27]
                elseif Nx == 28
                    @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28
                    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28]
                elseif Nx == 29
                    @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29
                    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29]
                elseif Nx == 30
                    @variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30
                    return [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30]
                else
                    dddhdhdhd
                end
            end
        end
    end
end

