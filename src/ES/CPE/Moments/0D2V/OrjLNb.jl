"""
  `j - L ∈ 2N⁺ - 2`

  Inputs:
    `N = (j - L) / 2 |> Int`
  
  Outputs:
    OrjL = OrjLNb(uvth2,j,L,N)
    OrjL = OrjLNb(uvth2,j,L,N;rtol_OrjL=rtol_OrjL)
"""

function OrjLNb(uvth2::AbstractVector{T},j::Int,L::Int,N::Int) where {T<:Real}
    
    if j == L
        return 0.0uvth2
    elseif j == L + 2
        return CjLk(T(j),T(L),T(1)) * uvth2
    else
        CjLks = CjLk.(T(j),T(L),T.(1:N))
        if j == L + 4
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2
        elseif j == L + 6
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2 + CjLks[3] * uvth2 .^3
        elseif j == L + 8
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2 + CjLks[3] * uvth2 .^3 + CjLks[4] * uvth2 .^4
        elseif j == L + 10
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2 + CjLks[3] * uvth2 .^3 + CjLks[4] * uvth2 .^4 + CjLks[5] * uvth2 .^5
        elseif j == L + 12
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2 + CjLks[3] * uvth2 .^3 + CjLks[4] * uvth2 .^4 + CjLks[5] * uvth2 .^5 + CjLks[6] * uvth2 .^6
        elseif j == L + 14
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2 + CjLks[3] * uvth2 .^3 + CjLks[4] * uvth2 .^4 + CjLks[5] * uvth2 .^5 + CjLks[6] * uvth2 .^6 + CjLks[7] * uvth2 .^7
        elseif j == L + 16
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2 + CjLks[3] * uvth2 .^3 + CjLks[4] * uvth2 .^4 + CjLks[5] * uvth2 .^5 + CjLks[6] * uvth2 .^6 + CjLks[7] * uvth2 .^7 + CjLks[8] * uvth2 .^8
        else
            rtrhhh
        end
    end
end

function OrjLNb(uvth2::T,j::Int,L::Int,N::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == L
        return 0.0 |> T
    elseif j == L + 2
        return CjLk(T(j),T(L),T(1)) * uvth2
    else
        a = CjLk(T(j),T(L),T(1)) * uvth2
        for k in 2:N
            ak = CjLk(T(j),T(L),T(k)) * uvth2.^k
            a += ak
            if ak ≤ rtol_OrjL
                break
            end
        end
        return a
    end
end

"""
  `j - L ∈ 2N⁺ - 2`

  Inputs:
    `N = (j - L) / 2 |> Int`
  
  Outputs:
    OrjLN2b(OrjL2,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
    OrjL2 = OrjLN2b(uvth2,j,L,N;rtol_OrjL=rtol_OrjL)
"""

function OrjLN2b(uvth2::AbstractVector{T},j::Int,L::Int,N::Int) where {T<:Real}
    
    if j == L
        return 0.0uvth2
    elseif j == L + 2
        return CjLk2k(T(j),T(L),T(1)) * uvth2
    else
        CjLks = CjLk2k.(T(j),T(L),T.(1:N))
        if j == L + 4
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2
        elseif j == L + 6
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2 + CjLks[3] * uvth2 .^3
        elseif j == L + 8
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2 + CjLks[3] * uvth2 .^3 + CjLks[4] * uvth2 .^4
        elseif j == L + 10
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2 + CjLks[3] * uvth2 .^3 + CjLks[4] * uvth2 .^4 + CjLks[5] * uvth2 .^5
        elseif j == L + 12
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2 + CjLks[3] * uvth2 .^3 + CjLks[4] * uvth2 .^4 + CjLks[5] * uvth2 .^5 + CjLks[6] * uvth2 .^6
        elseif j == L + 14
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2 + CjLks[3] * uvth2 .^3 + CjLks[4] * uvth2 .^4 + CjLks[5] * uvth2 .^5 + CjLks[6] * uvth2 .^6 + CjLks[7] * uvth2 .^7
        elseif j == L + 16
            return CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2 + CjLks[3] * uvth2 .^3 + CjLks[4] * uvth2 .^4 + CjLks[5] * uvth2 .^5 + CjLks[6] * uvth2 .^6 + CjLks[7] * uvth2 .^7 + CjLks[8] * uvth2 .^8
        else
            rtrhhh
        end
    end
end

function OrjLN2b(uvth2::T,j::Int,L::Int,N::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == L
        return 0.0 |> T
    elseif j == L + 2
        return 2 * CjLk(T(j),T(L),T(1)) * uvth2
    else
        a = 2 * CjLk(T(j),T(L),T(1)) * uvth2
        for k in 2:N
            ak = CjLk(T(j),T(L),T(k)) * uvth2.^k
            a += 2k * ak
            if ak ≤ rtol_OrjL
                break
            end
        end
        return a
    end
end

"""
  `j - L ∈ 2N⁺ - 2`

  Inputs:
    `N = (j - L) / 2 |> Int`
  
  Outputs:
    OrjLN2Nb(OrjL2,OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
    OrjL2,OrjL = OrjLN2Nb(uvth2,j,L,N;rtol_OrjL=rtol_OrjL)
"""

function OrjLN2Nb(OrjL2::AbstractVector{T},OrjL::AbstractVector{T},uvth2::AbstractVector{T},
    j::Int,L::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == L
        return 0.0uvth2, 0.0uvth2
    elseif j == L + 2
        OrjL[:] = CjLk(T(j),T(L),T(1)) * uvth2
        OrjL2[:] = 2 * OrjL[:]
    else
        # if j == L + 4
        # elseif j == L + 6
        # else
        # end
        CjLks = CjLk.(T(j),T(L),T.(1:N))
        OrjL[:] = CjLks[1] * uvth2
        OrjL2[:] = 2 * OrjL[:]
        for i in 1:nMod
            for k in 2:N
                ak = CjLks[k] * uvth2[i]^k
                OrjL[i] += ak
                OrjL2[i] += 2k * ak
                if ak ≤ rtol_OrjL
                    break
                end
            end
        end
    end
end

function OrjLN2Nb(uvth2::T,j::Int,L::Int,N::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == L
        return T(0), T(0)
    elseif j == L + 2
        a = CjLk(T(j),T(L),T(1)) * uvth2
        return 2a, a
    else
        a = CjLk(T(j),T(L),T(1)) * uvth2
        OrjL2 = 2a
        for k in 2:N
            ak = CjLk(T(j),T(L),T(k)) * uvth2.^k
            a += ak
            OrjL2 += 2k * ak
            if ak ≤ rtol_OrjL
                break
            end
        end
        return OrjL2, a
    end
end
