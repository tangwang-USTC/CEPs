"""
  `j - L ∈ 2N⁺ - 2`

  Inputs:
    `N = (j - L) / 2 |> Int`
  
  Outputs:
    OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
    OrjL = OrjLNb(uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
"""

function OrjLNb!(OrjL::AbstractVector{T},uvth2::AbstractVector{T},
    j::Int,L::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == L
        OrjL[:] .= 0.0
    elseif j == L + 2
        OrjL[:] =  CjLk(j,L,1) * uvth2
    else
        OrjL[:] = CjLk(j,L,1) * uvth2
        CjLks = CjLk.(j,L,1:N)
        for i in 1:nMod
            for k in 2:N
                ak = CjLks[k] * uvth2[i]^k
                OrjL[i] += ak
                if ak ≤ rtol_OrjL
                    break
                end
            end
        end
        return a
    end
end

function OrjLNb(uvth2::T,j::Int,L::Int,N::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == L
        return 0.0
    elseif j == L + 2
        return CjLk(j,L,1) * uvth2
    else
        a = CjLk(j,L,1) * uvth2
        for k in 2:N
            ak = CjLk(j,L,k) * uvth2.^k
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
    OrjLN2b!(OrjL2,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
    OrjL2 = OrjLN2b(uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
"""

function OrjLN2b!(OrjL2::AbstractVector{T},uvth2::AbstractVector{T},
    j::Int,L::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == L
        OrjL2[:] .= 0.0
    elseif j == L + 2
        OrjL2[:] = 2 * CjLk(j,L,1) * uvth2
    else
        OrjL2[:] = 2 * CjLk(j,L,1) * uvth2
        CjLks = CjLk.(j,L,1:N)
        for i in 1:nMod
            for k in 2:N
                ak = CjLks[k] * uvth2[i]^k
                OrjL2[i] += 2k * ak
                if ak ≤ rtol_OrjL
                    break
                end
            end
        end
    end
end

function OrjLN2b(uvth2::T,j::Int,L::Int,N::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == L
        return 0.0
    elseif j == L + 2
        return 2 * CjLk(j,L,1) * uvth2
    else
        a = 2 * CjLk(j,L,1) * uvth2
        for k in 2:N
            ak = CjLk(j,L,k) * uvth2.^k
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
    OrjLN2Nb!(OrjL2,OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
    OrjL2,OrjL = OrjLN2Nb(uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
"""

function OrjLN2Nb!(OrjL2::AbstractVector{T},OrjL::AbstractVector{T},uvth2::AbstractVector{T},
    j::Int,L::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == L
        OrjL[:] .= 0.0
        OrjL2[:] .= 0.0
    elseif j == L + 2
        OrjL[:] = CjLk(j,L,1) * uvth2
        OrjL2[:] = 2 * OrjL[:]
    else
        OrjL[:] = CjLk(j,L,1) * uvth2
        OrjL2[:] = 2 * OrjL[:]
        CjLks = CjLk.(j,L,1:N)
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
        return 0.0, 0.0
    elseif j == L + 2
        a = CjLk(j,L,1) * uvth2
        return 2a, a
    else
        a = CjLk(j,L,1) * uvth2
        OrjL2 = 2a
        for k in 2:N
            ak = CjLk(j,L,k) * uvth2.^k
            a += ak
            OrjL2 += 2k * ak
            if ak ≤ rtol_OrjL
                break
            end
        end
        return OrjL2, a
    end
end
