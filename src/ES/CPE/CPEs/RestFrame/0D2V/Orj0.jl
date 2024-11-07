"""
  `j - L ∈ 2N⁺ - 2`

  Inputs:
    `N = (j - L) / 2 |> Int`
  
  Outputs:
    Orj0Nb!(OrjL,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
    OrjL = Orj0Nb(uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
"""

function Orj0Nb!(OrjL::AbstractVector{T},uvth2::AbstractVector{T},
    j::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == 0
        OrjL[:] .= 0.0
    elseif j == 2
        OrjL[:] =  CjLk(j,1) * uvth2
    else
        OrjL[:] = CjLk(j,1) * uvth2
        CjLks = CjLk.(j,1:N)
        for i in 1:nMod
            for k in 2:N
                ak = CjLks[k] * uvth2[i]^k
                OrjL[i] += ak
                if ak ≤ rtol_OrjL
                    break
                end
            end
        end
    end
end

function Orj0Nb(uvth2::T,j::Int,N::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == 0
        return 0.0
    elseif j == 2
        return CjLk(j,1) * uvth2
    else
        a = CjLk(j,1) * uvth2
        for k in 2:N
            ak = CjLk(j,k) * uvth2.^k
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
    OrjLN2b!(OrjL2,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
    OrjL2 = OrjLN2b(uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
"""

function Orj0N2b!(OrjL2::AbstractVector{T},uvth2::AbstractVector{T},
    j::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == 0
        OrjL2[:] .= 0.0
    elseif j == 2
        OrjL2[:] = 2 * CjLk(j,1) * uvth2
    else
        OrjL2[:] = 2 * CjLk(j,1) * uvth2
        CjLks = CjLk.(j,1:N)
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

function Orj0N2b(uvth2::T,j::Int,N::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == 0
        return 0.0
    elseif j == 2
        return 2 * CjLk(j,1) * uvth2
    else
        a = 2 * CjLk(j,1) * uvth2
        for k in 2:N
            ak = CjLk(j,k) * uvth2.^k
            a += 2k * ak
            if ak ≤ rtol_OrjL
                break
            end
        end
        return a
    end
end

"""
  `j - L ∈ 2N⁺ - 2` where `L = 0`

  Inputs:
    `N = (j - L) / 2 |> Int`
  
  Outputs:
    Orj0N2Nb!(Orj02,Orj0,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
    Orj02,Orj0 = Orj0N2Nb(uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
"""

function Orj0N2Nb!(Orj02::AbstractVector{T},Orj0::AbstractVector{T},uvth2::AbstractVector{T},
    j::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == 0
        Orj0[:] .= 0.0
        Orj02[:] .= 0.0
    elseif j == 2
        Orj0[:] = CjLk(j,1) * uvth2
        Orj02[:] = 2 * Orj0[:]
    else
        Orj0[:] = CjLk(j,1) * uvth2
        Orj02[:] = 2 * Orj0[:]
        Cj0ks = CjLk.(j,1:N)
        for i in 1:nMod
            for k in 2:N
                ak = Cj0ks[k] * uvth2[i]^k
                Orj0[i] += ak
                Orj02[i] += 2k * ak
                if ak ≤ rtol_OrjL
                    break
                end
            end
        end
    end
end

function Orj0N2Nb(uvth2::T,j::Int,N::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == 0
        return 0.0, 0.0
    elseif j == 2
        a = CjLk(j,1) * uvth2
        return 2a, a
    else
        a = CjLk(j,1) * uvth2
        Orj02 = 2a
        for k in 2:N
            ak = CjLk(j,k) * uvth2.^k
            a += ak
            Orj02 += 2k * ak
            if ak ≤ rtol_OrjL
                break
            end
        end
        return Orj02, a
    end
end
