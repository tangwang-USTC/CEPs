"""
  `j - L ∈ 2N⁺ - 2`

  Inputs:
    `N = (j - L) / 2 |> Int`
  
  Outputs:
    Orj0Nb!(OrjL,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
"""

function Orj0Nb!(OrjL::AbstractVector{Num},uvth2::AbstractVector{Num},
    j::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == 0
        sedfvbbbb
        OrjL[:] .= 0.0
    elseif j == 2
        OrjL[:] =  CjLk(T(j),T(1)) * uvth2
    else
        OrjL[:] = CjLk(T(j),T(1)) * uvth2
        CjLks = CjLk.(T(j),T.(1:N))
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


"""
  `j - L ∈ 2N⁺ - 2`

  Inputs:
    `N = (j - L) / 2 |> Int`
  
  Outputs:
    OrjLN2b!(OrjL2,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
"""

function Orj0N2b!(OrjL2::AbstractVector{Num},uvth2::AbstractVector{Num},
    j::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == 0
        OrjL2[:] .= 0.0
    elseif j == 2
        OrjL2[:] = 2 * CjLk(T(j),T(1)) * uvth2
    else
        OrjL2[:] = 2 * CjLk(T(j),T(1)) * uvth2
        CjLks = CjLk.(T(j),T.(1:N))
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

"""
  `j - L ∈ 2N⁺ - 2` where `L = 0`

  Inputs:
    `N = (j - L) / 2 |> Int`
  
  Outputs:
    Orj0N2Nb!(Orj02,Orj0,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
"""

function Orj0N2Nb!(Orj02::AbstractVector{Num},Orj0::AbstractVector{Num},uvth2::AbstractVector{Num},
    j::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == 0
        sedfvbbbb
        Orj0[:] .= 0.0
        Orj02[:] .= 0.0
    elseif j == 2
        Orj0[:] = CjLk(T(j),T(1)) * uvth2
        Orj02[:] = 2 * Orj0[:]
    else
        Cj0ks = CjLk.(j,1:N)
        k = 1
        Orj0[:] = Cj0ks[k] * uvth2
        Orj02[:] = 2 * Orj0
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
