"""
  `j - L ∈ 2N⁺ - 2`

  Inputs:
    `N = (j - L) / 2 |> Int`
  
  Outputs:
    OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
"""

function OrjLNb!(OrjL::AbstractVector{Num},uvth2::AbstractVector{Num},
    j::Int,L::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == L
        sedfvbbbb
        # OrjL[:] .= 0.0
    elseif j == L + 2
        OrjL[:] =  CjLk(T(j),T(L),T(1)) * uvth2
    elseif j == L + 4
        CjLks = CjLk.(T(j),T(L),T.(1:2))
        OrjL[:] =  CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2
    else
        CjLks = CjLk.(T(j),T(L),T.(1:N))
        OrjL[:] = CjLks[1] * uvth2
        for i in 1:nMod
            for k in 2:N
                ak = CjLks[k] * uvth2[i]^k
                OrjL[i] += ak
                # if ak ≤ rtol_OrjL
                #     break
                # end
            end
        end
    end
end

"""
  `j - L ∈ 2N⁺ - 2`

  Inputs:
    `N = (j - L) / 2 |> Int`
  
  Outputs:
    OrjLN2b!(OrjL2,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
"""

function OrjLN2b!(OrjL2::AbstractVector{Num},uvth2::AbstractVector{Num},
    j::Int,L::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == L
        sedfvbbbb33333
        # OrjL[:] .= 0.0
    elseif j == L + 2
        OrjL2[:] = 2 * CjLk(T(j),T(L),T(1)) * uvth2
    elseif j == L + 4
        CjLks = CjLk.(T(j),T(L),T.(1:2))
        OrjL[:] =  CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2
    else
        CjLks = CjLk.(T(j),T(L),T.(1:N))
        OrjL2[:] = 2 * CjLks[1] * uvth2
        for i in 1:nMod
            for k in 2:N
                ak = CjLks[k] * uvth2[i]^k
                OrjL2[i] += 2k * ak
                # if ak ≤ rtol_OrjL
                #     break
                # end
            end
        end
    end
end

"""
  `j - L ∈ 2N⁺ - 2`

  Inputs:
    `N = (j - L) / 2 |> Int`
  
  Outputs:
    OrjLN2Nb!(OrjL2,OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
"""

function OrjLN2Nb!(OrjL2::AbstractVector{Num},OrjL::AbstractVector{Num},uvth2::AbstractVector{Num},
    j::Int,L::Int,N::Int,nMod::Int;rtol_OrjL::T=1e-10) where {T<:Real}
    
    if j == L
        sedfvbbbb66666
        OrjL[:] .= 0.0
        OrjL2[:] .= 0.0
    elseif j == L + 2
        OrjL[:] = CjLk(T(j),T(L),T(1)) * uvth2
        OrjL2[:] = 2 * OrjL[:]
    else
        CjLks = CjLk.(T(j),T(L),T.(1:N))
        OrjL[:] = CjLks[1] * uvth2
        OrjL2[:] = 2 * OrjL[:]
        for i in 1:nMod
            for k in 2:N
                ak = CjLks[k] * uvth2[i]^k
                OrjL[i] += ak
                OrjL2[i] += 2k * ak
                # if ak ≤ rtol_OrjL
                #     break
                # end
            end
        end
    end
end
