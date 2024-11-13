"""
  Conservation-constraits (ccs) of order `j ∈ {L,L+2,L+4}` for cCPEs.
"""

"""
  `x₁ₗ = [n̂₁ₗ,û₁ₗ,v̂th₁ₗ] = [nh[nMod],uh[nMod],vhth[nMod]]`

  Inputs:
  Outputs:
    ccsj0!(nh,uh,vhth,vhth2,uvth2,Mhst,nMod)
"""

function ccsj0!(nh::AbstractVector{T},uh::AbstractVector{T},vhth::AbstractVector{T},
    vhth2::AbstractVector{T},uvth2::AbstractVector{T},Mhst::AbstractVector{T},nMod::Int) where {T <: Real}

    vec = 1:nMod-1
    vhth2[vec] = vhth[vec].^2 
    uvth2[vec] = (uh[vec]).^2 ./ vhth2[vec]              # uh .^ 2 ./ vhth .^ 2
    
    # nMod = 1
    nj = 1
    # j = 0
    Mhr0 = Mhst[nj] - sum_kbn(nh[vec])

    nj += 1
    j = 2
    Orj0 = CjLk(j,1) * uvth2[vec]
    Mh2 = Mhst[nj] - sum_kbn((nh[vec] .* vhth2[vec]) .* (1 .+ Orj0))

    nj += 1
    j = 4
    N = 2
    Orj0Nb!(Orj0,uvth2[vec],j,N,nMod-1;rtol_OrjL=rtol_OrjL)
    Mh4 = Mhst[nj] - sum_kbn((nh[vec] .* vhth2[vec].^N) .* (1 .+ Orj0))
    
    uhr2 = 1.5 * (2.5 * ((Mh2 / Mhr0)^2 - (Mh4 / Mhr0)))^0.5

    uh[nMod] = (uhr2)^0.5
    if abs(Mhr0) ≤ rtol_Mh
        nh[nMod] = Mhr0
    else
        nh[nMod] = Mhr0
    end
    vhth[nMod] = ((Mh2 / Mhr0) - 2/3 * uhr2)^0.5
end
