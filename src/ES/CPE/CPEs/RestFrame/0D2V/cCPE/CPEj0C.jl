

"""
  Characteristic parameter equations (CPEs) when `L = 0`
    for weakly anisotropic and moderate anisotropic plasma system 
    with general axisymmetric velocity space. 
    The plasma is in a local sub-equilibrium state.

    This version is based on the conservation-constrait CPEs (CPEsC)
    where CPEs with order `j ∈ {L,L+2,L+4}` are enforced by the algorithm.

  The general kinetic moments are renormalized by `CMjL`.
    
    If `is_renorm == true`,
      `Mhst = 𝓜ⱼ(f̂ₗ) / CMjL`
    end
 
  Notes: `{M̂₁₁}/3 = Î = n̂ û`, generally.
  
  See Ref. of Wang (2025) titled as "StatNN: An inherently interpretable physics-informed neural networks designed for moderate anisotropic plasma simulation".
  
"""

"""
  CPEs

  Inputs:
    out = zeros(nMod-1,nMod-1)
    x = x(3nMod-3)
    nh = nai
    uh = uai
    vhth = vthi
    Mhst = M̂ⱼₗ*, which is the renormalized general kinetic moments.
    M1jL := [M1LL, RM12L, RM14L]

  Outputs:
    out = CPEj0C(x,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
            rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)

"""

# nMod ≥ 3
function CPEj0C!(out::AbstractVector{T}, x::AbstractVector{T}, nMod::Int;
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0],
    M1jL::AbstractVector{T}=[1.0,1.0,1.0],
    rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T}

    if nMod == 2
        return CPEj0C(x;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
              rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    else
      nMod1 = nMod - 1
      vec = 1:nMod1
      nh[vec] = x[1:3:end]
      uh[vec] = x[2:3:end]
      vhth[vec] = x[3:3:end]
  
      vhth2[vec] = vhth[vec].^2 
      uvth2[vec] = (uh[vec]).^2 ./ vhth2[vec]              # uh .^ 2 ./ vhth .^ 2
      
      # nMod = 1
      nj = 1
      # j = 0
      M1jL[1] = Mhst[nj] - sum_kbn(nh[vec])
  
      nj += 1
      j = 2
      Orj0 = CjLk(T(j),T(1)) * uvth2
      Mr2 = sum_kbn((nh[vec] .* vhth2[vec]) .* (1 .+ Orj0[vec]))
      M1jL[2] = (Mhst[nj] - Mr2) / M1jL[1]               # RM12L
  
      nj += 1
      j = 4
      N = 2
      Orj0Nb!(Orj0,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
      Mr4 = sum_kbn((nh[vec] .* vhth2[vec].^N) .* (1 .+ Orj0[vec]))
      M1jL[3] = (Mhst[nj] - Mr4) / M1jL[1]               # RM14L
      
      # uhr4
      uhr2 = M1jL[2]^2 - M1jL[3]
      uhr2 = 1.5 * (2.5 * uhr2)^0.5
  
      uh[nMod] = (uhr2)^0.5
      if abs(M1jL[1]) ≤ rtol_Mh
          # nh[nMod] = 0.0
          nh[nMod] = M1jL[1]
      else
          nh[nMod] = M1jL[1]
      end
      vhth2[nMod] = M1jL[2] - T(2)/3 * uhr2
      vhth[nMod] = (vhth2[nMod])^0.5
      uvth2[nMod] = uhr2 ./ vhth2[nMod]
  
      for k in 1:nMod-1
          for s in 1:3
              nj += 1
              j += 2
              N = j / 2 |> Int
              Orj0Nb!(Orj0,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
              out[nj-3] = sum_kbn((nh .* vhth2.^N) .* (1 .+ Orj0)) - Mhst[nj]
          end
      end
      return out
    end
end

# nMod = 2
function CPEj0C!(out::AbstractVector{T}, x::AbstractVector{T};
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0],
    M1jL::AbstractVector{T}=[1.0,1.0,1.0],
    rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T}

    vec = 1
    nh[vec] = x[1]
    uh[vec] = x[2]
    vhth[vec] = x[3]

    vhth2[vec] = vhth[vec]^2 
    uvth2[vec] = (uh[vec])^2 / vhth2[vec]              # uh .^ 2 ./ vhth .^ 2
    
    # nMod = 1
    nj = 1
    # j = 0
    M1jL[1] = Mhst[nj] - sum_kbn(nh[vec])

    nj += 1
    j = 2
    Orj0 = CjLk(T(j),T(1)) * uvth2
    Mr2 = (nh[vec] * vhth2[vec]) * (1 + Orj0[vec])
    M1jL[2] = (Mhst[nj] - Mr2) / M1jL[1]               # RM12L

    nj += 1
    j = 4
    N = 2
    Orj0Nb!(Orj0,uvth2,j,N,2;rtol_OrjL=rtol_OrjL)
    Mr4 = (nh[vec] * vhth2[vec]^N) * (1 + Orj0[vec])
    M1jL[3] = (Mhst[nj] - Mr4) / M1jL[1]               # RM14L
    
    # uhr4
    uhr2 = M1jL[2]^2 - M1jL[3]
    uhr2 = 1.5 * (2.5 * uhr2)^0.5

    uh[2] = (uhr2)^0.5
    if abs(M1jL[1]) ≤ rtol_Mh
        # nh[2] = 0.0
        nh[2] = M1jL[1]
    else
        nh[2] = M1jL[1]
    end
    vhth2[2] = M1jL[2] - T(2)/3 * uhr2
    vhth[2] = (vhth2[2])^0.5
    uvth2[2] = uhr2 / vhth2[2]

    for s in 1:3
        nj += 1
        j += 2
        N = j / 2 |> Int
        Orj0Nb!(Orj0,uvth2,j,N,2;rtol_OrjL=rtol_OrjL)
        out[nj-3] = sum_kbn((nh .* vhth2.^N) .* (1 .+ Orj0)) - Mhst[nj]
    end
    return out
end

# nMod = 1

