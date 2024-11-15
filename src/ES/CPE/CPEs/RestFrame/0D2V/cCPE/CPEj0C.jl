

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
    CPEj0C!(out, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
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
      CPEj0C!(out, x;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
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
    end
end
  
# @show nh, uh, vhth
# @show naiLt0, uaiLt0, vthiLt0
# @show nh ./ naiLt0 .- 1
# @show uh ./ uaiLt0 .- 1
# @show vhth ./ vthiLt0 .- 1

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
end

# nMod = 1



"""
  The first part of Jacobian arising from the `nModᵗʰ` component can be expressed as:

        `JM = JMC = 𝕁ₘᶜ = zeros(nMod-1,nMod-1)
        
  Inputs:
    JM = zeros(3nMod-3,3nMod-3)
    x = x(3nMod-3)
    nh = nai[1:nMod]
    uh = uai[1:nMod]
    vhth = vthi[1:nMod]
    DMjL = zeros(T,3)
    J = zeros(3,3)
    DM1RjL = zeros(T,3)
    vth1jL = zeros(T,3nMod-3) 
    vthijL = zeros(T,4)                # [vthiLL,vthi2L,vthi4L,vthijL]
    Vn1L = zeros(T,3)
    VnrL = zeros(T,3)
    M1jL := [M1LL, RM12L, RM14L]
    crjL := zeros(T,3)
    arjL := zeros(T,3)
    O1jL2 = zeros(T,3nMod-3)
    O1jL = zeros(T,3nMod-3)
    OrnL2 = zeros(T,3)                 # = [Or0L2, Or2L2, Or4L2]
    OrnL = zeros(T,3)                  # = [Or0L, Or2L, Or4L]

  Outputs:
    JacobCPEj0C!(JM, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                DMjL=DMjL,J=J,DM1RjL=DM1RjL,vth1jL=vth1jL,vthijL=vthijL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,
                crjL=crjL,arLL=arLL,ar2L=ar2L,ar4L=ar4L,O1jL2=O1jL2,O1jL=O1jL,
                OrnL2=OrnL2,OrnL=OrnL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
"""

# The Jacobian matrix: JM

# nMod ≥ 3
function JacobCPEj0C!(JM::AbstractArray{T,N2}, x::AbstractVector{T}, nMod::Int; 
  nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
  vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
  uvth2::AbstractVector{T}=[0.1, 1.0],DMjL::AbstractVector{T}=[0.0,0.0,0.0],
  J::AbstractArray{T,N2}=[0.0 0.0;0.0 0.0],DM1RjL::AbstractVector{T}=[0.0,0.0,0.0],
  vth1jL::AbstractVector{T}=[0.0,0.0,0.0],vthijL::AbstractVector{T}=[0.0,0.0,0.0],
  Vn1L::AbstractVector{T}=[0.0,0.0,0.0],VnrL::AbstractVector{T}=[0.0,0.0,0.0],M1jL::AbstractVector{T}=[1.0,1.0,1.0],
  crjL::AbstractVector{T}=[0.0,0.0,0.0],arLL::AbstractVector{T}=[0.0,0.0,0.0],
  ar2L::AbstractVector{T}=[0.0,0.0,0.0],ar4L::AbstractVector{T}=[0.0,0.0,0.0],
  O1jL2::AbstractVector{T}=[0.0,0.0,0.0],O1jL::AbstractVector{T}=[0.0,0.0,0.0],
  OrnL2::AbstractVector{T}=[0.0,0.0,0.0],OrnL::AbstractVector{T}=[0.0,0.0,0.0],
  rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T,N2}

  if nMod == 2
    JacobCPEj0C!(JM, x;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
            DMjL=DMjL,J=J,DM1RjL=DM1RjL,vth1jL=vth1jL,vthijL=vthijL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,
            crjL=crjL,arLL=arLL,ar2L=ar2L,ar4L=ar4L,O1jL2=O1jL2,O1jL=O1jL,
            OrnL2=OrnL2,OrnL=OrnL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
  else
    fill!(JM, 0.0)
    vec13 = 1:3
  
    vec = 1:nMod-1
    nh[vec] = x[1:3:end]
    uh[vec] = x[2:3:end]
    vhth[vec] = x[3:3:end]
  
    vhth2[vec] = vhth[vec].^2 
    uvth2[vec] = (uh[vec]).^2 ./ vhth2[vec]  
  
    nj = 0
    j = 4
    N = 2 |> Int
    for k in vec 
        for i in vec13
            nj += 1
            j += 2
            N += 1
            vth1jL[nj] = vhth2[nMod]^N
            O1jL2[nj],O1jL[nj] = Orj0N2Nb(uvth2[nMod],j,N;rtol_OrjL=rtol_OrjL)
        end
    end
  
    Vn1L[:] = [1.0, nh[nMod] / uh[nMod], nh[nMod] / vhth[nMod]]
    for s in vec
        # JMC[:,s] = ∂\∂cs (MhjL[:])
        vthijL[vec13] = [1.0, vhth2[s], vhth2[s]^2]
        VnrL[:] = [1.0, nh[s] / uh[s], nh[s] / vhth[s]]
        sn = 3(s - 1) .+ vec13
  
        nj = 1
        j = 0  # 0
        N = 0
        OrnL2[nj],OrnL[nj] = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
        nj = 2
        j = 2
        N = 1
        OrnL2[nj],OrnL[nj] = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
        nj = 3
        j = 4
        N = 2
        OrnL2[nj],OrnL[nj] = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
  
        nj = 0
        # j = 4
        for k in vec
            for i in vec13
                nj += 1
                j += 2
                N = j / 2 |> Int
                # vth1jL = vhth2[nMod]^N
                vthijL[4] = vhth2[s]^N
                OrjL2,OrjL = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
                ddcMhj0C0D2V!(DMjL,J,DM1RjL,uh[nMod],vhth[nMod],vth1jL[nj],vthijL,Vn1L,VnrL,
                       M1jL,crjL,arLL,ar2L,ar4L,O1jL2[nj],O1jL[nj],OrjL2,OrjL,OrnL2,OrnL,j)
                JM[nj, sn] = DMjL
            end
        end
    end
  end
  # @show JM
  # @show cond(JM)
end
     

# nMod = 2
function JacobCPEj0C!(JM::AbstractArray{T,N2}, x::AbstractVector{T}; 
  nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
  vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
  uvth2::AbstractVector{T}=[0.1, 1.0],DMjL::AbstractVector{T}=[0.0,0.0,0.0],
  J::AbstractArray{T,N2}=[0.0 0.0;0.0 0.0],DM1RjL::AbstractVector{T}=[0.0,0.0,0.0],
  vth1jL::AbstractVector{T}=[0.0,0.0,0.0],vthijL::AbstractVector{T}=[0.0,0.0,0.0],
  Vn1L::AbstractVector{T}=[0.0,0.0,0.0],VnrL::AbstractVector{T}=[0.0,0.0,0.0],M1jL::AbstractVector{T}=[1.0,1.0,1.0],
  crjL::AbstractVector{T}=[0.0,0.0,0.0],arLL::AbstractVector{T}=[0.0,0.0,0.0],
  ar2L::AbstractVector{T}=[0.0,0.0,0.0],ar4L::AbstractVector{T}=[0.0,0.0,0.0],
  O1jL2::AbstractVector{T}=[0.0,0.0,0.0],O1jL::AbstractVector{T}=[0.0,0.0,0.0],
  OrnL2::AbstractVector{T}=[0.0,0.0,0.0],OrnL::AbstractVector{T}=[0.0,0.0,0.0],
  rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T,N2}

  fill!(JM, 0.0)
  
  vec13 = 1:3
  vec = 1
  nh[vec] = x[1]
  uh[vec] = x[2]
  vhth[vec] = x[3]

  vhth2[vec] = vhth[vec]^2 
  uvth2[vec] = (uh[vec])^2 / vhth2[vec]  

  nj = 0
  j = 4
  N = 2 |> Int
  for k in vec 
      for i in vec13
          nj += 1
          j += 2
          N += 1
          vth1jL[nj] = vhth2[2]^N
          O1jL2[nj],O1jL[nj] = Orj0N2Nb(uvth2[2],j,N;rtol_OrjL=rtol_OrjL)
      end
  end

  Vn1L[:] = [1.0, nh[2] / uh[2], nh[2] / vhth[2]]
  for s in vec
      # JMC[:,s] = ∂\∂cs (MhjL[:])

      vthijL[vec13] = [1.0, vhth2[s], vhth2[s]^2]
      VnrL[:] = [1.0, nh[s] / uh[s], nh[s] / vhth[s]]
      sn = 3(s - 1) .+ vec13

      nj = 1
      j = 0  # 0
      N = 0
      OrnL2[nj],OrnL[nj] = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
      nj = 2
      j = 2
      N = 1
      OrnL2[nj],OrnL[nj] = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
      nj = 3
      j = 4
      N = 2
      OrnL2[nj],OrnL[nj] = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)

      nj = 0
      # j = 4
      for i in vec13
          nj += 1
          j += 2
          N = j / 2 |> Int
          # vth1jL = vhth2[2]^N
          vthijL[4] = vhth2[s]^N
          OrjL2,OrjL = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
          ddcMhj0C0D2V!(DMjL,J,DM1RjL,uh[2],vhth[2],vth1jL[nj],vthijL,Vn1L,VnrL,
                 M1jL,crjL,arLL,ar2L,ar4L,O1jL2[nj],O1jL[nj],OrjL2,OrjL,OrnL2,OrnL,j)
          JM[nj, sn] = DMjL
      end
  end
end

        