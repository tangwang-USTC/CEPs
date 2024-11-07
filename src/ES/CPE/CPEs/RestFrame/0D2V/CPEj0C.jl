

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

  Outputs:
    CPEj0C!(out, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,
            rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)

"""

# nMode = 1

# nMode = 2
function CPEj0C!(out::AbstractVector{T}, x::AbstractVector{T};
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0],
    rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T}

    # vec = 1:nMod - 1
    vec = 1
    nh[vec] = x[1]
    uh[vec] = x[2]
    vhth[vec] = x[3]

    vhth2[vec] = vhth[vec].^2 
    uvth2[vec] = (uh[vec]).^2 ./ vhth2[vec]              # uh .^ 2 ./ vhth .^ 2
    
    # nMod = 1
    nj = 1
    # j = 0
    Mhr0 = Mhst[nj] - nh[vec]

    nj += 1
    j = 2
    Orj0 = CjLk(j,1) * uvth2
    Mr2 = (nh[vec] .* vhth2[vec]) .* (1 .+ Orj0[1])
    Mh2 = (Mhst[nj] - Mr2) / Mhr0

    nj += 1
    j = 4
    N = 2
    Orj0[1] = Orj0Nb(uvth2[vec],j,N;rtol_OrjL=rtol_OrjL)
    Mr4 = (nh[vec] .* vhth2[vec].^N) .* (1 .+ Orj0[1])
    Mh4 = (Mhst[nj] - Mr4) / Mhr0

    uhr2 = (2.5 * Mh2^2 - 1.5 * Mh4)^0.5

    # uh[nMod]
    uh[nMod] = (uhr2)^0.5
    # nh[nMod]
    if abs(Mhr0) ≤ rtol_Mh
        # nh[nMod] = 0.0
        nh[nMod] = Mhr0
    else
        nh[nMod] = Mhr0
    end
    vhth[nMod] = (2/3 * Mh2 - uhr2)^0.5

    vhth2[nMod] = vhth[nMod].^2 
    uvth2[nMod] = (uh[nMod]).^2 ./ vhth2[nMod]

    for s in 1:3
        nj += 1
        j += 2
        N = j / 2 |> Int
        Orj0Nb!(Orj0,uvth2,j,N,1;rtol_OrjL=rtol_OrjL)
        out[nj-3] = sum((nh .* vhth2.^N) .* (1 .+ Orj0)) - Mhst[nj]
    end
end

# nMode ≥ 3
function CPEj0C!(out::AbstractVector{T}, x::AbstractVector{T}, nMod::Int;
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0],
    rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T}

    # @show x
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
    Mhr0 = Mhst[nj] - sum_kbn(nh[vec])

    nj += 1
    j = 2
    Orj0 = CjLk(j,1) * uvth2
    # Mr2 = sum_kbn((nh[vec] .* vhth2[vec]) .* (1 .+ Orj0[vec]))
    Mh2 = Mhst[nj] - sum_kbn((nh[vec] .* vhth2[vec]) .* (1 .+ Orj0[vec]))

    
    # Mr2 = sum_kbn(nh[vec] .* (vhth2[vec] + CjLk(j,1) * uh[vec].^2))
    # Mh2 = (Mhst[nj] - Mr2)
    # @show nj, Mhst[nj]
    # @show Mr2
    # @show Mh2
    # @show nh

    nj += 1
    j = 4
    N = 2
    Orj0Nb!(Orj0,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
    # Mr4 = sum((nh[vec] .* vhth2[vec].^N) .* (1 .+ Orj0[vec]))
    Mh4 = Mhst[nj] - sum((nh[vec] .* vhth2[vec].^N) .* (1 .+ Orj0[vec]))
    # @show Orj0
    # @show nj, Mhst[nj],sum(Mhr4s)
    # @show Mhr4s
    # @show Mr4
    # @show Mhr0,Mh2,Mh4

    # @show Mh2 / Mhr0, Mh4 / Mhr0
    uhr2 = 1.5 * (2.5 * ((Mh2 / Mhr0)^2 - (Mh4 / Mhr0)))^0.5

    # uh[nMod]
    uh[nMod] = (uhr2)^0.5
    # nh[nMod]
    if abs(Mhr0) ≤ rtol_Mh
        # nh[nMod] = 0.0
        nh[nMod] = Mhr0
    else
        nh[nMod] = Mhr0
    end
    vhth[nMod] = ((Mh2 / Mhr0) - 2/3 * uhr2)^0.5

    vhth2[nMod] = vhth[nMod].^2 
    uvth2[nMod] = (uh[nMod]).^2 ./ vhth2[nMod]

    for k in 1:nMod-1
        for s in 1:3
            nj += 1
            j += 2
            N = j / 2 |> Int
            Orj0Nb!(Orj0,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
            out[nj-3] = sum((nh .* vhth2.^N) .* (1 .+ Orj0)) - Mhst[nj]
        end
    end
end


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
    J  = zeros(3,3)
    DMrjL = zeros(T,3)
    VnrL = zeros(T,3)
    M1jL = zeros(T,3)                  # = [M10L, M12L, M14L]
    vth1jL = zeros(T,nMod-1) 
    O1jL2 = zeros(T,3nMod-3)
    O1jL = zeros(T,3nMod-3)
    OrnL2 = zeros(T,3)                 # = [Or0L2, Or2L2, Or4L2]
    OrnL = zeros(T,3)                  # = [Or0L, Or2L, Or4L]

  Outputs:
    JacobCPEj0C!(JM, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                DMjL=DMjL,J=J,DMrjL=DMrjL,VnrL=VnrL,M1jL=M1jL,vth1jL=vth1jL,
                O1jL2=O1jL2,O1jL=O1jL,OrnL2=OrnL2,OrnL=OrnL,
                rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
"""

# The Jacobian matrix: JM

# nMode ≥ 3
function JacobCPEj0C!(JM::AbstractArray{T,N2}, x::AbstractVector{T}, nMod::Int; 
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0],DMjL::AbstractVector{T}=[0.0,0.0,0.0],
    J::AbstractArray{T,N2}=[0.0 0.0;0.0 0.0],DMrjL::AbstractVector{T}=[0.0,0.0,0.0],
    VnrL::AbstractVector{T}=[0.0,0.0,0.0],M1jL::AbstractVector{T}=[0.0,0.0,0.0],vth1jL::AbstractVector{T}=[0.0,0.0,0.0],
    O1jL2::AbstractVector{T}=[0.0,0.0,0.0],O1jL::AbstractVector{T}=[0.0,0.0,0.0],
    OrnL2::AbstractVector{T}=[0.0,0.0,0.0],OrnL::AbstractVector{T}=[0.0,0.0,0.0],
    rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T,N2}

    fill!(JM, 0.0)

    vec = 1:nMod-1
    nh[vec] = x[1:3:end]
    uh[vec] = x[2:3:end]
    vhth[vec] = x[3:3:end]

    vhth2[vec] = vhth[vec].^2 
    uvth2[vec] = (uh[vec]).^2 ./ vhth2[vec]  

    nj = 0
    j = 4
    N = j / 2 |> Int
    for k in vec 
        for i in 1:3
            nj += 1
            j += 2
            N += 1
            vth1jL[nj] = vhth2[nMod]^N
            O1jL2[nj],O1jL[nj] = Orj0N2Nb(uvth2[nMod],j,N;rtol_OrjL=rtol_OrjL)
        end
    end

    for s in vec
        # JMC[:,s] = ∂\∂cs (MhjL[:])

        VnrL[:] = [1.0, nh[s] / uh[s], nh[s] / vhth[s]]
        sn = 3(s - 1)
        sn1 = sn + 1
        sn2 = sn + 2
        sn3 = sn + 3

        nj = 1
        j = 0  # 0
        OrnL2[nj],OrnL[nj] = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
        nj = 2
        j = 2
        OrnL2[nj],OrnL[nj] = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
        nj = 3
        j = 4
        OrnL2[nj],OrnL[nj] = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)

        nj = 0
        j = 4
        for k in vec
            for i in 1:3
                nj += 1
                j += 2
                N = j / 2 |> Int
                # vthijL = vhth2[s]^N
                # vth1jL = vhth2[nMod]^N
                OrjL2,Orj0 = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
                ddcMhj0C0D2V!(DMjL,J,DMrjL,nh[s],uh[s],vhth[s],uh[nMod],vhth[nMod],VnrL,M1jL,
                       O1jL2[nj],O1jL[nj],OrjL2,Orj0,OrnL2,OrnL,j;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
                JM[nj, sn1] = DMjL[1]
                JM[nj, sn2] = DMjL[2]
                JM[nj, sn3] = DMjL[3]
            end
        end
    end
end



        