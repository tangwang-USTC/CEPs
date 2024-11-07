

"""
  Characteristic parameter equations (CPEs) when `L = 1`
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
    CPEj1C!(out, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,
                rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)

"""

# nMode = 1

# nMode = 2
function CPEj1C!(out::AbstractVector{T}, x::AbstractVector{T};
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

    uhL = ones(T,nMod)
    uhL[vec] = uh[vec]
    
    # nMod = 1
    nj = 1
    # j = L + 0
    Mhr0 = Mhst[nj] - nh[vec] .* uhL[vec]

    nj += 1
    j = 3          # L + 2
    OrjL = CjLk(j,1,1) * uvth2
    Mr2 = (nh[vec] .* vhth2[vec] .* uhL[vec]) .* (1 .+ OrjL[1])
    Mh2 = Mhst[nj] - Mr2

    nj += 1
    j = 5
    N = 2
    OrjL[1] = OrjLNb(uvth2[vec],j,1,N;rtol_OrjL=rtol_OrjL)
    Mr4 = (nh[vec] .* vhth2[vec].^N .* uhL[vec]) .* (1 .+ OrjL[1])
    Mh4 = Mhst[nj] - Mr4

    uhr2 = (3.5 * Mh2^2 - 2.5 * Mh4)^0.5

    # uh[nMod]
    uh[nMod] = sign(Mhr0) * (uhr2)^0.5
    # nh[nMod]
    if abs(Mhr0) ≤ rtol_Mh
        if abs(uh[nMod]) ≤ rtol_Mh
            nh[nMod] = 0.0
        else
            nh[nMod] = sign(Mhr0) * Mhr0 / uhr2 ^(1/2)
        end
    else
        if abs(uh[nMod]) ≤ rtol_Mh
            @error(`Error: nhr ∝ MhrjL / uhr^L. Both MhrjL and uhr are almost zero!! L=1`)
            nh[nMod] = sign(Mhr0) * Mhr0 / uhr2 ^(1/2)
        else
            nh[nMod] = sign(Mhr0) * Mhr0 / uhr2 ^(1/2)
        end
    end
    vhth[nMod] = (0.4 * Mh2 - uhr2)^0.5

    vhth2[nMod] = vhth[nMod].^2 
    uvth2[nMod] = (uh[nMod]).^2 ./ vhth2[nMod]
    uhL[nMod] = uh[nMod]

    for s in 1:3
        nj += 1
        j += 2
        N = (j - 1) / 2 |> Int
        OrjLNb!(OrjL,uvth2,j,1,N,nMod;rtol_OrjL=rtol_OrjL)
        out[nj-3] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    end
end

# nMode ≥ 3
function CPEj1C!(out::AbstractVector{T}, x::AbstractVector{T}, nMod::Int;
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0],
    rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T}

    vec = 1:nMod-1
    nh[vec] = x[1:3:end]
    uh[vec] = x[2:3:end]
    vhth[vec] = x[3:3:end]

    vhth2[vec] = vhth[vec].^2 
    uvth2[vec] = (uh[vec]).^2 ./ vhth2[vec]              # uh .^ 2 ./ vhth .^ 2

    uhL = uh
    
    # nMod = 1
    nj = 1
    # j = L + 0
    Mhr0 = Mhst[nj] - sum((nh[vec] .* uhL[vec]))

    nj += 1
    j = 2
    OrjL = CjLk(j,1,1) * uvth2
    Mr2 = sum((nh[vec] .* vhth2[vec] .* uhL[vec]) .* (1 .+ OrjL[vec]))
    Mh2 = Mhst[nj] - Mr2

    nj += 1
    j = 5
    N = 2
    OrjL[vec] = OrjLNb(uvth2[vec],j,1,N;rtol_OrjL=rtol_OrjL)
    Mr4 = sum((nh[vec] .* vhth2[vec].^N .* uhL[vec]) .* (1 .+ OrjL[vec]))
    Mh4 = Mhst[nj] - Mr4

    uhr2 = (3.5 * (Mh2) - 2.5 * Mh4)^0.5

    # uh[nMod]
    uh[nMod] = sign(Mhr0) * (uhr2)^0.5
    # nh[nMod]
    if abs(Mhr0) ≤ rtol_Mh
        if abs(uh[nMod]) ≤ rtol_Mh
            nh[nMod] = 0.0
        else
            nh[nMod] = sign(Mhr0) * Mhr0 / uhr2 ^(1/2)
        end
    else
        if abs(uh[nMod]) ≤ rtol_Mh
            @error(`Error: nhr ∝ MhrjL / uhr^L. Both MhrjL and uhr are almost zero!! L=1`)
            nh[nMod] = sign(Mhr0) * Mhr0 / uhr2 ^(1/2)
        else
            nh[nMod] = sign(Mhr0) * Mhr0 / uhr2 ^(1/2)
        end
    end
    vhth[nMod] = (0.4 * Mh2 - uhr2)^0.5

    vhth2[nMod] = vhth[nMod].^2 
    uvth2[nMod] = (uh[nMod]).^2 ./ vhth2[nMod]
    uhL[nMod] = uh[nMod]

    for k in 1:nMod-1
        for s in 1:3
            nj += 1
            j += 2
            N = (j - 1) / 2 |> Int
            OrjLNb!(OrjL,uvth2,j,1,N,nMod;rtol_OrjL=rtol_OrjL)
            out[nj-3] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
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
    JacobCPEj1C!(JM, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                DMjL=DMjL,J=J,DMrjL=DMrjL,VnrL=VnrL,M1jL=M1jL,vth1jL=vth1jL,
                O1jL2=O1jL2,O1jL=O1jL,OrnL2=OrnL2,OrnL=OrnL,
                rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
"""

# The Jacobian matrix: JM

# nMode ≥ 3
function JacobCPEj1C!(JM::AbstractArray{T,N2}, x::AbstractVector{T}, nMod::Int; 
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
    j = 5   # L + 4
    N = (j - 1) / 2 |> Int
    for k in vec 
        for i in 1:3
            nj += 1
            j += 2
            N += 1
            vth1jL[nj] = vhth2[nMod]^N
            O1jL2[nj],O1jL[nj] = OrjLN2Nb(uvth2[nMod],j,1,N;rtol_OrjL=rtol_OrjL)
        end
    end

    ua1L = uh[nMod]
    for s in vec
        # JMC[:,s] = ∂\∂cs (MhjL[:])

        uaiL = uh[s]
        VnrL[:] = [1.0, nh[s] / uh[s], nh[s] / vhth[s]]
        sn = 3(s - 1)
        sn1 = sn + 1
        sn2 = sn + 2
        sn3 = sn + 3

        nj = 1
        j = 1  # L + 0
        OrnL2[nj],OrnL[nj] = OrjLN2Nb(uvth2[s],j,1,N;rtol_OrjL=rtol_OrjL)
        nj = 2
        j = 3
        OrnL2[nj],OrnL[nj] = OrjLN2Nb(uvth2[s],j,1,N;rtol_OrjL=rtol_OrjL)
        nj = 3
        j = 5
        OrnL2[nj],OrnL[nj] = OrjLN2Nb(uvth2[s],j,1,N;rtol_OrjL=rtol_OrjL)

        nj = 0
        # j = L + 4
        for k in vec
            for i in 1:3
                nj += 1
                j += 2
                N = (j - 1) / 2 |> Int
                vthijL = vhth2[s]^N
                # vth1jL = vhth2[nMod]^N
                OrjL2,OrjL = OrjLN2Nb(uvth2[s],j,1,N;rtol_OrjL=rtol_OrjL)
                ddcMhj1C0D2V!(DMjL,J,DMrjL,nh[s],uh[s],vhth[s],
                       uh[nMod],vhth[nMod],uaiL,vthijL,ua1L,vth1jL[nj],VnrL,
                       M1jL,O1jL2[nj],O1jL[nj],OrjL2,OrjL,OrnL2,OrnL,j,1;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
                JM[nj, sn1] = DMjL[1]
                JM[nj, sn2] = DMjL[2]
                JM[nj, sn3] = DMjL[3]
            end
        end
    end
end


        