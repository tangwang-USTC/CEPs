

"""
  Characteristic parameter equations (CPEs)
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
    crjL,arLL,ar2L,ar4L = zeros(T,3),zeros(T,3),zeros(T,3),zeros(T,3)

  Outputs:
    CPEjLC!(out, x, uhLN, L, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)

"""

# nMod ≥ 3
function CPEjLC!(out::AbstractVector{T}, x::AbstractVector{T}, uhLN::T, L::Int, nMod::Int;
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0],
    M1jL::AbstractVector{T}=[1.0,1.0,1.0],
    is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T}

    if L == 20
        CPEj0C!(out, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,
                M1jL=M1jL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    elseif L == 11
        CPEj1C!(out, x, uhLN, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    else 
        if nMod == 2
            CPEjLC!(out, x, uhLN, L;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
                    is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
        else
            vec = 1:nMod-1
            nh[vec] = x[1:3:end]
            uh[vec] = x[2:3:end]
            vhth[vec] = x[3:3:end]
        
            vhth2[vec] = vhth[vec].^2 
            uvth2[vec] = (uh[vec]).^2 ./ vhth2[vec]              # uh .^ 2 ./ vhth .^ 2
        
            uhL = ones(T,nMod)
            if is_norm_uhL
                uhL[vec] = (uh[vec] / uhLN).^L
            else
                uhL[vec] = uh[vec].^L
            end
            
            # nMod = 1
            nj = 1
            # j = L + 0
            M1jL[1] = Mhst[nj] - sum_kbn((nh[vec] .* uhL[vec]))     # M1LL
        
            nj += 1
            j = L + 2
            OrjL = CjLk(j,L,1) * uvth2
            Mr2 = sum_kbn((nh[vec] .* vhth2[vec] .* uhL[vec]) .* (1 .+ OrjL[vec]))
            M1jL[2] = (Mhst[nj] - Mr2) / M1jL[1]               # RM12L
        
            nj += 1
            j = L + 4
            N = 2
            OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
            Mr4 = sum_kbn((nh[vec] .* vhth2[vec].^N .* uhL[vec]) .* (1 .+ OrjL[vec]))
            M1jL[3] = (Mhst[nj] - Mr4) / M1jL[1]               # RM14L
        
            # uhr4
            uhr2 = M1jL[2]^2 - M1jL[3]
            (uhr2 ≥ 0) || error("`uhr4` must be a positive value",fmt2(uhr2))
            uhr2 = (L+1.5) * ((L+2.5) * uhr2)^0.5
            (uhr2 ≥ 0) || error("`uhr2` must be a positive value",fmt2(uhr2))
        
            # uh[nMod]
            if iseven(L)
                uh[nMod] = (uhr2)^0.5
            else
                uh[nMod] = sign(M1jL[1]) * (uhr2)^0.5
            end
            # nh[nMod] * (uhLN^2 / uhr2) ^(L/2)
            if is_norm_uhL
                if abs(M1jL[1]) ≤ rtol_Mh
                    if abs(uh[nMod]) ≤ rtol_Mh
                        nh[nMod] = 0.0
                    else
                        if iseven(L)
                            nh[nMod] = M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
                        else
                            nh[nMod] = sign(M1jL[1]) * M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
                        end
                    end
                else
                    if abs(uh[nMod]) ≤ rtol_Mh
                        @error(`Error: nhr ∝ MhrjL / uhr^L. Both MhrjL and uhr are almost zero!! L=`,L)
                        if iseven(L)
                            nh[nMod] = M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
                        else
                            nh[nMod] = sign(M1jL[1]) * M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
                        end
                    else
                        if iseven(L)
                            nh[nMod] = M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
                        else
                            nh[nMod] = sign(M1jL[1]) * M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
                        end
                    end
                end
            else
                if abs(M1jL[1]) ≤ rtol_Mh
                    if abs(uh[nMod]) ≤ rtol_Mh
                        nh[nMod] = 0.0
                    else
                        if iseven(L)
                            nh[nMod] = M1jL[1] / uhr2 ^(L/2)
                        else
                            nh[nMod] = sign(M1jL[1]) * M1jL[1] / uhr2 ^(L/2)
                        end
                    end
                else
                    if abs(uh[nMod]) ≤ rtol_Mh
                        @error(`Error: nhr ∝ MhrjL / uhr^L. Both MhrjL and uhr are almost zero!! L=`,L)
                        if iseven(L)
                            nh[nMod] = M1jL[1] / uhr2 ^(L/2)
                        else
                            nh[nMod] = sign(M1jL[1]) * M1jL[1] / uhr2 ^(L/2)
                        end
                    else
                        if iseven(L)
                            nh[nMod] = M1jL[1] / uhr2 ^(L/2)
                        else
                            nh[nMod] = sign(M1jL[1]) * M1jL[1] / uhr2 ^(L/2)
                        end
                    end
                end
            end
            vhth2[nMod] = M1jL[2] - T(1)/(L+1.5) * uhr2
            (vhth2[nMod] ≥ 0) || error("`vhth2` must be a positive value",fmt2(vhth2[nMod]))
            vhth[nMod] = √(vhth2[nMod])
            uvth2[nMod] = uhr2 ./ vhth2[nMod]
            uhL[nMod] = uhr2^(L/2)

            @show fmt2.([nh[nMod],uh[nMod],vhth[nMod]])
            isnan(nh[nMod]) == false || error("`nMod` must be a positive value belonging to `[0,1]`",fmt2(nh[nMod]))
        
            for k in 1:nMod-1
                for s in 1:3
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
                    out[nj-3] = sum_kbn((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                end
            end
        end
    end
end

# nMod = 2
function CPEjLC!(out::AbstractVector{T}, x::AbstractVector{T}, uhLN::T, L::Int;
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0],
    M1jL::AbstractVector{T}=[1.0,1.0,1.0],
    is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T}

    if L == 20
        CPEj0C!(out, x;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,
                M1jL=M1jL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    elseif L == 11
        CPEj1C!(out, x, uhLN;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    else 
        vec = 1
        nh[vec] = x[1]
        uh[vec] = x[2]
        vhth[vec] = x[3]
    
        vhth2[vec] = vhth[vec]^2 
        uvth2[vec] = (uh[vec])^2 / vhth2[vec]              # uh .^ 2 ./ vhth .^ 2
    
        uhL = ones(T,2)
        if is_norm_uhL
            uhL[vec] = (uh[vec] / uhLN).^L
        else
            uhL[vec] = uh[vec]^L
        end
        
        # nMod = 1
        nj = 1
        # j = L + 0
        M1jL[1] = Mhst[nj] - nh[vec] * uhL[vec]     # M1LL
    
        nj += 1
        j = L + 2
        OrjL = CjLk(j,L,1) * uvth2
        Mr2 = (nh[vec] * vhth2[vec] * uhL[vec]) * (1 + OrjL[1])
        M1jL[2] = (Mhst[nj] - Mr2) / M1jL[1]
    
        nj += 1
        j = L + 4
        N = 2
        OrjL[1] = OrjLNb(uvth2[vec],j,L,N;rtol_OrjL=rtol_OrjL)
        Mr4 = (nh[vec] * vhth2[vec]^N * uhL[vec]) * (1 + OrjL[1])
        M1jL[3] = (Mhst[nj] - Mr4) / M1jL[1]
    
        # uhr4
        uhr2 = M1jL[2]^2 - M1jL[3]
        uhr2 = (L+1.5) * ((L+2.5) * uhr2)^0.5
    
        # uh[nMod]
        if iseven(L)
            uh[2] = (uhr2)^0.5
        else
            uh[2] = sign(M1jL[1]) * (uhr2)^0.5
        end
        # nh[nMod] * (uhLN^2 / uhr2) ^(L/2)
        if is_norm_uhL
            if abs(M1jL[1]) ≤ rtol_Mh
                if abs(uh[2]) ≤ rtol_Mh
                    nh[2] = 0.0
                    erffffff
                else
                    if iseven(L)
                        nh[2] = M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
                    else
                        nh[2] = sign(M1jL[1]) * M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
                    end
                end
            else
                if abs(uh[2]) ≤ rtol_Mh
                    @error(`Error: nhr ∝ MhrjL / uhr^L. Both MhrjL and uhr are almost zero!! L=`,L)
                    if iseven(L)
                        nh[2] = M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
                    else
                        nh[2] = sign(M1jL[1]) * M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
                    end
                else
                    if iseven(L)
                        nh[2] = M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
                    else
                        nh[2] = sign(M1jL[1]) * M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
                    end
                end
            end
        else
            if abs(M1jL[1]) ≤ rtol_Mh
                if abs(uh[2]) ≤ rtol_Mh
                    nh[2] = 0.0
                else
                    if iseven(L)
                        nh[2] = M1jL[1] / uhr2 ^(L/2)
                    else
                        nh[2] = sign(M1jL[1]) * M1jL[1] / uhr2 ^(L/2)
                    end
                end
            else
                if abs(uh[2]) ≤ rtol_Mh
                    @error(`Error: nhr ∝ MhrjL / uhr^L. Both MhrjL and uhr are almost zero!! L=`,L)
                    if iseven(L)
                        nh[2] = M1jL[1] / uhr2 ^(L/2)
                    else
                        nh[2] = sign(M1jL[1]) * M1jL[1] / uhr2 ^(L/2)
                    end
                else
                    if iseven(L)
                        nh[2] = M1jL[1] / uhr2 ^(L/2)
                    else
                        nh[2] = sign(M1jL[1]) * M1jL[1] / uhr2 ^(L/2)
                    end
                end
            end
        end
        vhth2[2] = M1jL[2] - T(1)/(T(L)+1.5) * uhr2
        vhth[2] = √(vhth2[2])
        uvth2[2] = uhr2 / vhth2[2]
        uhL[2] = uhr2^(L/2)
    
        for s in 1:3
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjLNb!(OrjL,uvth2,j,L,N,2;rtol_OrjL=rtol_OrjL)
            out[nj-3] = sum_kbn((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        end
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
    JacobCPEjLC!(JM, x, uhLN, L, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                DMjL=DMjL,J=J,DM1RjL=DM1RjL,vth1jL=vth1jL,vthijL=vthijL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,
                crjL=crjL,arLL=arLL,ar2L=ar2L,ar4L=ar4L,O1jL2=O1jL2,O1jL=O1jL,OrnL2=OrnL2,OrnL=OrnL,
                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
"""

# The Jacobian matrix: JM

# nMod ≥ 3
function JacobCPEjLC!(JM::AbstractArray{T,N2}, x::AbstractVector{T}, uhLN::T, L::Int, nMod::Int; 
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0],DMjL::AbstractVector{T}=[0.0,0.0,0.0],
    J::AbstractArray{T,N2}=[0.0 0.0;0.0 0.0],DM1RjL::AbstractVector{T}=[0.0,0.0,0.0],
    vth1jL::AbstractVector{T}=[0.0,0.0,0.0],vthijL::AbstractVector{T}=[0.0,0.0,0.0],
    Vn1L::AbstractVector{T},VnrL::AbstractVector{T},M1jL::AbstractVector{T}=[1.0,1.0,1.0],
    crjL::AbstractVector{T}=[0.0,0.0,0.0],arLL::AbstractVector{T}=[0.0,0.0,0.0],
    ar2L::AbstractVector{T}=[0.0,0.0,0.0],ar4L::AbstractVector{T}=[0.0,0.0,0.0],
    O1jL2::AbstractVector{T}=[0.0,0.0,0.0],O1jL::AbstractVector{T}=[0.0,0.0,0.0],
    OrnL2::AbstractVector{T}=[0.0,0.0,0.0],OrnL::AbstractVector{T}=[0.0,0.0,0.0],
    is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T,N2}

    if L == 20
        JacobCPEj0C!(JM, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                DMjL=DMjL,J=J,DM1RjL=DM1RjL,vth1jL=vth1jL,vthijL=vthijL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,
                crjL=crjL,arLL=arLL,ar2L=ar2L,ar4L=ar4L,O1jL2=O1jL2,O1jL=O1jL,
                OrnL2=OrnL2,OrnL=OrnL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    elseif L == 11
        JacobCPEj1C!(JM, x, uhLN, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                DMjL=DMjL,J=J,DM1RjL=DM1RjL,vth1jL=vth1jL,vthijL=vthijL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,
                crjL=crjL,arLL=arLL,ar2L=ar2L,ar4L=ar4L,O1jL2=O1jL2,O1jL=O1jL,OrnL2=OrnL2,OrnL=OrnL,
                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    else
        if nMod == 2
            JacobCPEjLC!(JM, x, uhLN, L;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                        DMjL=DMjL,J=J,DM1RjL=DM1RjL,vth1jL=vth1jL,vthijL=vthijL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,
                        crjL=crjL,arLL=arLL,ar2L=ar2L,ar4L=ar4L,O1jL2=O1jL2,O1jL=O1jL,OrnL2=OrnL2,OrnL=OrnL,
                        is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
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
            j = L + 4
            # N = (j - 1) / 2 |> Int
            N = 2
            for k in vec 
                for i in vec13
                    nj += 1
                    j += 2
                    N += 1
                    vth1jL[nj] = vhth2[nMod]^N
                    O1jL2[nj],O1jL[nj] = OrjLN2Nb(uvth2[nMod],j,L,N;rtol_OrjL=rtol_OrjL)
                end
            end
    
            if is_norm_uhL
                uh1L = (uh[nMod] / uhLN).^L
            else
                uh1L = uh[nMod]^L
            end
            Vn1L[:] = [1.0, nh[nMod] / uh[nMod], nh[nMod] / vhth[nMod]]
            for s in vec
                # JMC[:,s] = ∂\∂cs (MhjL[:])
                if is_norm_uhL
                    uaiL = (uh[s] / uhLN).^L
                else
                    uaiL = uh[s]^L
                end
                vthijL[vec13] = [1.0, vhth2[s], vhth2[s]^2]
                VnrL[:] = [1.0, nh[s] / uh[s], nh[s] / vhth[s]]
                sn = 3(s - 1) .+ vec13
    
                nj = 1
                j = L  # L + 0
                N = 0
                OrnL2[nj],OrnL[nj] = OrjLN2Nb(uvth2[s],j,L,N;rtol_OrjL=rtol_OrjL)
                nj = 2
                j = L + 2
                N = 1
                OrnL2[nj],OrnL[nj] = OrjLN2Nb(uvth2[s],j,L,N;rtol_OrjL=rtol_OrjL)
                nj = 3
                j = L + 4
                N = 2
                OrnL2[nj],OrnL[nj] = OrjLN2Nb(uvth2[s],j,L,N;rtol_OrjL=rtol_OrjL)
    
                nj = 0
                # j = L + 4
                for k in vec
                    for i in vec13
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        # vth1jL = vhth2[nMod]^N
                        vthijL[4] = vhth2[s]^N
                        OrjL2,OrjL = OrjLN2Nb(uvth2[s],j,L,N;rtol_OrjL=rtol_OrjL)
                        ddcMhjLC0D2V!(DMjL,J,DM1RjL,uh[nMod],vhth[nMod],uh1L,vth1jL[nj],uaiL,vthijL,Vn1L,VnrL,uhLN,M1jL,
                               crjL,arLL,ar2L,ar4L,O1jL2[nj],O1jL[nj],OrjL2,OrjL,OrnL2,OrnL,j,L;is_norm_uhL=is_norm_uhL)
                        JM[nj, sn] = DMjL
                    end
                end
            end
        end
    end
end

# nMod = 2
function JacobCPEjLC!(JM::AbstractArray{T,N2}, x::AbstractVector{T}, uhLN::T, L::Int; 
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0],DMjL::AbstractVector{T}=[0.0,0.0,0.0],
    J::AbstractArray{T,N2}=[0.0 0.0;0.0 0.0],DM1RjL::AbstractVector{T}=[0.0,0.0,0.0],
    vth1jL::AbstractVector{T}=[0.0,0.0,0.0],vthijL::AbstractVector{T}=[0.0,0.0,0.0],
    Vn1L::AbstractVector{T},VnrL::AbstractVector{T},M1jL::AbstractVector{T}=[1.0,1.0,1.0],
    crjL::AbstractVector{T}=[0.0,0.0,0.0],arLL::AbstractVector{T}=[0.0,0.0,0.0],
    ar2L::AbstractVector{T}=[0.0,0.0,0.0],ar4L::AbstractVector{T}=[0.0,0.0,0.0],
    O1jL2::AbstractVector{T}=[0.0,0.0,0.0],O1jL::AbstractVector{T}=[0.0,0.0,0.0],
    OrnL2::AbstractVector{T}=[0.0,0.0,0.0],OrnL::AbstractVector{T}=[0.0,0.0,0.0],
    is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T,N2}

    if L == 20
        JacobCPEj0C!(JM, x;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                DMjL=DMjL,J=J,DM1RjL=DM1RjL,vth1jL=vth1jL,vthijL=vthijL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,
                crjL=crjL,arLL=arLL,ar2L=ar2L,ar4L=ar4L,O1jL2=O1jL2,O1jL=O1jL,
                OrnL2=OrnL2,OrnL=OrnL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    elseif L == 11
        JacobCPEj1C!(JM, x, uhLN;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                DMjL=DMjL,J=J,DM1RjL=DM1RjL,vth1jL=vth1jL,vthijL=vthijL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,
                crjL=crjL,arLL=arLL,ar2L=ar2L,ar4L=ar4L,O1jL2=O1jL2,O1jL=O1jL,OrnL2=OrnL2,OrnL=OrnL,
                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    else
        fill!(JM, 0.0)
        vec13 = 1:3

        vec = 1
        nh[vec] = x[1]
        uh[vec] = x[2]
        vhth[vec] = x[3]
    
        vhth2[vec] = vhth[vec]^2 
        uvth2[vec] = (uh[vec])^2 / vhth2[vec]  

        nj = 0
        j = L + 4
        # N = (j - L) / 2 |> Int
        N = 2
        for i in vec13
            nj += 1
            j += 2
            N += 1
            vth1jL[nj] = vhth2[2]^N
            O1jL2[nj],O1jL[nj] = OrjLN2Nb(uvth2[2],j,L,N;rtol_OrjL=rtol_OrjL)
        end

        if is_norm_uhL
            uh1L = (uh[2] / uhLN)^L
        else
            uh1L = uh[2]^L
        end
        Vn1L[:] = [1.0, nh[2] / uh[2], nh[2] / vhth[2]]
        s = 1
        # JMC[:,s] = ∂\∂cs (MhjL[:])
        if is_norm_uhL
            uaiL = (uh[s] / uhLN)^L
        else
            uaiL = uh[s]^L
        end
        vthijL[vec13] = [1.0, vhth2[s], vhth2[s]^2]
        VnrL[:] = [1.0, nh[s] / uh[s], nh[s] / vhth[s]]
        sn = 3(s - 1) .+ vec13

        nj = 1
        j = L  # L + 0
        N = 0
        OrnL2[nj],OrnL[nj] = OrjLN2Nb(uvth2[s],j,L,N;rtol_OrjL=rtol_OrjL)
        nj = 2
        j = L + 2
        N = 1
        OrnL2[nj],OrnL[nj] = OrjLN2Nb(uvth2[s],j,L,N;rtol_OrjL=rtol_OrjL)
        nj = 3
        j = L + 4
        N = 2
        OrnL2[nj],OrnL[nj] = OrjLN2Nb(uvth2[s],j,L,N;rtol_OrjL=rtol_OrjL)

        nj = 0
        # j = L + 4
        for i in vec13
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            # vth1jL = vhth2[2]^N
            vthijL[4] = vhth2[s]^N
            OrjL2,OrjL = OrjLN2Nb(uvth2[s],j,L,N;rtol_OrjL=rtol_OrjL)
            ddcMhjLC0D2V!(DMjL,J,DM1RjL,uh[2],vhth[2],uh1L,vth1jL[nj],uaiL,vthijL,Vn1L,VnrL,uhLN,M1jL,
                   crjL,arLL,ar2L,ar4L,O1jL2[nj],O1jL[nj],OrjL2,OrjL,OrnL2,OrnL,j,L;is_norm_uhL=is_norm_uhL)
            JM[nj, sn] = DMjL
        end
    end
end
