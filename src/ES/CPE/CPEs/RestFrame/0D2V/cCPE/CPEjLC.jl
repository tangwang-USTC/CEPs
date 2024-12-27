

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
    out = CPEjLC(x,uhLN,L,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)

"""

# nMod ≥ 3
function CPEjLC(x::AbstractVector{T}, uhLN::T, L::Int, nMod::Int;out::AbstractVector{T}=[0.1, 1.0],
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0],
    M1jL::AbstractVector{T}=[1.0,1.0,1.0],
    is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T}

    if L == 0
        return CPEj0C(x,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,
                M1jL=M1jL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    elseif L == 111
        return CPEj1C(x,uhLN,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    else 
        if nMod == 22
            return CPEjLC(x,uhLN,L;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
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

            # # nh[nMod] * (uhLN^2 / uhr2) ^(L/2)
            # if is_norm_uhL
            #     if abs(M1jL[1]) ≤ rtol_Mh
            #         if abs(uh[nMod]) ≤ rtol_Mh
            #             nh[nMod] = 0.0
            #             dfghghgbb
            #         else
            #             if iseven(L)
            #                 nh[nMod] = M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
            #             else
            #                 nh[nMod] = sign(M1jL[1]) * M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
            #             end
            #         end
            #     else
            #         if abs(uh[nMod]) ≤ rtol_Mh
            #             @error(`Error: nhr ∝ MhrjL / uhr^L. Both MhrjL and uhr are almost zero!! L=`,L)
            #             if iseven(L)
            #                 nh[nMod] = M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
            #             else
            #                 nh[nMod] = sign(M1jL[1]) * M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
            #             end
            #         else
            #             if iseven(L)
            #                 nh[nMod] = M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
            #             else
            #                 nh[nMod] = sign(M1jL[1]) * M1jL[1] * (uhLN^2 / uhr2) ^(L/2)
            #             end
            #         end
            #     end
            # else
            # end

            if is_norm_uhL
                M1jL[1] *= uhLN^L
            end
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

            vhth2[nMod] = M1jL[2] - T(1)/(L+1.5) * uhr2
            (vhth2[nMod] ≥ 0) || error("`vhth2` must be a positive value",fmt2(vhth2[nMod]))
            vhth[nMod] = √(vhth2[nMod])
            uvth2[nMod] = uhr2 ./ vhth2[nMod]
            if is_norm_uhL
                if L == 1
                    uhL[nMod] = (uh[nMod] / uhLN)
                elseif L == 2
                    uhL[nMod] = (uhr2 / uhLN^2)
                else
                    uhL[nMod] = (uhr2 / uhLN^2)^(L/2)
                end
            else
                if L == 1
                    uhL[nMod] = uh[nMod]
                elseif L == 2
                    uhL[nMod] = uhr2
                else
                    uhL[nMod] = uhr2^(L/2)
                end
            end

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
            return out
        end
    end
end

# nMod = 2
function CPEjLC(x::AbstractVector{T}, uhLN::T, L::Int;out::AbstractVector{T}=[0.1, 1.0],
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0],
    M1jL::AbstractVector{T}=[1.0,1.0,1.0],
    is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T}

    if L == 0
        return CPEj0C(x;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,
                M1jL=M1jL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    elseif L == 111
        return CPEj1C(x,uhLN;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
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

        # `nh[nMod] *= (uhLN^2 / uhr2) ^(L/2)`. However, `M1jL[1]` without the normalization coefficient, `uhLN^L`. Checking the Jacobian matrix `JV` given by `jacobrL.jl`.
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

        # uhL[nMod]
        if is_norm_uhL
            if L == 1
                uhL[2] = (uh[2] / uhLN)
            elseif L == 2
                uhL[2] = (uhr2 / uhLN^2)
            else
                uhL[2] = (uhr2 / uhLN^2)^(L/2)
            end
        else
            if L == 1
                uhL[2] = uh[2]
            elseif L == 2
                uhL[2] = uhr2
            else
                uhL[2] = uhr2^(L/2)
            end
        end
    
        for s in 1:3
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjLNb!(OrjL,uvth2,j,L,N,2;rtol_OrjL=rtol_OrjL)
            out[nj-3] = sum_kbn((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        end
        return out
    end
end

# nMod = 1

