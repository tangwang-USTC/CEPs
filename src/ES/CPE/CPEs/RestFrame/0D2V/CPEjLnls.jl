

"""
  Applying Nonlinear Least Square (NLS) method to solve the Characteristic parameter equations (CPEs) 
    for weakly anisotropic and moderate anisotropic plasma system 
    with general axisymmetric velocity space. 
    The plasma is in a local sub-equilibrium state.

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
    out: = zeros(nMod,nMod)
    Mhst: = M̂ⱼₗ*, which is the renormalized general kinetic moments.

  Outputs:
    out = CPEjLnls(x,uhLN,L,nMod;out=out,nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
    out = CPEjLnls(x,uhLN,L,nMod,Mhst;out=out,nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)

"""

# nMode ≥ 2
function CPEjLnls(x::AbstractVector{T}, uhLN::T, L::Int, nMod::Int;out::AbstractVector{T}=[0.1, 1.0],
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0], 
    is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10) where{T}

    if L == 0
        return CPEj0(x,nMod;out=out,nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,rtol_OrjL=rtol_OrjL)
    elseif L == 111
        return CPEj1(x,uhLN, nMod;out=out,nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
    else
        nh[:] = x[1:3:end]
        uh[:] = x[2:3:end]
        vhth[:] = x[3:3:end]
    
        vhth2[:] = vhth.^2 
        uvth2[:] = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2
        
        if is_norm_uhL
            uhL = (uh / uhLN).^L
        else
            uhL = uh.^L
        end
    
        nj = 1
        # j = L + 0
        out[nj] = sum_kbn(nh .* uhL) - Mhst[nj]
    
        nj += 1
        j = L + 2
        OrjL = CjLk(j,L,1) * uvth2
        out[nj] = sum_kbn((nh .* vhth2 .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    
        nj += 1
        j = L + 4
        # OrjL = zeros(T,nMod)
        N = 2
        OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
        out[nj] = sum_kbn((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    
        for k in 2:nMod
            for s in 1:3
                nj += 1
                j += 2
                N = (j - L) / 2 |> Int
                OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
                out[nj] = sum_kbn((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            end
        end
        return out
    end
end

# For "Optimization.jl" and `is_constraint=true`
function CPEjLnls(x::AbstractVector{T}, uhLN::T, L::Int, nMod::Int, Ncons::Int;out::AbstractVector{T}=[0.1, 1.0],
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0], 
    is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10) where{T}

    if L == 10000
    #     return CPEj0(x,nMod;out=out,nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,rtol_OrjL=rtol_OrjL)
    # elseif L == 111
    #     return CPEj1(x,uhLN, nMod;out=out,nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
    else
        nh[:] = x[1:3:end]
        uh[:] = x[2:3:end]
        vhth[:] = x[3:3:end]
    
        vhth2[:] = vhth.^2 
        uvth2[:] = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2
        
        if is_norm_uhL
            uhL = (uh / uhLN).^L
        else
            uhL = uh.^L
        end
    
        if Ncons == 3
            nj = 0
            j = L + 4
        elseif Ncons == 2
            nj = 1
            j = L + 4
            # OrjL = zeros(T,nMod)
            N = 2
    
            # OrjL = zeros(T,nMod)
            # OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
            
            CjLks = CjLk(T(j),T(L),T.(1:N))
            OrjL =  CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2
    
            out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        elseif Ncons == 1
            nj = 1
            j = L + 2
            OrjL = CjLk(j,L,1) * uvth2
            out[nj] = sum((nh .* vhth2 .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        
            nj += 1
            j = L + 4
            # OrjL = zeros(T,nMod)
            N = 2
    
            # OrjL = zeros(T,nMod)
            # OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
            
            CjLks = CjLk(T(j),T(L),T.(1:N))
            OrjL =  CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2
    
            out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        elseif Ncons == 0
            nj = 1
            # j = L + 0
            out[nj] = sum(nh .* uhL) - Mhst[nj]
        
            nj += 1
            j = L + 2
            OrjL = CjLk(j,L,1) * uvth2
            out[nj] = sum((nh .* vhth2 .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        
            nj += 1
            j = L + 4
            # OrjL = zeros(T,nMod)
            N = 2
    
            # OrjL = zeros(T,nMod)
            # OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
            
            CjLks = CjLk(T(j),T(L),T.(1:N))
            OrjL =  CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2
    
            out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        else
            dfbffd
        end
    
        for k in 2:nMod
            for s in 1:3
                nj += 1
                j += 2
                N = (j - L) / 2 |> Int
                OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
                out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            end
        end
        return out
    end
end

# nMode = 1

## Verification
function CPEjLnls(x::AbstractVector{T}, uhLN::T, L::Int, nMod::Int, Mhst::AbstractVector{T};
    out::AbstractVector{T}=[0.1, 1.0], nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10) where{T}

    if L == 0
        return CPEj0(x,nMod;out=out,nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,rtol_OrjL=rtol_OrjL)
    elseif L == 111
        return CPEj1(x,uhLN, nMod;out=out,nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
    else
        nh[:] = x[1:3:end]
        uh[:] = x[2:3:end]
        vhth[:] = x[3:3:end]
    
        vhth2[:] = vhth.^2 
        uvth2[:] = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2

        uhLN = uhLNorm(nh,uh,L)
        if is_norm_uhL
            uhL = (uh / uhLN).^L
        else
            uhL = uh.^L
        end
    
        nj = 1
        # j = L + 0
        out[nj] = sum_kbn(nh .* uhL) - Mhst[nj]
    
        nj += 1
        j = L + 2
        OrjL = CjLk(T(j),T(L),T(1)) * uvth2
        out[nj] = sum_kbn((nh .* vhth2 .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    
        nj += 1
        j = L + 4
        # OrjL = zeros(T,nMod)
        N = 2
        OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
        out[nj] = sum_kbn((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    
        for k in 2:nMod
            for s in 1:3
                nj += 1
                j += 2
                N = (j - L) / 2 |> Int
                OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
                out[nj] = sum_kbn((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            end
        end
        return out
    end
end


