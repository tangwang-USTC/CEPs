

"""
  Characteristic parameter equations (CPEs) 
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
    CPEjl!(out, x, L, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,rtol_OrjL=rtol_OrjL)
    CPEjl!(out, x, L, nMod,Mhst;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,rtol_OrjL=rtol_OrjL)

"""

# nMode = 1

# nMode ≥ 2
function CPEjl!(out::AbstractVector{T}, x::AbstractVector{T}, L::Int, nMod::Int;
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0], 
    rtol_OrjL::T=1e-10) where{T}

    if L == 0
        CPEj0!(out, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst)
    elseif L == 1
        CPEj2!(out, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst)
    else
        nh[:] = x[1:3:end]
        uh[:] = x[2:3:end]
        vhth[:] = x[3:3:end]
    
        vhth2[:] = vhth.^2 
        uvth2[:] = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2
        
        uhL = uh.^L
    
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
        OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
        out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    
        for k in 2:nMod
            for s in 1:3
                nj += 1
                j += 2
                N = (j - L) / 2 |> Int
                OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
                out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            end
        end
    end
end

function CPEjl!(out::AbstractVector{T}, x::AbstractVector{T}, L::Int, nMod::Int, Mhst::AbstractVector{T};
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], rtol_OrjL::T=1e-10) where{T}

    if L == 0
        CPEj0!(out, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst)
    elseif L == 1
        CPEj2!(out, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst)
    else
        nh[:] = x[1:3:end]
        uh[:] = x[2:3:end]
        vhth[:] = x[3:3:end]
    
        vhth2[:] = vhth.^2 
        uvth2[:] = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2
        
        uhL = uh.^L
    
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
        OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
        out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    
        for k in 2:nMod
            for s in 1:3
                nj += 1
                j += 2
                N = (j - L) / 2 |> Int
                OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
                out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            end
        end
    end
end

"""
  Inputs:
    J: = zeros(nMod,nMod)
    x: = x(nMod)
    nh: = nai
    uh: = uai[1:nMod]
    vhth: = vthi[1:nMod]

  Outputs:
    JacobCPEjl!(J, x, L, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,rtol_OrjL=rtol_OrjL)
"""

# The Jacobian matrix: J = zeros(T,nMod-1,nMod-1)

# nMode ≥ 2
function JacobCPEjl!(J::AbstractArray{T,N2}, x::AbstractVector{T}, L::Int, nMod::Int; 
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], rtol_OrjL::T=1e-10) where{T,N2}
    
    if L == 0
        JacobCPEj0!(J, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2)
    elseif L == 1
        JacobCPEj1!(J, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2)
    else
        fill!(J, 0.0)
    
        nh[:] = x[1:3:end]
        uh[:] = x[2:3:end]
        vhth[:] = x[3:3:end]
    
        vhth2[:] = vhth.^2 
        uvth2[:] = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2
    
        vec = 1:nMod

        if L == 2
            uhL1 = uh
        else
            uhL1 = uh.^(L-1)
        end
    
        nj = 1
        # j = L + 0
        # N = 0
        for s in vec
            OrjL2,OrjL = OrjLN2Nb(uvth2[s],j,L,N;rtol_OrjL=rtol_OrjL)
            sn = 3(s - 1)
            O1 = 1 + OrjL
            J[nj, sn+1] = uhL1[s] * uh[s] * O1
            J[nj, sn+2] = (nh[s] * uhL1[s]) * (L * O1 + OrjL2)
            J[nj, sn+3] = (nh[s] * uhL1[s] * uh[s]) / vhth[s] * ( - OrjL2)
        end
    
        nj = 2
        j = L + 2
        jL = 2
        # N = 1
        for s in vec
            OrjL2,OrjL = OrjLN2Nb(uvth2[s],j,L,N;rtol_OrjL=rtol_OrjL)
            sn = 3(s - 1)
            TNuL1 = vhth2[s] * uhL1[s]
            O1 = 1 + OrjL
            J[nj, sn+1] = TNuL1 * uh[s] * O1
            J[nj, sn+2] = TNuL1 * nh[s] * (L * O1 + OrjL2)
            J[nj, sn+3] = TNuL1 * (nh[s] * uh[s]) / vhth[s] * (jL * O1 - OrjL2)
        end
    
        nj += 1
        j = L + 4
        jL = 4
        N = 2
        for s in vec
            OrjL2,OrjL = OrjLN2Nb(uvth2[s],j,L,N;rtol_OrjL=rtol_OrjL)
            sn = 3(s - 1)
            TNuL1 = vhth2[s]^N * uhL1[s]
            O1 = 1 + OrjL
            J[nj, sn+1] = TNuL1 * uh[s] * O1
            J[nj, sn+2] = TNuL1 * nh[s] * (L * O1 + OrjL2)
            J[nj, sn+3] = TNuL1 * (nh[s] * uh[s]) / vhth[s] * (jL * O1 - OrjL2)
        end
    
        for k in 2:nMod
            for i in 1:3
                nj += 1
                j += 2
                jL = j-L
                N = jL / 2 |> Int
                for s in vec
                    OrjL2,OrjL = OrjLN2Nb(uvth2[s],j,L,N;rtol_OrjL=rtol_OrjL)
                    sn = 3(s - 1)
                    TNuL1 = vhth2[s]^N * uhL1[s]
                    O1 = 1 + OrjL
                    J[nj, sn+1] = TNuL1 * uh[s] * O1
                    J[nj, sn+2] = TNuL1 * nh[s] * (L * O1 + OrjL2)
                    J[nj, sn+3] = TNuL1 * (nh[s] * uh[s]) / vhth[s] * (jL * O1 - OrjL2)
                end
            end
        end
    end
end
