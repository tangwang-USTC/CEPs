

"""
  Characteristic parameter equations (CPEs) when `L = 1`
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
    CPEj1!(out, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,rtol_OrjL=rtol_OrjL)
    CPEj1!(out, x, nMod,Mhst;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,rtol_OrjL=rtol_OrjL)

"""

# nMode = 1

# nMode ≥ 2
function CPEj1!(out::AbstractVector{T}, x::AbstractVector{T}, nMod::Int;
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhst::AbstractVector{T}=[0.1, 1.0], 
    rtol_OrjL::T=1e-10) where{T}

    nh[:] = x[1:3:end]
    uh[:] = x[2:3:end]
    vhth[:] = x[3:3:end]

    vhth2[:] = vhth.^2 
    uvth2[:] = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2

    nj = 1
    # j = 0
    out[nj] = sum(nh .* uh) - Mhst[nj]

    nj += 1
    j = 3
    Orj1 = CjLk(j,1,1) * uvth2
    out[nj] = sum((nh .* vhth2 .* uh) .* (1 .+ Orj1)) - Mhst[nj]

    nj += 1
    j = 5
    # Orj1 = zeros(T,nMod)
    N = 2
    OrjLNb!(Orj1,uvth2,j,1,N,nMod;rtol_OrjL=rtol_OrjL)
    out[nj] = sum((nh .* vhth2.^N .* uh) .* (1 .+ Orj1)) - Mhst[nj]

    for k in 2:nMod
        for s in 1:3
            nj += 1
            j += 2
            N = (j - 1) / 2 |> Int
            OrjLNb!(Orj1,uvth2,j,1,N,nMod;rtol_OrjL=rtol_OrjL)
            out[nj] = sum((nh .* vhth2.^N .* uh) .* (1 .+ Orj1)) - Mhst[nj]
        end
    end
end

function CPEj1!(out::AbstractVector{T}, x::AbstractVector{T}, nMod::Int, Mhst::AbstractVector{T};
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], rtol_OrjL::T=1e-10) where{T}

    nh[:] = x[1:3:end]
    uh[:] = x[2:3:end]
    vhth[:] = x[3:3:end]

    vhth2[:] = vhth.^2 
    uvth2[:] = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2

    nj = 1
    # j = 0
    out[nj] = sum(nh .* uh) - Mhst[nj]

    nj += 1
    j = 3
    Orj1 = CjLk(j,1,1) * uvth2
    out[nj] = sum((nh .* vhth2 .* uh) .* (1 .+ Orj1)) - Mhst[nj]

    nj += 1
    j = 5
    N = 2
    # Orj1 = zeros(T,nMod)
    OrjLNb!(Orj1,uvth2,j,1,N,nMod;rtol_OrjL=rtol_OrjL)
    out[nj] = sum((nh .* vhth2.^N .* uh) .* (1 .+ Orj1)) - Mhst[nj]

    for k in 2:nMod
        for i in 1:3
            nj += 1
            j += 2
            N = (j - 1) / 2 |> Int
            OrjLNb!(Orj1,uvth2,j,1,N,nMod;rtol_OrjL=rtol_OrjL)
            out[nj] = sum((nh .* vhth2.^N .* uh) .* (1 .+ Orj1)) - Mhst[nj]
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
    JacobCPEj1!(J, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,rtol_OrjL=rtol_OrjL)
"""

# The Jacobian matrix: J = zeros(T,nMod-1,nMod-1)

# nMode ≥ 2
function JacobCPEj1!(J::AbstractArray{T,N2}, x::AbstractVector{T}, nMod::Int; 
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], rtol_OrjL::T=1e-10) where{T,N2}

    fill!(J, 0.0)

    nh[:] = x[1:3:end]
    uh[:] = x[2:3:end]
    vhth[:] = x[3:3:end]

    vhth2[:] = vhth.^2 
    uvth2[:] = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2

    vec = 1:nMod
    nj = 1
    # j = 0
    # N = 0
    for s in vec
        Orj12,Orj1 = OrjLN2Nb(uvth2[s],j,1,N;rtol_OrjL=rtol_OrjL)
        sn = 3(s - 1)
        O1 = 1 + Orj1
        J[nj, sn+1] = uh[s] * O1
        J[nj, sn+2] = (nh[s]) * (O1 + Orj12)
        J[nj, sn+3] = (nh[s] * uh[s]) / vhth[s] * ( - Orj12)
    end

    nj = 2
    j = 3
    j1 = 2
    # N = 1
    for s in vec
        Orj12,Orj1 = OrjLN2Nb(uvth2[s],j,1,N;rtol_OrjL=rtol_OrjL)
        sn = 3(s - 1)
        O1 = 1 + Orj1
        J[nj, sn+1] = vhth2[s] * uh[s] * O1
        J[nj, sn+2] = (nh[s] * vhth2[s]) * (O1 + Orj12)
        J[nj, sn+3] = (nh[s] * vhth2[s] * uh[s]) / vhth[s] * (j1 * O1 - Orj12)
    end

    nj += 1
    j = 5
    j1 = 4
    N = 2
    for s in vec
        Orj12,Orj1 = OrjLN2Nb(uvth2[s],j,1,N;rtol_OrjL=rtol_OrjL)
        sn = 3(s - 1)
        TNuL1 = vhth2[s]^N
        O1 = 1 + Orj1
        J[nj, sn+1] = TNuL1 * uh[s] * O1
        J[nj, sn+2] = TNuL1 * nh[s] * (O1 + Orj12)
        J[nj, sn+3] = TNuL1 * (nh[s] * uh[s]) / vhth[s] * (j1 * O1 - Orj12)
    end

    for k in 2:nMod
        for i in 1:3
            nj += 1
            j += 2
            j1 = j-1
            N = j1 / 2 |> Int
            for s in vec
                Orj12,Orj1 = OrjLN2Nb(uvth2[s],j,1,N;rtol_OrjL=rtol_OrjL)
                sn = 3(s - 1)
                TNuL1 = vhth2[s]^N
                O1 = 1 + Orj1
                J[nj, sn+1] = TNuL1 * uh[s] * O1
                J[nj, sn+2] = TNuL1 * nh[s] * (O1 + Orj12)
                J[nj, sn+3] = TNuL1 * (nh[s] * uh[s]) / vhth[s] * (j1 * O1 - Orj12)
            end
        end
    end
end
