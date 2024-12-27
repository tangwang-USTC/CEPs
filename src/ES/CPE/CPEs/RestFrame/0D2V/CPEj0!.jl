

"""
  Characteristic parameter equations (CPEs) when `j = 0`
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
    CPEj0!(out, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,rtol_OrjL=rtol_OrjL)
    CPEj0!(out, x, nMod,Mhst;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,rtol_OrjL=rtol_OrjL)

"""

# nMode = 1

# nMode ≥ 2
function CPEj0!(out::AbstractVector{T}, x::AbstractVector{T}, nMod::Int;
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
    out[nj] = sum_kbn(nh) - Mhst[nj]

    nj += 1
    j = 2
    Orj0 = CjLk(T(j),T(1)) * uvth2
    out[nj] = sum_kbn((nh .* vhth2) .* (1 .+ Orj0)) - Mhst[nj]

    nj += 1
    j = 4
    N = 2
    Orj0Nb!(Orj0,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
    out[nj] = sum_kbn((nh .* vhth2.^N) .* (1 .+ Orj0)) - Mhst[nj]

    for k in 2:nMod
        for s in 1:3
            nj += 1
            j += 2
            N = j / 2 |> Int
            Orj0Nb!(Orj0,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
            out[nj] = sum_kbn((nh .* vhth2.^N) .* (1 .+ Orj0)) - Mhst[nj]
        end
    end
end

function CPEj0!(out::AbstractVector{T}, x::AbstractVector{T}, nMod::Int, Mhst::AbstractVector{T};
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
    out[nj] = sum_kbn(nh) - Mhst[nj]

    nj += 1
    j = 2
    Orj0 = CjLk(T(j),T(1)) * uvth2
    out[nj] = sum_kbn((nh .* vhth2) .* (1 .+ Orj0)) - Mhst[nj]

    nj += 1
    j = 4
    N = 2
    Orj0Nb!(Orj0,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
    out[nj] = sum_kbn((nh .* vhth2.^N) .* (1 .+ Orj0)) - Mhst[nj]

    for k in 2:nMod
        for s in 1:3
            nj += 1
            j += 2
            N = j / 2 |> Int
            Orj0Nb!(Orj0,uvth2,j,N,nMod;rtol_OrjL=rtol_OrjL)
            out[nj] = sum_kbn((nh .* vhth2.^N) .* (1 .+ Orj0)) - Mhst[nj]
        end
    end
end

"""
  Inputs:
    JM: = zeros(nMod,nMod)
    x: = x(nMod)
    nh: = nai
    uh: = uai[1:nMod]
    vhth: = vthi[1:nMod]

  Outputs:
    JacobCPEj0!(JM, x, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,rtol_OrjL=rtol_OrjL)
"""

# The Jacobian matrix: JM = zeros(T,nMod-1,nMod-1)

# nMode ≥ 2
function JacobCPEj0!(JM::AbstractArray{T,N2}, x::AbstractVector{T}, nMod::Int; 
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], rtol_OrjL::T=1e-10) where{T,N2}

    fill!(JM, 0.0)

    vec = 1:nMod
    nh[:] = x[1:3:end]
    uh[:] = x[2:3:end]
    vhth[:] = x[3:3:end]

    vhth2[:] = vhth.^2 
    uvth2[:] = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2

    nj = 1
    j = 0
    # N = 0    
    for s in vec
        sn = 3(s - 1)
        JM[nj, sn+1] = 1.0
        JM[nj, sn+2] = 0.0
        JM[nj, sn+3] = 0.0
    end

    nj = 2
    j = 2
    N = 1
    for s in vec
        Orj02,Orj0 = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
        sn = 3(s - 1)
        O1 = 1 + Orj0
        JM[nj, sn+1] = vhth2[s] * O1
        JM[nj, sn+2] = (nh[s] * vhth2[s]) / uh[s] * Orj02
        JM[nj, sn+3] = (nh[s] * vhth2[s]) / vhth[s] * (j * O1 - Orj02)
    end

    nj += 1
    j = 4
    N = 2
    for s in vec
        Orj02,Orj0 = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
        sn = 3(s - 1)
        TNuL1 = vhth2[s]^N
        O1 = 1 + Orj0
        JM[nj, sn+1] = TNuL1 * O1
        JM[nj, sn+2] = TNuL1 * nh[s] / uh[s] * Orj02
        JM[nj, sn+3] = TNuL1 * nh[s] / vhth[s] * (j * O1 - Orj02)
    end

    for k in 2:nMod
        for i in 1:3
            nj += 1
            j += 2
            N = j / 2 |> Int
            for s in vec
                Orj02,Orj0 = Orj0N2Nb(uvth2[s],j,N;rtol_OrjL=rtol_OrjL)
                sn = 3(s - 1)
                TNuL1 = vhth2[s]^N
                O1 = 1 + Orj0
                JM[nj, sn+1] = TNuL1 * O1
                JM[nj, sn+2] = TNuL1 * nh[s] / uh[s] * Orj02
                JM[nj, sn+3] = TNuL1 * nh[s] / vhth[s] * (j * O1 - Orj02)
            end
        end
    end
end
