

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
    out = CPEj0(x,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,rtol_OrjL=rtol_OrjL)
    out = CPEj0(x,nMod,Mhst;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,rtol_OrjL=rtol_OrjL)

"""

# nMode = 1

# nMode ≥ 2
function CPEj0(x::AbstractVector{T}, nMod::Int;out::AbstractVector{T}=[0.1, 1.0],
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
    return out
end

function CPEj0(x::AbstractVector{T}, nMod::Int, Mhst::AbstractVector{T};out::AbstractVector{T}=[0.1, 1.0],
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
    return out
end
