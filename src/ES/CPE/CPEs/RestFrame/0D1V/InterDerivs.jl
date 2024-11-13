

"""
    Intermedinate derivatives for the optimization of 
    the conservation-constrait characteristic parameter equations (CPEsC)
    in weakly anisotropic and moderate anisotropic plasma system 
    with general axisymmetric velocity space. 
    The plasma is in a local sub-equilibrium state.

    Order `j ∈ {L,L+2,L+4}` are enforced by the algorithm.

    The general kinetic moments are renormalized by `CMjL`.
    
      `Mhst = 𝓜ⱼ(f̂ₗ) / CMjL`
  
  See Ref. of Wang (2025) titled as "StatNN: Inherently interpretable PINNs designed for
    axisymmetric velocity space with moderate anisotropic plasma simulation".
  
"""


"""
  Jacobian:

    `𝕁 = Vᵣₗⁿ [Dᵣₗⁿ¹ Dᵣₗᵘ¹ Dᵣₗᵀ¹]`

  Inputs:
    𝕁: = zeros(T,3,3)
    DMrjL: = zeros(T,3)
    M1jL: = [M10L, M12L, M14L]
    OrnL2: = [Or0jL2, Or2L2, Or4L2]
    OrnL: = [Or0jL, Or2L, Or4L]

  Outputs:
    JacobC0D2V!(J,DMrjL,nai,uai,vthi,ua1,vth1,M1jL,OrnL2,OrnL,L;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
"""

function JacobC0D2V!(J::AbstractArray{T,N},DMrjL::AbstractVector{T}, 
  nai::T,uai::T,vthi::T,ua1::T,vth1::T,M1jL::AbstractVector{T},
  OrnL2::AbstractVector{T},OrnL::AbstractVector{T},L::Int;atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where {T <: Real, N}


  # j = L + 2
  nj = 2
  DM1jlC0D2V!(DMrjL,nai,uai,vthi,OrnL2[nj],OrnL[nj];atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)

  # Drlu1
  J[2,:] = (2L + 5) * M1jL[nj] * DMrjL
  # Drlu1
  J[3,:] = DMrjL

  # j = L + 4
  nj = 3
  DM1jlC0D2V!(DMrjL,nai,uai,vthi,OrnL2[nj],OrnL[nj];atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)

  # Drlu1
  J[2,:] -= (L + 1.5) * DMrjL
  J[2,:] /= 4 * (ua1)^2

  # DrlT1
  J[3,:] -= 2 * J[2,:]
  J[2,:] /= abs(ua1)
  J[3,:] /= ((2l+3) * vth1)

  # j = L
  nj = 1
  DM1jlC0D2V!(DMrjL,nai,uai,vthi,OrnL2[nj],OrnL[nj];atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
  # Drln1
  if L == 0
    J[1,:] = DMrjL
  elseif L == 1
    J[1,:] = (DMrjL - M1jL[nj] * Drlu1 / ua1) / ua1
  else
    J[1,:] = (DMrjL - L * M1jL[nj] * Drlu1 / ua1) / ua1^L
  end
end

"""
  The derivatives of `𝓜₁ⱼₗ` respective to 
  characteristic parameters, `n̂ᵣₗ`, `ûᵣₗ` and `v̂thᵣₗ` when `r ≥ 2`:
  
    `DMrjL = [∂/∂n̂ᵣₗ ∂/∂ûᵣₗ ∂/∂v̂thᵣₗ]ᵀ 𝓜₁ⱼₗ'

  Inputs:
  Outputs:
    DM1jlC0D2V!(DMrjL,nai,uai,vthi,OrnL2,OrnL;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
"""

function DM1jlC0D2V!(DMrjL::AbstractVector{T}, nai::T, uai::T, vthi::T, 
  OrnL2::T, OrnL::T;atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where {T <: Real}

  if L == 0
    DMrjL[1] = (1 + OrnL)
    DMrjL[2] = nai * OrnL2
    DMrjL[3] = - DMrjL[2] / vthi
    if abs(uai) ≥ rtol_Mh
      DMrjL[2] = DMrjL[2] / uai
    else
      @warn("DM1jlC0D2V!: `uai ≤ rtol_Mh` may cause instability of the algorithm when `L=0`.",uai)
      DMrjL[2] = DMrjL[2] / uai
    end
  elseif L == 1
    DMrjL[1] = (1 + OrnL)
    DMrjL[2] = - nai * (DMrjL[1] - OrnL2)
    DMrjL[3] = - nai * (uai / vthi * OrnL2)
    DMrjL[1] *= - uai
    
  elseif L == 2
    DMrjL[1] = (1 + OrnL) 
    DMrjL[3] = - uai
    DMrjL[2] = DMrjL[3] * (L * DMrjL[1] - OrnL2)
    DMrjL[1] *= DMrjL[3] * uai
    DMrjL[2] *= nai
    DMrjL[3] *= nai * (uai / vthi * OrnL2)
  else
    DMrjL[1] = (1 + OrnL) 
    DMrjL[3] = - uai^(L-1)
    DMrjL[2] = DMrjL[3] * (L * DMrjL[1] - OrnL2)
    DMrjL[1] *= DMrjL[3] * uai
    DMrjL[2] *= nai
    DMrjL[3] *= nai * (uai / vthi * OrnL2)
  end
end
