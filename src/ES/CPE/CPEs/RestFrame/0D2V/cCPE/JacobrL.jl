

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
    𝕁 := zeros(T,3,3)
    DM1RjL := zeros(T,3)
    arjL := zeros(T,3)
    uh1 := û₁ₗ
    M1jL := [M1LL, RM12L, RM14L]
    arjL := 

  Outputs:
    JacobC0D2V!(J,DM1RjL,ar2L,ar4L,arLL,M1jL,uh1,vhth1,uh1L,L)
"""

function JacobC0D2V!(J::AbstractArray{T,N},DM1RjL::AbstractVector{T},
  ar2L::AbstractVector{T},ar4L::AbstractVector{T},arLL::AbstractVector{T},
  M1jL::AbstractVector{T},uh1::T,vhth1::T,uh1L::T,L::Int) where {T <: Real, N}

  if L == 0
    JacobL0C0D2V!(J,DM1RjL,ar2L,ar4L,arLL,M1jL,uh1,vhth1)
  elseif L == 1
  else
    # j = L + 2
    DM1RjL[:] = DM1RjLC0D2V(ar2L,M1jL[2],arLL)           # M1LL * DM1R2L
    # DrLT1
    J[3,:] = DM1RjL / M1jL[1]
  
    # DrLu1
    if L == 0
      # CDx = 1.5^2 * 2.5 / (4 * uh1^3)
      CDx = 1.40625 / uh1^3
    else
      CDx = (L+1.5)^2 * (L+2.5) / (4 * uh1^3)
    end
    J[2,:] = 2 * M1jL[2] * DM1RjL
  
    # j = L + 4
    DM1RjL[:] = DM1RjLC0D2V(ar4L,M1jL[3],arLL)           # M1LL * DM1R4L
    J[2,:] -= DM1RjL
    J[2,:] *= (CDx / M1jL[1])
  
    # DrLn1, DrLT1
    if L == 0
      J[1,:] = - arLL
      J[3,:] -= 4 / 3 * uh1 * J[2,:]
    elseif L == 1
      J[1,:] = - (arLL + J[2,:] * M1jL[1] / uh1) / uh1
      J[3,:] -= 0.8 * uh1 * J[2,:]
    else
      J[1,:] = - (arLL + L * J[2,:] * M1jL[1] / uh1) / uh1L
      J[3,:] -= 4 / (2L+3) * uh1 * J[2,:]
    end
    J[3,:] /= (2 * vhth1)
  end
end

function JacobL0C0D2V!(J::AbstractArray{T,N},DM1RjL::AbstractVector{T},
  ar2L::AbstractVector{T},ar4L::AbstractVector{T},arLL::AbstractVector{T},
  M1jL::AbstractVector{T},uh1::T,vhth1::T) where {T <: Real, N}

  # j = 2
  DM1RjL[:] = DM1RjLC0D2V(ar2L,M1jL[2],arLL)           # M1LL * DM1R2L
  # DrLT1
  J[3,:] = DM1RjL / M1jL[1]

  # DrLu1
  CDx = 1.40625 / uh1^3
  J[2,:] = 2 * M1jL[2] * DM1RjL

  # j = 4
  DM1RjL[:] = DM1RjLC0D2V(ar4L,M1jL[3],arLL)           # M1LL * DM1R4L
  J[2,:] -= DM1RjL
  J[2,:] *= (CDx / M1jL[1])

  # DrLn1, DrLT1
  J[1,:] = - arLL
  J[3,:] -= 4 / 3 * uh1 * J[2,:]
  J[3,:] /= (2 * vhth1)
end

"""
  The derivatives of `(𝓜₁ⱼₗ/𝓜₁ₗₗ)` respective to 
  characteristic parameters, `n̂ᵣₗ`, `ûᵣₗ` and `v̂thᵣₗ` when `r ≥ 2`:
  
    `DM1RjL = M1LL * [∂/∂n̂ᵣₗ ∂/∂ûᵣₗ ∂/∂v̂thᵣₗ]ᵀ (𝓜₁ⱼₗ/𝓜₁ₗₗ)'

  Inputs:
    `M1LL = 𝓜₁ⱼₗ` where `j = L`
    `RM1jL = 𝓜₁ⱼₗ/𝓜₁ₗₗ` where `j ≥ L + 2`

  Outputs:
    DM1RjL = DM1RjLC0D2V(arjL,RM1jL,arLL)
"""

function DM1RjLC0D2V(arjL::AbstractVector{T},RM1jL::T,arLL::AbstractVector{T}) where {T <: Real}

  return RM1jL * arLL - arjL
end
