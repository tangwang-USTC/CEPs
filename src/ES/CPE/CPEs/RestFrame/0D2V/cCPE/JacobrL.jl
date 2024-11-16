

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
    uh1L := (uh1)^L

  Outputs:
    JacobC0D2V!(J,DM1RjL,ar2L,ar4L,arLL,M1jL,uh1,vhth1,uh1L,uhLN,L;is_norm_uhL=is_norm_uhL)
"""
 
function JacobC0D2V!(J::AbstractArray{T,N},DM1RjL::AbstractVector{T},
  ar2L::AbstractVector{T},ar4L::AbstractVector{T},arLL::AbstractVector{T},M1jL::AbstractVector{T},
  uh1::T,vhth1::T,uh1L::T,uhLN::T,L::Int;is_norm_uhL::Bool=true) where {T <: Real, N}

  # @show is_norm_uhL, uhLN
  if L == 0
    JacobL0C0D2V!(J,DM1RjL,ar2L,ar4L,arLL,M1jL,uh1,vhth1)
  else
    # j = L + 2
    DM1RjL[:] = DM1RjLC0D2V(ar2L,M1jL[2],arLL)           # M1LL * DM1R2L
    # DrLT1
    J[3,:] = DM1RjL / M1jL[1]
  
    # DrLu1
    if L == 111
      # CDx = 2.5^2 * 3.5 / (4 * uh1^3)
      CDx = 5.46875 / uh1^3
    else
      CDx = (L+1.5)^2 * (L+2.5) / (4 * uh1^3)
    end
    J[2,:] = 2 * M1jL[2] * DM1RjL
  
    # j = L + 4
    DM1RjL[:] = DM1RjLC0D2V(ar4L,M1jL[3],arLL)           # M1LL * DM1R4L
    J[2,:] -= DM1RjL

    # if is_norm_uhL
    #   CDx *= uhLN^L
    # end
    J[2,:] *= (CDx / M1jL[1])
    # J[2,1] *= uhLN^L
  
    # # DrLn1, DrLT1
    # if is_norm_uhL == 8
    #   # uhLNL = uhLN^L
    #   if L == 111
    #     J[1,:] = - (arLL + J[2,:] * M1jL[1] / uh1) * (uhLN / uh1)
    #     J[3,:] -= 0.8 * uh1 * J[2,:]
    #   else
    #     J[1,:] = - (arLL + L * J[2,:] * M1jL[1] / uh1) * (uhLN / uh1)^L
    #     J[3,:] -= 4 / (2L+T(3)) * uh1 * J[2,:]
    #   end
    # else
      if L == 111
        J[1,:] = - (arLL + J[2,:] * M1jL[1] / uh1) / uh1L
        J[3,:] -= 0.8 * uh1 * J[2,:]
      else
        J[1,:] = - (arLL + L * J[2,:] * M1jL[1] / uh1) / uh1L
        J[3,:] -= 4 / (2L+T(3)) * uh1 * J[2,:]
      end
    # end
    J[3,:] /= (2 * vhth1)
    if is_norm_uhL
      # J[:,:] *= uhLN^L
      J[2:3,:] *= uhLN^L
      # J[3,:] *= uhLN^L
      # J[3,1] *= uhLN^L
    end
  end
  # when `M1LL *= 1`
  # 1. `J[:,:] *= 1`  ->  false
  # 2. `J[:,:] *= uhLN^L`  ->  false
  # 3. `J[1,:] *= 1`, `J[2:3,:] *= uhLN^L`  ->  ok
  # 4. `J[1,:] *= 1`, `J[2:3,1] *= uhLN^L`  ->  false
  # 5. `J[1,:] *= uhLN^L`, `J[2:3,:] *= 1`  ->  false
  # 6. `J[1,:] *= 1`, `CDx *= uhLN^L`, `J[3,:] *= uhLN^L  ->  false
  # 7. `J[1,:] *= uhLN^L`, `J[3,:] *= uhLN^L  ->  false
  # 8. `J[1,:] *= 1`, `J[2,1] *= uhLN^L, `J[3,1] *= uhLN^L  ->  false

  # @show is_C, is_norm_uhL, is_Jacobian , fmt2(uhLN)       
  # @show fmt2.(J)
  # @show Float64.(J)
  # @show fmtf2.([maximum(abs.(J)),cond(J)])
  # jhhggfg
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
  J[3,:] -= 4 / T(3) * uh1 * J[2,:]
  J[3,:] /= (2 * vhth1)
end

