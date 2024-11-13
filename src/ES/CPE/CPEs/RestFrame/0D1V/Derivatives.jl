

"""
    First-order derivatives for the re-normalzied kinetic moments, `∂/∂c 𝓜ⱼ(f̂ₗ)`: 
      These derivatives are used in the optimization when solving 
      the conservation-constrait characteristic parameter equations (CPEsC)
      in weakly anisotropic and moderate anisotropic plasma system 
      with general axisymmetric velocity space. 
      The plasma is in a local sub-equilibrium state.

    Order `j ∈ {L,L+2,L+4}` are enforced by the algorithm.

    The general kinetic moments are renormalized by `CMjL`.
    
      `Mhst = 𝓜ⱼ(f̂ₗ) / CMjL`
  
  See Ref. of Wang (2025) titled as "StatNN: An inherently interpretable physics-informed neural networks designed for moderate anisotropic plasma simulation".
  
"""

"""
  First-order derivatives for the re-normalzied kinetic moments:

    `DMjL = [∂/∂n̂ᵣₗ ∂/∂ûᵣₗ ∂/∂v̂thᵣₗ]ᵀ 𝓜ⱼₗ'

  Inputs:
    DMjL: = zeros(T,3,3)
    DMrjL: = zeros(T,3)
    M1jL: = [M10L, M12L, M14L]
    OrnL2: = [Or0L2, Or2L2, Or4L2]
    OrnL: = [Or0L, Or2L, Or4L]

  Outputs:
    ddcMhjlC0D2V!(DMjL,J,DMrjL,nai,uai,vthi,ua1,vth1,uaiL,vthijL,ua1L,vth1jL,VnrL,
           M1jL,O1jL2,O1jL,OrjL2,OrjL,OrnL2,OrnL,j,L;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
"""

function ddcMhjlC0D2V!(DMjL::AbstractVector{T},J::AbstractArray{T,N2},DMrjL::AbstractVector{T}, 
    nai::T,uai::T,vthi::T,ua1::T,vth1::T,uaiL::T,vthijL::T,ua1L::T,vth1jL::T,
    VnrL::AbstractVector{T},M1jL::AbstractVector{T},O1jL2::T,O1jL::T,
    OrjL2::T,OrjL::T,OrnL2::AbstractVector{T},OrnL::AbstractVector{T},
    j::Int,L::Int;atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where {T <: Real,N2}
  
    if j == L 
        ddcMhllC0D2V!(DMjL,J,DMrjL,nai,uai,vthi,ua1,vth1,uaiL,ua1L,VnrL,M1jL,
               O1jL2,O1jL,OrjL2,OrjL,OrnL2,OrnL,L;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    else
        if L == 0
            ddcMhj0C0D2V!(DMjL,J,DMrjL,nai,uai,vthi,ua1,vth1,VnrL,M1jL,
                   O1jL2,O1jL,OrjL2,OrjL,OrnL2,OrnL,j;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
        else
            row = zeros(T,1,3)
            row[1] = 1 + O1jL
            row[2] = L * row[1] + O1jL2
            row[3] = (j - L) * row[1] - O1jL
        
            JacobC0D2V!(J,DMrjL,nai,uai,vthi,ua1,vth1,M1jL,OrnL2,OrnL,L;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
        
            DMjL[:] = (vth1jL * ua1L) * (row * J)[:]
        
            row[1] = 1 + OrjL
            row[2] = L * row[1] + OrjL2
            row[3] = (j - L) * row[1] - OrjL
        
            DMjL[:] += (vthijL * uaiL) * row
        
            DMjL[:] .*= [1.0, nai / uai, nai / vthi]
            # VnrL = [1.0, nai / uai, nai / vthi]
            DMjL[:] .*= VnrL
        end
    end
end

function ddcMhllC0D2V!(DMjL::AbstractVector{T},J::AbstractArray{T,N2},DMrjL::AbstractVector{T}, 
    nai::T,uai::T,vthi::T,ua1::T,vth1::T,uaiL::T,ua1L::T,VnrL::AbstractVector{T},M1jL::AbstractVector{T},
    O1jL2::T,O1jL::T,OrjL2::T,OrjL::T,OrnL2::AbstractVector{T},OrnL::AbstractVector{T},
    L::Int;atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where {T <: Real,N2}
  
    if L == 0
        ddcMh00C0D2V!(DMjL,J,DMrjL,nai,uai,vthi,ua1,vth1,VnrL,M1jL,
               O1jL2,O1jL,OrjL2,OrjL,OrnL2,OrnL;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    else
        row = zeros(T,1,3)
        row[1] = 1 + O1jL
        row[2] = L * row[1] + O1jL2
        row[3] = - O1jL
    
        JacobC0D2V!(J,DMrjL,nai,uai,vthi,ua1,vth1,M1jL,OrnL2,OrnL,L;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    
        DMjL[:] = ua1L * (row * J)[:]
    
        row[1] = 1 + OrjL
        row[2] = L * row[1] + OrjL2
        row[3] = - OrjL
    
        DMjL[:] += uaiL * row
    
        # DMjL[:] .*= [1.0, nai / uai, nai / vthi]
        # VnrL = [1.0, nai / uai, nai / vthi]
        DMjL[:] .*= VnrL
    end
end


function ddcMhj0C0D2V!(DMjL::AbstractVector{T},J::AbstractArray{T,N2},DMrjL::AbstractVector{T}, 
    nai::T,uai::T,vthi::T,ua1::T,vth1::T,VnrL::AbstractVector{T},M1jL::AbstractVector{T},
    O1jL2::T,O1jL::T,OrjL2::T,OrjL::T,OrnL2::AbstractVector{T},OrnL::AbstractVector{T},
    j::Int;atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where {T <: Real,N2}
  
    if j == 0
        ddcMh00C0D2V!(DMjL,J,DMrjL,nai,uai,vthi,ua1,vth1,VnrL,M1jL,
               O1jL2,O1jL,OrjL2,OrjL,OrnL2,OrnL;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    else
        if j == 1
            row = zeros(T,1,3)
            row[1] = 1 + O1jL
            row[2] = O1jL2
            row[3] = row[1] - O1jL
        
            JacobC0D2V!(J,DMrjL,nai,uai,vthi,ua1,vth1,M1jL,OrnL2,OrnL,0;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
        
            DMjL[:] = vth1 * (row * J)[:]
        
            row[1] = 1 + OrjL
            row[2] = OrjL2
            row[3] = row[1] - OrjL
        
            DMjL[:] += vthi * row
        
            # DMjL[:] .*= [1.0, nai / uai, nai / vthi]
            # VnrL = [1.0, nai / uai, nai / vthi]
            DMjL[:] .*= VnrL
        else
            row = zeros(T,1,3)
            row[1] = 1 + O1jL
            row[2] = O1jL2
            row[3] = j * row[1] - O1jL
        
            JacobC0D2V!(J,DMrjL,nai,uai,vthi,ua1,vth1,M1jL,OrnL2,OrnL,0;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
        
            DMjL[:] = vth1^j * (row * J)[:]
        
            row[1] = 1 + OrjL
            row[2] = OrjL2
            row[3] = j * row[1] - OrjL
        
            DMjL[:] += vthi^j * row
        
            # DMjL[:] .*= [1.0, nai / uai, nai / vthi]
            # VnrL = [1.0, nai / uai, nai / vthi]
            DMjL[:] .*= VnrL
        end
    end
end


function ddcMh00C0D2V!(DMjL::AbstractVector{T},J::AbstractArray{T,N2},DMrjL::AbstractVector{T}, 
    nai::T,uai::T,vthi::T,ua1::T,vth1::T,VnrL::AbstractVector{T},M1jL::AbstractVector{T},
    O1jL2::T,O1jL::T,OrjL2::T,OrjL::T,OrnL2::AbstractVector{T},OrnL::AbstractVector{T};
    atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where {T <: Real,N2}
  
    row = zeros(T,1,3)
    row[1] = 1 + O1jL
    row[2] = O1jL2
    row[3] = - O1jL

    JacobC0D2V!(J,DMrjL,nai,uai,vthi,ua1,vth1,M1jL,OrnL2,OrnL,0;atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)

    DMjL[:] = (row * J)[:]

    row[1] = 1 + OrjL
    row[2] = OrjL2
    row[3] = - OrjL

    DMjL[:] += row

    # DMjL[:] .*= [1.0, nai / uai, nai / vthi]
    # VnrL = [1.0, nai / uai, nai / vthi]
    DMjL[:] .*= VnrL
end

function vthijLs(vthi::T,j::Int,L::Int) where {T <: Real}
    
    if j == L
        return 1.0
    else
        if L == 0
            return vthi^j
        else
            return vthi^(j-L)
        end
    end
end



# function VnrL0D2V!(VnrL::AbstractVector{T},nai::T, uai::T, vthi::T) where {T <: Real}

#   VnrL[1] = 1.0
#   VnrL[2] = nai / uai
#   VnrL[3] = nai / vthi
# end


