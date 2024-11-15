

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

    `DMjL = [ ∂/∂xᵣₗ 𝓜ⱼₗ = [∂/∂n̂ᵣₗ ∂/∂ûᵣₗ ∂/∂v̂thᵣₗ]ᵀ 𝓜ⱼₗ'

  Inputs:
    DMjL := zeros(T,3,3)
    DM1RjL := zeros(T,3)
    vth1jL = zeros(T,3nMod-3) 
    vthijL := [vthiLL,vthi2L,vthi4L,vthijL]
    M1jL := [M1LL, RM12L, RM14L]
    crjL := zeros(T,3)
    arjL := zeros(T,3)
    OrnL2 := [Or0L2, Or2L2, Or4L2]
    OrnL := [Or0L, Or2L, Or4L]

  Outputs:
    ddcMhjLC0D2V!(DMjL,J,DM1RjL,uh1,vhth1,uh1L,vth1jL,uaiL,vthijL,Vn1L,VnrL,uhLN,M1jL,
           crjL,arLL,ar2L,ar4L,O1jL2,O1jL,OrjL2,OrjL,OrnL2,OrnL,j,L;is_norm_uhL=is_norm_uhL)
"""

function ddcMhjLC0D2V!(DMjL::AbstractVector{T},J::AbstractArray{T,N2},
    DM1RjL::AbstractVector{T},uh1::T,vhth1::T,uh1L::T,vth1jL::T,uaiL::T,vthijL::AbstractVector{T},
    Vn1L::AbstractVector{T},VnrL::AbstractVector{T},uhLN::T,
    M1jL::AbstractVector{T},crjL::AbstractVector{T},arLL::AbstractVector{T},
    ar2L::AbstractVector{T},ar4L::AbstractVector{T},O1jL2::T,O1jL::T,OrjL2::T,OrjL::T,
    OrnL2::AbstractVector{T},OrnL::AbstractVector{T},j::Int,L::Int;is_norm_uhL::Bool=true) where {T <: Real,N2}
  
    if j == L 
        ddcMhLLC0D2V!(DMjL,J,DM1RjL,uh1,vhth1,uh1L,uaiL,vthijL,Vn1L,VnrL,uhLN,
                        M1jL,crjL,arLL,ar2L,ar4L,OrnL2,OrnL,L;is_norm_uhL=is_norm_uhL)
    else
        if L == 20
            ddcMhj0C0D2V!(DMjL,J,DM1RjL,uh1,vhth1,vth1jL,vthijL,Vn1L,VnrL,
                        M1jL,crjL,arLL,ar2L,ar4L,O1jL2,O1jL,OrjL2,OrjL,OrnL2,OrnL,j)
        else
            # jL = L
            arLL0D2V!(arLL,L,uaiL,VnrL)
            jL = L + 2
            arjL0D2V!(ar2L,jL,L,uaiL,vthijL[2],VnrL,OrnL2[2],OrnL[2])
            jL = L + 4
            arjL0D2V!(ar4L,jL,L,uaiL,vthijL[3],VnrL,OrnL2[3],OrnL[3])

            # DMjL[:] = arjL 
            if j - L == 20
                DMjL[:] = arLL
            elseif j - L == 2
                DMjL[:] = ar2L
            elseif j - L == 4
                DMjL[:] = ar4L
            else
                arjL0D2V!(DMjL,j,L,uaiL,vthijL[4],VnrL,OrjL2,OrjL)
            end

            # DMjL[:] += arjL|(r=1)
            JacobC0D2V!(J,DM1RjL,ar2L,ar4L,arLL,M1jL,uh1,vhth1,uh1L,uhLN,L;is_norm_uhL=is_norm_uhL)
            crjL0D2V!(crjL,j,L,O1jL2,O1jL)          # c1jL
            DMjL[:] += (vth1jL * uh1L) * (reshape((Vn1L .* crjL),1,3) * J)[:]
        end
    end
end

function ddcMhLLC0D2V!(DMjL::AbstractVector{T},J::AbstractArray{T,N2},
    DM1RjL::AbstractVector{T},uh1::T,vhth1::T,uh1L::T,uaiL::T,vthijL::AbstractVector{T},
    Vn1L::AbstractVector{T},VnrL::AbstractVector{T},uhLN::T,
    M1jL::AbstractVector{T},crjL::AbstractVector{T},
    arLL::AbstractVector{T},ar2L::AbstractVector{T},ar4L::AbstractVector{T},
    OrnL2::AbstractVector{T},OrnL::AbstractVector{T},L::Int;is_norm_uhL::Bool=true) where {T <: Real,N2}
  
    if L == 20
        ddcMh00C0D2V!(DMjL,J,DM1RjL,uh1,vhth1,vthijL,Vn1L,VnrL,
                    M1jL,crjL,arLL,ar2L,ar4L,OrnL2,OrnL)
    else
        # jL = L
        arLL0D2V!(arLL,L,uaiL,VnrL)
        jL = L + 2
        arjL0D2V!(ar2L,jL,L,uaiL,vthijL[2],VnrL,OrnL2[2],OrnL[2])
        jL = L + 4
        arjL0D2V!(ar4L,jL,L,uaiL,vthijL[3],VnrL,OrnL2[3],OrnL[3])

        # DMjL[:] = arjL    # + arjL|(r=1)
        DMjL[:] = arLL

        # DMjL[:] += arjL|(r=1)
        JacobC0D2V!(J,DM1RjL,ar2L,ar4L,arLL,M1jL,uh1,vhth1,uh1L,uhLN,L;is_norm_uhL=is_norm_uhL)
        crLL0D2V!(crjL,L)          # c1jL
        DMjL[:] += uh1L * (reshape((Vn1L .* crjL),1,3) * J)[:]
    end
end

"""
  When `L == 20`

  Outputs:
    ddcMhj0C0D2V!(DMjL,J,DM1RjL,uh1,vhth1,vth1jL,vthijL,VnrL,
           M1jL,crjL,arLL,ar2L,ar4L,O1jL2,O1jL,OrjL2,OrjL,OrnL2,OrnL,j)
"""

function ddcMhj0C0D2V!(DMjL::AbstractVector{T},J::AbstractArray{T,N2},
    DM1RjL::AbstractVector{T},uh1::T,vhth1::T,vth1jL::T,vthijL::AbstractVector{T},
    Vn1L::AbstractVector{T},VnrL::AbstractVector{T},
    M1jL::AbstractVector{T},crjL::AbstractVector{T},arLL::AbstractVector{T},
    ar2L::AbstractVector{T},ar4L::AbstractVector{T},O1jL2::T,O1jL::T,OrjL2::T,OrjL::T,
    OrnL2::AbstractVector{T},OrnL::AbstractVector{T},j::Int) where {T <: Real,N2}
  
    if j == 0 
        ddcMh00C0D2V!(DMjL,J,DM1RjL,uh1,vhth1,vthijL,Vn1L,VnrL,
                    M1jL,crjL,arLL,ar2L,ar4L,OrnL2,OrnL)
    else
        # jL = 0
        ar000D2V!(arLL)
        jL = 2
        arj00D2V!(ar2L,jL,vthijL[2],VnrL,OrnL2[2],OrnL[2])
        jL = 4
        arj00D2V!(ar4L,jL,vthijL[3],VnrL,OrnL2[3],OrnL[3])

        # DMjL[:] = arjL    # + arjL|(r=1)
        if j == 0
            DMjL[:] = arLL
        elseif j == 2
            DMjL[:] = ar2L
        elseif j == 4
            DMjL[:] = ar4L
        else
            arj00D2V!(DMjL,j,vthijL[4],VnrL,OrjL2,OrjL)
        end
        # @show 1, DMjL

        # DMjL[:] += arjL|(r=1)
        JacobL0C0D2V!(J,DM1RjL,ar2L,ar4L,arLL,M1jL,uh1,vhth1)
        crj00D2V!(crjL,j,O1jL2,O1jL)          # c1jL
        DMjL[:] += vth1jL * (reshape((Vn1L .* crjL),1,3) * J)[:]
        # @show 2, DMjL
        # @show J
        # @show cond(J)
        # fyufugig
    end
end


function ddcMh00C0D2V!(DMjL::AbstractVector{T},J::AbstractArray{T,N2},
    DM1RjL::AbstractVector{T},uh1::T,vhth1::T,vthijL::AbstractVector{T},
    Vn1L::AbstractVector{T},VnrL::AbstractVector{T},
    M1jL::AbstractVector{T},crjL::AbstractVector{T},
    arLL::AbstractVector{T},ar2L::AbstractVector{T},ar4L::AbstractVector{T},
    OrnL2::AbstractVector{T},OrnL::AbstractVector{T}) where {T <: Real,N2}
  
    # jL = L
    ar000D2V!(arLL)
    jL = 2
    arj00D2V!(ar2L,jL,vthijL[2],VnrL,OrnL2[2],OrnL[2])
    jL = 4
    arj00D2V!(ar4L,jL,vthijL[3],VnrL,OrnL2[3],OrnL[3])

    # DMjL[:] = arjL    # + arjL|(r=1)
    DMjL[:] = arLL

    # DMjL[:] += arjL|(r=1)
    JacobL0C0D2V!(J,DM1RjL,ar2L,ar4L,arLL,M1jL,uh1,vhth1)
    crL00D2V!(crjL)          # c1jL
    DMjL[:] += (reshape((Vn1L .* crjL),1,3) * J)[:]
end

function vthijLs(vthi::T,j::Int,L::Int) where {T <: Real}
    
    if j == L
        return 1.0 |> T
    else
        if L == 20
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


