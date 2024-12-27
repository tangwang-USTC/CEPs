"""

    # vthi
    # The parameter limits for MCF plasma.
    # x0 = zeros(T,3nMod)      # [uai1, vthi1, uai2, vthi2, ⋯]

  Inputs:
  Outputs:
    xccCPEs!(x0,lbs,ubs,nai,uai,vthi,nMod;uhMax=uhMax,vhthMin=vhthMin,vhthMax=vhthMax)
"""

function xccCPEs!(x0::AbstractVector{T},lbs::AbstractVector{T},ubs::AbstractVector{T},
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int;
    uhMax::T=-3.0,vhthMin::T=1e-2,vhthMax::T=100) where{T <: Real}

    for i in 1:nMod
        # nai
        i3 = 3i - 2
        x0[i3] = deepcopy(nai[i])
        lbs[i3] = 0.0
        ubs[i3] = 1.0
        # uai
        i3 += 1
        x0[i3] = deepcopy(uai[i])
        lbs[i3] = -uhMax
        ubs[i3] = uhMax
        # vthi
        i3 += 1
        x0[i3] = deepcopy(vthi[i])
        lbs[i3] = -vhthMin
        ubs[i3] = vhthMax
    end
    return nothing
end

# For "Optimization.jl" when "is_MTK=true"
function xccCPEs!(x0::AbstractVector{Pair{Num, T}},lbs::AbstractVector{T},ubs::AbstractVector{T},
    xx::AbstractVector{Num},nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},
    nMod::Int;uhMax::T=-3.0,vhthMin::T=1e-2,vhthMax::T=100) where{T <: Real}

    for i in 1:nMod
        # nai
        i3 = 3i -2
        x0[i3] = xx[i3] => deepcopy(nai[i])
        lbs[i3] = 0.0
        ubs[i3] = 1.0

        # uai
        i3 = 3i -1
        x0[i3] = xx[i3] => deepcopy(uai[i])
        lbs[i3] = -uhMax
        ubs[i3] = uhMax

        # vthi
        i3 = 3i
        x0[i3] = xx[i3] => deepcopy(vthi[i])
        lbs[i3] = -vhthMin
        ubs[i3] = vhthMax
    end
    return nothing
end

# # For "NonLinearSolve.jl"
# function xccCPEs!(x0::AbstractVector{T},lbs::AbstractVector{T},ubs::AbstractVector{T},
#     nai::AbstractVector{T}, uai::AbstractVector{T}, vthi::AbstractVector{T}, uhLN::T) where{T <: Real}

#     # vthi
#     # The parameter limits for MCF plasma.
#     # x0 = zeros(T,3nMod)      # [uai1, vthi1, uai2, vthi2, ⋯]

#     vec = 1:nMod
#     for i in vec
#         # nai
#         i1 = 3i - 2
#         lbs[i1] = 0
#         ubs[i1] = 1.0
#         x0[i1] = deepcopy(nai[i])

#         # uai
#         i2 = 3i - 1
#         lbs[i2] = - uhMax
#         ubs[i2] = uhMax
#         x0[i2] = deepcopy(uai[i])

#         # vthi
#         i3 = 3i
#         lbs[i3] = vhthMin
#         ubs[i3] = vhthMax
#         x0[i3] = deepcopy(vthi[i])
#     end
# end
