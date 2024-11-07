

"""

  Inputs:

  Outputs:
    is_nMod_reduced!(is_reduce,uaik, vthik;atol_nuTi=atol_nuTi,rtol_nuTi=rtol_nuTi)
"""

function is_nMod_reduce!(is_reduce::AbstractVector{Bool},uaik::AbstractVector{T},
    vthik::AbstractVector{T},nMod::Int;atol_nuTi::T=1e-10,rtol_nuTi::T=1e-10) where{T}

    if nMod == 2
        is_reduce[1] = is_nMod_reduce(uaik, vthik;atol_nuTi=atol_nuTi,rtol_nuTi=rtol_nuTi)
    else
        for i in 1:nMod-1
            is_reduce[i] = is_nMod_reduce(uaik[i,i+1], vthik[i,i+1];atol_nuTi=atol_nuTi,rtol_nuTi=rtol_nuTi)
        end
    end
end



function is_nMod_reduce(uaik::AbstractVector{T}, vthik::AbstractVector{T};
    atol_nuTi::T=1e-10,rtol_nuTi::T=1e-10) where{T}
 
    RDvthi = abs(vthik[1] / vthik[2] - 1.0)
    if RDvthi ≥ rtol_nuTi
        return false
    else
        um = maximum(abs.(uaik))
        # Duai = abs(uaik[1] - uaik[2])
        if um ≥ atol_nuTi
            # RDuai = Duai / um
            is_update = abs(uaik[1] - uaik[2]) / um ≤ rtol_nuTi
        else
            is_update = abs(uaik[1] - uaik[2]) ≤ atol_nuTi
        end
    end
end

