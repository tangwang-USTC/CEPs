




function is_nMod_reduce(uaik::AbstractVector{T}, vthik::AbstractVector{T};
    atol_Mh::T=1e-10,rtol_Mh::T=1e-10) where{T}
 
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







