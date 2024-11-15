

"""
  Reduceing the number of `nMod` according to `naik, uaik, vthik`

  Outputs:
    nMod_update!(is_nMod_renew, naik, uaik, vthik, nMod, ns;rtol_DnuTi=rtol_DnuTi)
    nMod_update!(is_nMod_renew, dtIKk1, naik, uaik, vthik, nMod, ns;rtol_DnuTi=rtol_DnuTi)
    is_nMod_renew, naik, uaik, vthik, nModel = nMod_update(naik, uaik, vthik, nModel;rtol_DnuTi=rtol_DnuTi)
    is_nMod_renew, naik, vthik, nModel = nMod_update(naik, vthik, nModel;rtol_DnuTi=rtol_DnuTi)
"""

# [nMod, ns]
function nMod_update!(is_nMod_renew::Vector{Bool}, 
    naik::Vector{TA}, uaik::Vector{TA}, vthik::Vector{TA}, nMod::Vector{Int64},
    ns::Int64;rtol_DnuTi::T=1e-7) where{T,TA}
    
    for isp in 1:ns
        nModel = nMod[isp]
        if  nModel ≥ 2
            if norm(uaik[isp]) ≤ epsT10
                # @show sum_kbn(naik[isp][1:nMod[isp]] .* vthik[isp][1:nMod[isp]].^2)
                is_nMod_renew[isp], naik[isp], vthik[isp], nModel = nMod_update(
                    naik[isp], vthik[isp], nModel;rtol_DnuTi=rtol_DnuTi)
                if is_nMod_renew[isp]
                    nMod[isp] = nModel
                    uaik[isp] = zero.(vthik[isp])
                end
            else
                is_nMod_renew[isp], naik[isp], uaik[isp], vthik[isp], nModel = nMod_update(
                    naik[isp], uaik[isp], vthik[isp], nModel)
                if is_nMod_renew[isp]
                    nMod[isp] = nModel
                end
            end
        else
            is_nMod_renew[isp] = false
        end
    end
end

################################################################
"""
"""

# [nMod]
function nMod_update(naik::AbstractVector{T}, uaik::AbstractVector{T}, 
    vthik::AbstractVector{T}, nModel::Int64;rtol_DnuTi::T=1e-7) where{T}

    if nModel == 2
        is_update = is_update_nMod2(uaik, vthik, rtol_DnuTi)
        if is_update
            nModel -= 1
            return is_update, [T(1)], [sum_kbn(naik .* uaik)], [T(1)], nModel
        else
            return is_update, naik, uaik, vthik, nModel
        end
    elseif nModel == 3
        is_update = zeros(Bool,3)               # C(nMod,2)
        i = 1 
        i1 = 2
        is_update[1] = is_update_nMod2(uaik[[i,i1]], vthik[[i,i1]], rtol_DnuTi)

        i1 = 3
        is_update[2] = is_update_nMod2(uaik[[i,i1]], vthik[[i,i1]], rtol_DnuTi)

        i = 2
        i1 = 3
        is_update[3] = is_update_nMod2(uaik[[i,i1]], vthik[[i,i1]], rtol_DnuTi)

        # Dropping the identical spices
        N1 = sum_kbn(is_update)
        if N1 == 0
            return false, naik, uaik, vthik, nModel
        elseif N1 == 3 || N1 == 2
            nModel = 1
            return true, [T(1)], [sum_kbn(naik .* uaik)], [T(1)], nModel
        elseif N1 == 1
            nModel = 2
            if is_update[1]
                i = 1 
                i1 = 2
                s  = 3
                nhN = 1 - naik[s]
                uhN = uhN_reduce(naik[[i,i1]],uaik[[i,i1]],nhN)
                vhthN = vhthN_reduce(naik[[i,i1]],uaik[[i,i1]],vthik[[i,i1]],nhN,uhN)
                # @show `Threduce`, nhN * vhthN ^2 + naik[s] * vthik[s]^2
                return true, [nhN,naik[s]], [uhN,uaik[s]], [vhthN,vthik[s]], nModel
            elseif is_update[2]
                i = 1
                i1 = 3
                s  = 2
                nhN = 1 - naik[s]
                uhN = uhN_reduce(naik[[i,i1]],uaik[[i,i1]],nhN)
                vhthN = vhthN_reduce(naik[[i,i1]],uaik[[i,i1]],vthik[[i,i1]],nhN,uhN)
                # @show `Threduce`, nhN * vhthN ^2 + naik[s] * vthik[s]^2
                return true, [nhN,naik[s]], [uhN,uaik[s]], [vhthN,vthik[s]], nModel
            else
                i = 2
                i1 = 3
                s  = 1
                nhN = 1 - naik[s]
                uhN = uhN_reduce(naik[[i,i1]],uaik[[i,i1]],nhN)
                vhthN = vhthN_reduce(naik[[i,i1]],uaik[[i,i1]],vthik[[i,i1]],nhN,uhN)
                # @show `Threduce`, nhN * vhthN ^2 + naik[s] * vthik[s]^2
                return true, [naik[s],nhN], [uaik[s],uhN], [vthik[s],vhthN], nModel
            end
        end
    else
        dfvgbnjh
        for i in 1:nModel-1
        end
    end
end

# [nMod] 

# [nMod] uaik == 0, -1
function nMod_update(naik::AbstractVector{T}, vthik::AbstractVector{T}, nModel::Int64;rtol_DnuTi::T=1e-7) where{T}

    if nModel == 2
        is_update = is_update_nMod2(vthik, rtol_DnuTi)
        if is_update
            nModel -= 1
            return is_update, [T(1)], [T(1)], nModel
            # return is_update, [T(1)], [sum_kbn(naik .* vthik.^2)], nModel
        else
            return is_update, naik, vthik, nModel
        end
    elseif nModel == 3
        RDvthik = zeros(T,3)               # C(nMod,2)
        i = 1 
        i1 = 2
        RDvthik[1] = is_update_nMod2(vthik[[i,i1]], rtol_DnuTi)

        i1 = 3
        RDvthik[2] = is_update_nMod2(vthik[[i,i1]], rtol_DnuTi)

        i = 2
        i1 = 3
        RDvthik[3] = is_update_nMod2(vthik[[i,i1]], rtol_DnuTi)

        # Dropping the identical spices
        if sum_kbn(RDvthik .≤ rtol_DnuTi) == 0
            return false, naik, vthik, nModel
        else
            b = sortperm(RDvthik)
            # @show `Threduce`, sum_kbn(naik .* vthik.^2)
            nModel = 2
            if b[1] == 1
                i = 1 
                i1 = 2
                s = 3
                nhN = 1 - naik[s]
                vhthN = vhthN_reduce(naik[[i,i1]],vthik[[i,i1]],nhN)
                # @show `Threduce`, nhN * vhthN ^2 + naik[s] * vthik[s]^2
                return true, [nhN,naik[s]], [vhthN,vthik[s]], nModel
            elseif b[1] == 2
                i = 1
                i1 = 3
                s = 2
                nhN = 1 - naik[s]
                vhthN = vhthN_reduce(naik[[i,i1]],vthik[[i,i1]],nhN)
                # @show `Threduce`, nhN * vhthN ^2 + naik[s] * vthik[s]^2
                return true, [nhN,naik[s]], [vhthN,vthik[s]], nModel
            else
                i = 2
                i1 = 3
                s = 1
                nhN = 1 - naik[s]
                vhthN = vhthN_reduce(naik[[i,i1]],vthik[[i,i1]],nhN)
                # @show `Threduce`, nhN * vhthN ^2 + naik[s] * vthik[s]^2
                return true, [naik[s], nhN], [vthik[s], vhthN], nModel
            end
        end
    else
        for i in 1:nModel-1
        end
    end
end

# [nMod]
function is_update_nMod2(vthik::AbstractVector{T}) where{T}

    RDvthi = abs(vthik[1] / vthik[2] - 1.0)
    # @show rtol_DnuTi, RDvthi
    # wegfh
    if RDvthi ≤ rtol_DnuTi_warn
        @warn("RDvthi: The relative differenc of `vthik` which will decide the parameter `nMod`.",RDvthi)
    # else
    #     @show RDvthi
    end
    return RDvthi
end
