


"""
  Inputs:
    nhN: the sum_kbn of reduced King `nhN = nh1 + nh2`
    nai: = nai / nhN

  Outputs:
    uh, vhth = uhvhth_reduce(nai,uai,vthi,uhN)
    uh, vhth = uhvhth_reduce(nai,vthi)
    
"""

# [nMod]
function uhvhth_reduce(nai::AbstractVector{T},uai::AbstractVector{T},
    vthi::AbstractVector{T},uhN::T) where{T}

    return sum_kbn(nai .* uai), ((sum_kbn(nai .* (1.5 * vthi .^2 + uai .^2)) - uhN^2) / 1.5) ^0.5
end

# [nMod]
function uhvhth_reduce(nai::AbstractVector{T},vthi::AbstractVector{T}) where{T}

    return sum_kbn(nai .* uai), (sum_kbn(nai .* vthi .^2)) ^0.5
end


"""
  Inputs:
    nhN: the sum_kbn of reduced King `nhN = nh1 + nh2`
    nai: = nai / nhN

  Outputs:
    uh = uh_reduce(nai,uai)
    vhth = vhth_reduce(nai,uai,vthi,uhN)
    vhth = vhth_reduce(nai,vthi)
    
"""

# [nMod]
function uh_reduce(nai::AbstractVector{T},uai::AbstractVector{T}) where{T}

    return sum_kbn(nai .* uai)
end

# [nMod]
function vhth_reduce(nai::AbstractVector{T},uai::AbstractVector{T},
    vthi::AbstractVector{T},uhN::T) where{T}

    return ((sum_kbn(nai .* (1.5 * vthi .^2 + uai .^2)) - uhN^2) / 1.5) ^0.5
end

# [nMod]
function vhth_reduce(nai::AbstractVector{T},vthi::AbstractVector{T}) where{T}

    return (sum_kbn(nai .* vthi .^2)) ^0.5
end
