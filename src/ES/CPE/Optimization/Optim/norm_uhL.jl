
"""

  uhLN = uhLNorm(nai,uai,L)

"""

function uhLNorm(nai::AbstractVector{T},uai::AbstractVector{T},L::Int) where{T}

    if L == 0
        return 1.0 |> T
    elseif L == 1
        return sum_kbn(nai .* abs.(uai))
    elseif L == 2
        return sum_kbn(nai .* (uai).^2)^0.5
    else
        if iseven(L)
            return sum_kbn(nai .* (uai).^L)^(T(1)/L)
        else
            return sum_kbn(nai .* abs.(uai).^L)^(T(1)/L)
        end
    end
end






