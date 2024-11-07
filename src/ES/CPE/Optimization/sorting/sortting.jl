

"""
  Sorting `King` according to `vthi` and `uai` to ensure that `vthi[k] ≤ vthi[k+1]` and `uai[k] ≥ uai[k+1]

  Inputs:

  Outputs:
    sort_nuTi!(nai, uai, vthi, nMod)
"""

function sort_nuTi!(nai::AbstractVector{T},uai::AbstractVector{T},
    vthi::AbstractVector{T},nMod::Int)

    vec = sorperm(vthi)

    nai[:], uai[:], vthi[:] = nai[vec], uai[vec], vthi[vec]

    # All `vthi` are different to each other
    if sun(vec) ≠ nMod / 2 * (nMod + 1)
        k = 1
        for i in 2:nMode
            if vthi[i] == vthi[k]
                if uai[i] > uai[k]
                    nai[k,i] = nai[i,k]
                    uai[k,i] = uai[i,k]
                    vthi[k,i] = vthi[i,k]
                end
            end
            k += 1
        end
    end
end
