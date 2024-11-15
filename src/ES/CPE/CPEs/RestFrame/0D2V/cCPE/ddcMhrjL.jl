
"""
  The derivatives of `(𝓜₁ⱼₗ/𝓜₁ₗₗ)` respective to 
  characteristic parameters, `n̂ᵣₗ`, `ûᵣₗ` and `v̂thᵣₗ` when `r ≥ 2`:
  
    `DM1RjL = 𝓜₁ₗₗ * [∂/∂n̂ᵣₗ ∂/∂ûᵣₗ ∂/∂v̂thᵣₗ]ᵀ (𝓜₁ⱼₗ/𝓜₁ₗₗ)'

  Inputs:
    `M1LL = 𝓜₁ⱼₗ` where `j = L`
    `RM1jL = 𝓜₁ⱼₗ/𝓜₁ₗₗ` where `j ≥ L + 2`

  Outputs:
    DM1RjL = DM1RjLC0D2V(arjL,RM1jL,arLL)
"""

function DM1RjLC0D2V(arjL::AbstractVector{T},RM1jL::T,arLL::AbstractVector{T}) where {T <: Real}

  return RM1jL * arLL - arjL
end

