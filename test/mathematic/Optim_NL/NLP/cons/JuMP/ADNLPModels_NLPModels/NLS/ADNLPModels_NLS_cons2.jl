using ADNLPModels

nlshs20_autodiff0(::Type{T}; kwargs...) where {T <: Number} = nlshs20_autodiff(Vector{T}; kwargs...)

function nlshs20_autodiff0(::Type{S} = Vector{Float64}; kwargs...) where {S}

    @show S
  F(x) = [1 - x[1]; 
         10 * (x[2] - x[1]^2)]
  
  c(x) = [x[1] + x[2]^2; 
          x[1]^2 + x[2]; 
          x[1]^2 + x[2]^2 - 1]

  lcon = fill!(S(undef, 3), 0)
  ucon = fill!(S(undef, 3), Inf)

  x0 = S([-2; 1])
  lbs = S([-1 // 2; -Inf])
  ubs = S([1 // 2; Inf])
  return ADNLSModel(F, x0, 2, lbs, ubs, c, lcon, ucon, name = "nlshs20_autodiff"; kwargs...)
end


F(x) = [1 - x[1]; 
       10 * (x[2] - x[1]^2)]
c(x) = [x[1] + x[2]^2; 
        x[1]^2 + x[2]; 
        x[1]^2 + x[2]^2 - 1]

S = Vector{Float64}
lcon = fill!(S(undef, 3), 0)
ucon = fill!(S(undef, 3), Inf)

x0 = S([-2; 1])
lbs = S([-1 // 2; -Inf])
ubs = S([1 // 2; Inf])
nls = ADNLSModel(F, x0, 2, lbs, ubs, c, lcon, ucon, name = "nlshs20_autodiff")

