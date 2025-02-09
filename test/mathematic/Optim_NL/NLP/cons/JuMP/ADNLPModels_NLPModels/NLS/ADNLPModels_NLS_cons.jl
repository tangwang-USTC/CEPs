using ADNLPModels

is_inplace = true
# is_inplace = false

function F!(output, x)
  output[1] = x[2]
  output[2] = x[1]
end

function c!(output, x) 
  output[1] = 1x[1] + x[2]
  output[2] = x[2]
end

nbs, ncon = 3, 2
lcon, ucon = zeros(ncon), zeros(ncon)

nequ = 2
x0 = ones(3)
nls = ADNLSModel!(F!, x0, nequ, c!, lcon, ucon)


