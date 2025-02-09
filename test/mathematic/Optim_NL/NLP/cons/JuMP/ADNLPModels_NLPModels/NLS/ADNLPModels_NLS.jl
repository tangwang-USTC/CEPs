using ADNLPModels
using CaNNOLeS        # equality-constrained nonlinear least-squares solver

F(x) = [x[2]; x[1]]

nequ = 2
nbs = 3
x0 = ones(nbs)

nls = ADNLSModel(F, x0, nequ) # uses the default ForwardDiffAD backend.
# ADNLSModel(F, x0, nequ; backend = :forward)

# ADNLSModel(F, x0, nequ; backend = ADNLPModels.ReverseDiffAD) # uses ReverseDiffAD backend.

# using Zygote
    # ADNLSModel(F, x0, nequ; backend = ADNLPModels.ZygoteAD)


# Rosenbrock
nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)

output = cannoles(nls)
@show output.iter
@show output.status
@show output.solution
@show output.objective
@show output.multipliers

