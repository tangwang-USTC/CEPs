using NLPModels, NLPModelsJuMP, JuMP
using CaNNOLeS        # equality-constrained nonlinear least-squares solver


model = Model()

# model = Model(NLPModelsJuMP.Optimizer)

# using Percival
# set_attribute(model, "solver", Percival.PercivalSolver)

x0 = [-1.2; 1.0]
@variable(model, x[i=1:2], start=x0[i])

@NLexpression(model, F1, x[1] - 1)
@NLexpression(model, F2, 10 * (x[2] - x[1]^2))

nls = MathOptNLSModel(model, [F1, F2], name="rosen-nls")

output = cannoles(nls)
@show output.iter
@show output.status
@show output.solution
@show output.objective
@show output.multipliers





residual(nls, nls.meta.x0)
jac_residual(nls, nls.meta.x0)

@show nls.Feval
@show nls.ceval

# JuMP.optimize!(nls)
# NLPModelsJuMP.optimize!(nls)

