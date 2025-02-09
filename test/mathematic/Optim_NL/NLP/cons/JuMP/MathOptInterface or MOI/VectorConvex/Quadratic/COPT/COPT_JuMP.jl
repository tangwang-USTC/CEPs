using JuMP
using COPT
using LinearAlgebra
model = Model(COPT.ConeOptimizer)
C = [1.0 -1.0; -1.0 2.0]
@variable(model, X[1:2, 1:2], PSD)
@variable(model, z[1:2] >= 0)
@objective(model, Min, C ⋅ X)
@constraint(model, c1, X[1, 1] - z[1] == 1)
@constraint(model, c2, X[2, 2] - z[2] == 1)
optimize!(model)
@show termination_status(model)
@show primal_status(model)
@show dual_status(model)
@show objective_value(model)
@show value.(X)
@show value.(z)
@show shadow_price(c1)
@show shadow_price(c2)

