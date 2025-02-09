using LinearAlgebra
using JuMP
using Hypatia

model = Model(() -> Hypatia.Optimizer(verbose = false))

@variable(model, x[1:3] >= 0)
@constraint(model, sum(x) == 5)
@variable(model, hypo)

@objective(model, Max, hypo)

V = rand(2, 3)
Q = V * diagm(x) * V'
aff = vcat(hypo, [Q[i, j] for i in 1:2 for j in 1:i]...)

@constraint(model, aff in MOI.RootDetConeTriangle(2))

# solve and query solution
JuMP.optimize!(model)

termination_status(model)
objective_value(model)
value.(x)

