using JuMP, Manopt, Manifolds

model = Model(Manopt.Optimizer)

# Change the solver with this option, `GradientDescentState` is the default

set_attribute("descent_state_type", GradientDescentState)
@variable(model, U[1:2, 1:2] in Stiefel(2, 2), start = 1.0)
@objective(model, Min, sum((A - U) .^ 2))

JuMP.optimize!(model)

solution_summary(model)

