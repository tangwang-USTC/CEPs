using JuMP, Optim

model = Model(Optim.Optimizer);

set_optimizer_attribute(model, "method", Optim.BFGS())

@variable(model, x[1:2]);

@objective(model, Min, (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2)

optimize!(model)

objective_value(model)

value.(x)

