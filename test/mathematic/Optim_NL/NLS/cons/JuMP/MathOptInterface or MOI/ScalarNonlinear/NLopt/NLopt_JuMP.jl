using JuMP, NLopt

model = Model(NLopt.Optimizer)

set_attribute(model, "algorithm", :LD_MMA)
set_attribute(model, "xtol_rel", 1e-4)
set_attribute(model, "constrtol_abs", 1e-8)

@variable(model, x[1:2])
set_lower_bound(x[2], 0.0)
set_start_value.(x, [1.234, 5.678])

@NLobjective(model, Min, sqrt(x[2]))

@NLconstraint(model, (2 * x[1] + 0)^3 - x[2] <= 0)
@NLconstraint(model, (-1 * x[1] + 1)^3 - x[2] <= 0)

JuMP.optimize!(model)

min_f, min_x, ret = objective_value(model), value.(x), raw_status(model)

println(
    """
    objective value       : $min_f
    solution              : $min_x
    solution status       : $ret
    """
)

