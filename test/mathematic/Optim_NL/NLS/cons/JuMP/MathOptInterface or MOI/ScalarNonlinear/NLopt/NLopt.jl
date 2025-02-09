using NLopt

function my_objective_fn(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5 / sqrt(x[2])
    end
    return sqrt(x[2])
end

function my_constraint_fn(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 3 * a * (a * x[1] + b)^2
        grad[2] = -1
    end
    return (a * x[1] + b)^3 - x[2]
end

opt = NLopt.Opt(:LD_MMA, 2)

NLopt.lower_bounds!(opt, [-Inf, 0.0])

NLopt.xtol_rel!(opt, 1e-4)
NLopt.min_objective!(opt, my_objective_fn)

NLopt.inequality_constraint!(opt, (x, g) -> my_constraint_fn(x, g, 2, 0), 1e-8)
NLopt.inequality_constraint!(opt, (x, g) -> my_constraint_fn(x, g, -1, 1), 1e-8)

min_f, min_x, ret = NLopt.optimize(opt, [1.234, 5.678])

num_evals = NLopt.numevals(opt)

println(
    """
    objective value       : $min_f
    solution              : $min_x
    solution status       : $ret
    # function evaluation : $num_evals
    """
)

