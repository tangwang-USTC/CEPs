

using NLSolvers     # Without AD 

function objective(x)
    fx = x^4 + sin(x)
end
function gradient(∇f, x)
    ∇f = 4x^3 + cos(x)
    return ∇f
end
objective_gradient(∇f, x) = objective(x), gradient(∇f, x)
function hessian(∇²f, x)
    ∇²f = 12x^2 - sin(x)
    return ∇²f
end
function objective_gradient_hessian(∇f, ∇²f, x)
    f, ∇f = objective_gradient(∇f, x)
    ∇²f = hessian(∇²f, x)
    return f, ∇f, ∇²f
end
scalarobj = ScalarObjective(f=objective,
                            g=gradient,
                            fg=objective_gradient,
                            fgh=objective_gradient_hessian,
                            h=hessian)
optprob = OptimizationProblem(scalarobj; inplace=false) # scalar input, so not inplace

solve(optprob, 0.3, LineSearch(Newton()), OptimizationOptions())


