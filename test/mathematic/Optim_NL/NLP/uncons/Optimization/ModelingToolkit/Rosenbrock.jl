using Optimization, OptimizationOptimJL, OptimizationEvolutionary
rosenbrock(x, p) = (1 - x[1])^2 + p * (x[2] - x[1]^2)^2
cons(res, x, p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[2]])
p = [big(1), 100]
p = 100

rosenbrock2(x,p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
cons2(res,x,p) = (res .= [x[1]^2 + x[2]^2, x[1] * x[2]])
function jacob_rose(x,p)
    [- 2(p[1] - x[1]) + -4p[2] * x[1] * (x[2] - x[1]^2) , 2p[2] * (x[2] - x[1]^2)]
end

x0 = zeros(2)

optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff(), cons = cons)
prob = OptimizationProblem(optprob, x0, p, lcons = [-Inf, -1.0], ucons = [0.8, 2.0])
sol1 = solve(prob, IPNewton())

########
prob2 = OptimizationProblem(rosenbrock, x0, p)
# prob2 = OptimizationProblem(rosenbrock, x0, p)
sol = solve(prob2, NelderMead())
