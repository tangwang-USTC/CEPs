using ModelingToolkit, Optimization, OptimizationOptimJL

@variables begin
    x = 0.14, [bounds = (-2.0, 2.0)]
    y = 0.14, [bounds = (-1.0, 3.0)]
end
@parameters a=1.0 b=100.0
rosenbrock = (a - x)^2 + b * (y - x^2)^2
cons = [
    x^2 + y^2 ≲ 1
]
@mtkbuild sys = OptimizationSystem(rosenbrock, [x, y], [a, b], constraints = cons)
prob = OptimizationProblem(sys, [], grad = true, hess = true, cons_j = true, cons_h = true)
u_opt = solve(prob, IPNewton())

