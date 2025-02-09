

using BenchmarkTools, LeastSquaresOptim
using Optimization
# using Krylov
using LinearSolve
using StaticArrays    # length(vector) ≤ 20 or 100, now static arrays can only be used for sufficiently small arrays



f(u, p) = u .* u .- p
f(u, p) = [u[1] * u[1] - p, u[2] * u[2] - p]
# f(u, p) = norm([u[1] * u[1] - p, u[2] * u[2] - p])
# f(u, p) = norm([u[1] * u[1] - 2.0, u[2] * u[2] - 2.0])

# f(u, p) = (u[1] * u[1] - p)^2 + (u[2] * u[2] - p)^2

p = 2.0
u0 = [1.0, 1.0]


ubs = [2.0, 2.0]
lbs = [1.0, 1.0]


optf = OptimizationFunction(f, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, u0, p)
# prob = OptimizationProblem(optf, u0)

solf = Optimization.solve(prob, Optimization.LBFGS())

