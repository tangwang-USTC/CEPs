

using ModelingToolkit, NonlinearSolve
using BenchmarkTools

@variables x y z
@parameters σ ρ β

# Define a nonlinear system
eqs = [0 ~ σ * (y - x), 0 ~ x * (ρ - z) - y, 0 ~ x * y - β * z]
@mtkbuild ns = NonlinearSystem(eqs, [x, y, z], [σ, ρ, β])

equations(ns)
calculate_jacobian(ns)
unknowns(ns)
parameters(ns)
observed(ns)

u0 = [x => 1.0, y => 0.0, z => 0.0]

ps = [σ => 10.0
      ρ => 26.0
      β => 8 / 3]

prob = NonlinearProblem(ns, u0, ps)
println("NP")
@benchmark sol = solve(prob, NewtonRaphson())

prob = NonlinearProblem(ns, u0, ps, jac = true)
println("NP_jac")
@benchmark sol = solve(prob, NewtonRaphson())

