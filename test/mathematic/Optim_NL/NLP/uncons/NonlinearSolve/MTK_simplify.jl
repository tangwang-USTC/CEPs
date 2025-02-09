

using ModelingToolkit, NonlinearSolve
using BenchmarkTools

@variables u1 u2 u3 u4 u5

# Define a nonlinear system
eqs = [0 ~ u1 - sin(u5), 0 ~ u2 - cos(u1), 0 ~ u3 - hypot(u1, u2),
       0 ~ u4 - hypot(u2, u3), 0 ~ u5 - hypot(u4, u1)]

@mtkbuild ns = NonlinearSystem(eqs, [u1, u2, u3, u4, u5], [])

equations(ns)
calculate_jacobian(ns)
unknowns(ns)
parameters(ns)
observed(ns)



u0 = [u5 .=> 1.0]

prob = NonlinearProblem(ns, u0)

sol = solve(prob, NewtonRaphson())

println("NP")
@benchmark sol = solve(prob, NewtonRaphson())

prob = NonlinearProblem(ns, u0, jac = true)
println("NP_jac")
@benchmark sol = solve(prob, NewtonRaphson())



@named nss = NonlinearSystem(eqs, [u1, u2, u3, u4, u5], [])
nss = structural_simplify(nss)

equations(nss)
calculate_jacobian(nss)
unknowns(nss)
parameters(nss)
observed(nss)

u0 = [u5 .=> 1.0]

probs = NonlinearProblem(nss, u0)
println("NP_simplify")
@benchmark sol = solve(probs, NewtonRaphson())

probs = NonlinearProblem(nss, u0, jac = true)
println("NP_simplify_jac")
@benchmark sol = solve(probs, NewtonRaphson())
println("NP_simplify_SNR_jac")
@benchmark sol = solve(probs, SimpleNewtonRaphson())

sols = solve(probs, NewtonRaphson())
# We can then use symbolic indexing to retrieve any variable:
sol[u4]
