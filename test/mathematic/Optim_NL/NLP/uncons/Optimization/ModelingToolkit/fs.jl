using Optimization, OptimizationOptimJL, OptimizationEvolutionary
using NLopt, OptimizationPolyalgorithms

function eqs2(x,p;pp=2)
    
    z = zero.(x)
    z[1] = x[1] + x[2] - 5.0
    z[2] = x[1] * x[2] - 6.0
    return z
end
eqs2(x,p) = (x[1] + x[2] - 5.0)^2 + (x[1] * x[2] - 6.0)^2
eqs2(x,p) = [(x[1] + x[2] - 5.0)^2, (x[1] * x[2] - 6.0)^2]
eqs2(x,p) = [x[1] + x[2] - 5.0, x[1] * x[2] - 6.0]
eqs2(x,p) = [abs(x[1] + x[2] - 5.0), abs(x[1] * x[2] - 6.0)]
p = 100

x0 = zeros(2)
x0 = [1.8, 2.7]

optprob = OptimizationFunction(eqs2, Optimization.AutoEnzyme())
prob = OptimizationProblem(optprob, x0, p)

# sol1 = solve(prob, IPNewton())
# sol = solve(prob, SAMIN())

sol = solve(prob, NelderMead())
sol = solve(prob, SimulatedAnnealing())
sol = solve(prob, ParticleSwarm())

sol = solve(prob, Optimization.LBFGS())
sol = solve(prob, Optim.LBFGS())
sol = solve(prob, Optim.BFGS())

sol = solve(prob, Optim.KrylovTrustRegion())
sol = solve(prob, Optim.Newton())
sol = solve(prob, Optim.NewtonTrustRegion())

sol = solve(prob, Optim.NGMRES())
sol = solve(prob, Optim.OACCEL())

sol = solve(prob, Optim.ConjugateGradient())
sol = solve(prob, Optim.GradientDescent())

sol = solve(prob, PolyOpt())
sol = solve(prob, ())
sol = solve(prob, ())
