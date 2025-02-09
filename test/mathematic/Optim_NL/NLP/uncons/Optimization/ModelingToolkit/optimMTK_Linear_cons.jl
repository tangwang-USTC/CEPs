using Optimization, OptimizationMOI, ModelingToolkit, HiGHS, LinearAlgebra

@variables u[1:5] [bounds = (0.0, 100.0)]
@variables v[1:3] [bounds = (0.0, Inf)]
@variables w[1:5] [bounds = (0.0, Inf)]
@variables m [bounds = (0.0, Inf)]

cons = [u[1] + v[1] - w[1] ~ 150 # January
        u[2] + v[2] - w[2] - 1.01u[1] + 1.003w[1] ~ 100 # February
        u[3] + v[3] - w[3] - 1.01u[2] + 1.003w[2] ~ -200 # March
        u[4] - w[4] - 1.02v[1] - 1.01u[3] + 1.003w[3] ~ 200 # April
        u[5] - w[5] - 1.02v[2] - 1.01u[4] + 1.003w[4] ~ -50 # May
        -m - 1.02v[3] - 1.01u[5] + 1.003w[5] ~ -300]

@named optsys = OptimizationSystem(m, [u..., v..., w..., m], [], constraints = cons)
optsys = complete(optsys)
optprob = OptimizationProblem(optsys,
    vcat(fill(0.0, 13), 300.0);
    grad = true,
    hess = true,
    sense = Optimization.MaxSense)
sol = solve(optprob, HiGHS.Optimizer())

