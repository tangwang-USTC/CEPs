using JSOSolvers     # solvers: tron, trunk


using ADNLPModels

F(x) = [x[1] - 1.0; 10 * (x[2] - x[1]^2)]

nequ = 2
x0 = [-1.2; 1.0]
nls = ADNLSModel(F, x0, nequ, zeros(2), 0.5 * ones(2))




output = tron(nls;x=x0,atol=1e-10)


# solver = TronSolverNLS(nls)
# output = solve!(solver, nls;x=x0,atol=1e-10)



