using JSOSolvers     # solvers: lbfgs, tron, trunk, R2, fomo


using JSOSolvers, ADNLPModels
nlp = ADNLPModel(x -> sum(x.^2), ones(3));


x0 = [0.0]
output = JSOSolvers.lbfgs(nlp;x=x0,atol=1e-10)


solver = LBFGSSolver(nlp;mem = 5);
output2 = solve!(solver, nlp;x=x0,atol=1e-10)


