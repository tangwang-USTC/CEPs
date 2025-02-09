using Optim

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

res = optimize(rosenbrock, zeros(2), Optim.BFGS())

@show Optim.minimizer(res)

@show Optim.minimum(result)
