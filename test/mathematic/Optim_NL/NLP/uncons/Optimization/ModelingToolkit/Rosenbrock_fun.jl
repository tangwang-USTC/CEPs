using Optimization, Zygote

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2

function rosenbrock(x, p)

    x1 = x[1]
    x2 = x[2]
    return (p[1] - x1)^2 + p[2] * (x2 - x1^2)^2
end

function rosenbrock(x, p)

    x1 = x[1]
    x2 = x[2]
    # out1 = p[1] - (sum_kbn(x) - x2)

    x12 = [x1^2]

    # x12 = [0.0]
    # xx1122!(x12,x1)
    # @show x1,x12
    out1 = p[1] - x1
    out2 = x2 - sum(x12)
    # @show typeof(out1)
    return (out1)^2 + p[2] * (out2)^2
end

#### However, the following form is unavailability
function rosenbrock3(x, p)

    x1 = x[1]
    x2 = x[2]
    out = zeros(2)
    out[1] = p[1] - x1
    out[2] = x2 - x1^2
    @show typeof(out[1])
    return (out[1])^2 + p[2] * (out[2])^2
end

function xx1122!(x12,x)

    x12[1] = x^2
    @show affs = x12
end

x0 = zeros(2)
p = [1.0, 100.0]

optf = OptimizationFunction(rosenbrock, AutoZygote()) 
prob = Optimization.OptimizationProblem(optf, x0, p)

res = solve(prob, Optimization.LBFGS())

@show res.u
@show prob.f(res.u,p)

