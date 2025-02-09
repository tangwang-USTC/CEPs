

using Optimization, Zygote

function rosenbrockvec(x, p)
    # @show typeof(x)
    return [p[1] - x[1], (p[2])^0.5 * (x[2] - x[1]^2)]
end

function rosenbrockvec(x, p;out=[1.0,2.0])
    # @show typeof(x)
    out[1] = p[1] - x[1]
    out[1] = (p[2])^0.5 * (x[2] - x[1]^2)
    return out
end

function rosenbrockvec(x, p)
    # @show typeof(x)
    f1(x) = p[1] - x[1]
    f2(x) = (p[2])^0.5 * (x[2] - x[1]^2)
    return [f1(x),f2(x)]
end

function rosenbrock(x, p)
    # @show typeof(x)
    return (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
end

rosenbrock(x, p) = norm(rosenbrockvec(x, p))
# rosenbrock(x, p) = rosenbrockvec(x, p)

x0 = zeros(2)
p = [0.2, 0.5]

optf = OptimizationFunction(rosenbrock, AutoZygote())
prob = Optimization.OptimizationProblem(optf, x0, p)

sol = solve(prob, Optimization.LBFGS())


if 1 == 2
    function con2_c(res, x, p)
        res .= [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1]) - 5]
    end
    
    optfc = OptimizationFunction(rosenbrock, AutoZygote(), cons = con2_c)
    probc = OptimizationProblem(optfc, x0, p, lcons = [1.0, -Inf],
        ucons = [1.0, 0.0], lb = [-1.0, -1.0],ub = [1.0, 1.0])
    resc = solve(probc, Optimization.LBFGS(), maxiters = 100)
    
    
    
    prob2 = OptimizationProblem(optf, x0, p; lb = [-1.0, -1.0],ub = [1.0, 1.0])
    res2 = solve(prob2, Optimization.LBFGS(), maxiters = 100)

end

fff(x) = x + x.^2 + x.^3

function fff2(x)

    ff431(x) = x
    # for i in 2:3
    #     ff43(x) += x.^i
    # end
    ff43(x) = ff431(x) + x.^2
    return ff43(x)
end




