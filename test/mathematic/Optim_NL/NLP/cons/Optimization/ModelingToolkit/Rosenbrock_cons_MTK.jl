using Optimization, Zygote
using OptimizationMOI, Ipopt
using OptimizationNLopt, ModelingToolkit

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



function con2_c!(res, x, p)
    res .= [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1]) - p[3]]
    return nothing
end


function con2_c!(res, x, p)
    res .= [sum(x[1:2] .^2), 
        (x[2] * sin(x[1]) + x[1]) - p[3]]
    return nothing
end

function con2_c(x, p)
    return [x[1]^2 + x[2]^2, (x[2] * sin(x[1]) + x[1]) - p[3]]
end

x0 = zeros(2)
p = [1.0, 100.0, 5.0]

ADtype = AutoZygote()              # cons_fun!(res::Zygote.Buffer{Float64, Vector{Float64}}, xx::Vector{Float64}, p::Vector{Float64}))
ADtype = AutoReverseDiff()
ADtype = AutoForwardDiff()         # (default) cons_fun!(res::Vector{ForwardDiff.Dual{…}}, xx::Vector{ForwardDiff.Dual{…}}, p::Vector{Float64})

# ADtype = AutoEnzyme()
# # ADtype = AutoTracker()
# ADtype = AutoModelingToolkit()

optf = OptimizationFunction(rosenbrock, ADtype; cons = con2_c!)
prob = OptimizationProblem(optf, x0, p; lcons = [1.0, -Inf],
    ucons = [1.0, 0.0], lb = [-1.0, -1.0], ub = [1.0, 1.0])

res = solve(prob, Optimization.LBFGS(); maxiters = 100)
# res = solve(prob, Ipopt.Optimizer(); maxiters = 100)
# res = solve(prob, Opt(:LD_LBFGS, 2))

@show res.u
@show prob.f(res.u,p)
res2 = zeros(2)
con2_c!(res2,res.u,p)
@show res2

22